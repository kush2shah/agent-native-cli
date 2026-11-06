"""Harness executor (safety rule 7): seeds and resets the target workspace.

It holds an SDK client with the target key, so it can do things the worker's
`run_cli` never may (create runs, delete projects). It is deliberately narrow:
- creates only objects named with the run's prefix, each written to the manifest first
- deletes only manifest entries, never by search or pattern
- reads (exists checks, listings) are unrestricted

The client is built with explicit target credentials; nothing comes from the
harness's os.environ (rule 4).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from tester.config import Config
from tester.manifest import CREATED, DELETED, FAILED, INTENDED, Manifest

DEFAULT_API_URL = "https://api.smith.langchain.com"
KINDS = ("dataset", "project", "prompt")


class HarnessError(Exception):
    pass


def target_client(cfg: Config):
    from langsmith import Client

    env = cfg.target_env
    if not env.get("LANGSMITH_API_KEY"):
        raise HarnessError("TARGET_API_KEY is not set")
    return Client(
        api_url=env.get("LANGSMITH_ENDPOINT") or DEFAULT_API_URL,
        api_key=env["LANGSMITH_API_KEY"],
        workspace_id=env.get("LANGSMITH_WORKSPACE_ID") or "",  # "" stops the SDK's env fallback
        auto_batch_tracing=False,  # synchronous posts: we know when seeding is done
    )


def _is_not_found(e: Exception) -> bool:
    return type(e).__name__ in ("LangSmithNotFoundError", "NotFoundError") or "404" in str(e) or "not found" in str(e).lower()


def _dotted(start: datetime, run_id: uuid.UUID) -> str:
    return start.strftime("%Y%m%dT%H%M%S%fZ") + str(run_id)


class HarnessExecutor:
    def __init__(self, client: Any, run_id: str, manifest: Manifest, sleep: Callable[[float], None] = time.sleep):
        self.client = client
        self.prefix = f"{run_id}-"
        self.manifest = manifest
        self.sleep = sleep

    def _check_name(self, name: str) -> None:
        if not name.startswith(self.prefix) or len(name) <= len(self.prefix):
            raise HarnessError(f"refusing to create {name!r}: harness objects must start with {self.prefix!r}")

    # --- reads -------------------------------------------------------------------

    def lookup_id(self, kind: str, name: str) -> str | None:
        """ID of the object with exactly this name, or None."""
        try:
            if kind == "dataset":
                return str(self.client.read_dataset(dataset_name=name).id)
            if kind == "project":
                return str(self.client.read_project(project_name=name).id)
            if kind == "prompt":
                p = self.client.get_prompt(name)
                return str(p.id) if p else None
        except Exception as e:
            if _is_not_found(e):
                return None
            raise
        raise HarnessError(f"unknown kind {kind!r}")

    def exists(self, name: str, kinds: tuple[str, ...] = KINDS) -> bool:
        return any(self.lookup_id(k, name) for k in kinds)

    def listing(self) -> dict[str, dict[str, str]]:
        """Every dataset, project, and private prompt in the workspace: {kind: {id: name}}."""
        out: dict[str, dict[str, str]] = {
            "dataset": {str(d.id): d.name for d in self.client.list_datasets()},
            "project": {str(p.id): p.name for p in self.client.list_projects()},
            "prompt": {},
        }
        offset = 0
        while True:
            page = self.client.list_prompts(limit=100, offset=offset, is_public=False)
            for r in page.repos:
                out["prompt"][str(r.id)] = r.repo_handle
            offset += len(page.repos)
            if not page.repos or offset >= (page.total or 0):
                break
        return out

    # --- seeding -----------------------------------------------------------------

    def _create(self, kind: str, name: str, created_by: str, make: Callable[[], str]) -> str:
        """Write the manifest entry, create, confirm. Idempotent across a resume."""
        self._check_name(name)
        prior = self.manifest.find(kind, name)
        if prior and prior["status"] == CREATED and prior.get("seeded"):
            return prior["id"]
        entry = prior if prior and prior["status"] in (INTENDED, CREATED) else self.manifest.intend(kind, name, created_by)
        try:
            new_id = entry.get("id") or self.lookup_id(kind, name) or make()
        except Exception:
            self.manifest.update(entry, id=self.lookup_id(kind, name))
            if entry["id"]:
                self.manifest.update(entry, status=CREATED)
            raise
        self.manifest.update(entry, id=new_id, status=CREATED)
        return new_id

    def _mark_seeded(self, kind: str, name: str) -> None:
        self.manifest.update(self.manifest.find(kind, name), seeded=True)

    def create_dataset(self, name: str, data: dict, created_by: str = "seed") -> str:
        def make() -> str:
            ds = self.client.create_dataset(name, description=data.get("description") or None,
                                            metadata={"cli_tester_run": self.prefix[:-1]})
            return str(ds.id)

        ds_id = self._create("dataset", name, created_by, make)
        if not self.manifest.find("dataset", name).get("seeded"):
            examples = [{"inputs": json.loads(e["inputs_json"]), "outputs": json.loads(e["outputs_json"])}
                        for e in data["examples"]]
            if examples:
                self.client.create_examples(dataset_id=ds_id, examples=examples)
            self._mark_seeded("dataset", name)
        return ds_id

    def create_prompt(self, name: str, data: dict, created_by: str = "seed") -> str:
        from langchain_core.prompts import ChatPromptTemplate

        def make() -> str:
            template = ChatPromptTemplate.from_messages([(m["role"], m["content"]) for m in data["messages"]])
            self.client.push_prompt(name, object=template, description=data.get("description") or None)
            return self.lookup_id("prompt", name) or name

        pid = self._create("prompt", name, created_by, make)
        self._mark_seeded("prompt", name)
        return pid

    def create_project_with_runs(self, name: str, data: dict, created_by: str = "seed", timeout_s: float = 180) -> str:
        def make() -> str:
            p = self.client.create_project(name, description=data.get("description") or None,
                                           metadata={"cli_tester_run": self.prefix[:-1]})
            return str(p.id)

        pid = self._create("project", name, created_by, make)
        if self.manifest.find("project", name).get("seeded"):
            return pid
        now = datetime.now(timezone.utc)
        expected = 0
        for run in data["runs"]:
            root_id = uuid.uuid4()
            start = now - timedelta(seconds=run["start_offset_s"])
            dotted = _dotted(start, root_id)
            self._post_run(name, run, root_id, root_id, None, start, dotted)
            expected += 1
            for child in run.get("children", []):
                cid = uuid.uuid4()
                cstart = start + timedelta(seconds=child["start_after_s"])
                self._post_run(name, child, cid, root_id, root_id, cstart, f"{dotted}.{_dotted(cstart, cid)}")
                expected += 1
        self._wait_visible(name, expected, timeout_s)
        self._mark_seeded("project", name)
        return pid

    def _post_run(self, project: str, run: dict, run_id, trace_id, parent_id, start: datetime, dotted: str) -> None:
        outputs = json.loads(run["outputs_json"]) if run.get("outputs_json") else None
        self.client.create_run(
            name=run["name"], inputs=json.loads(run["inputs_json"]), run_type=run["run_type"],
            project_name=project, id=run_id, trace_id=trace_id, parent_run_id=parent_id, dotted_order=dotted,
            start_time=start, end_time=start + timedelta(seconds=run["latency_s"]),
            outputs=outputs, error=run.get("error") or None,
        )

    def _wait_visible(self, project: str, expected: int, timeout_s: float) -> None:
        """Runs are ingested asynchronously; tasks must not start before they're queryable."""
        deadline = time.monotonic() + timeout_s
        seen = 0
        while time.monotonic() < deadline:
            seen = sum(1 for _ in self.client.list_runs(project_name=project))
            if seen >= expected:
                return
            self.sleep(3)
        raise HarnessError(f"only {seen} of {expected} seeded runs in {project!r} became visible in {timeout_s:.0f}s")

    # --- reset -------------------------------------------------------------------

    def delete(self, entry: dict) -> str:
        """Delete one manifest entry. Returns the outcome."""
        if entry not in self.manifest.entries:
            raise HarnessError("refusing to delete an object that isn't in the manifest")
        if not entry["name"].startswith(self.prefix):
            raise HarnessError(f"manifest entry {entry['name']!r} lacks the run prefix; not deleting")
        if entry["status"] not in (INTENDED, CREATED):
            return entry["status"]
        kind, name = entry["kind"], entry["name"]
        obj_id = entry.get("id") or self.lookup_id(kind, name)
        if not obj_id:
            self.manifest.update(entry, status=FAILED, note="never created")
            return "never created"
        try:
            if kind == "dataset":
                self.client.delete_dataset(dataset_id=obj_id)
            elif kind == "project":
                self.client.delete_project(project_id=obj_id)
            elif kind == "prompt":
                self.client.delete_prompt(name)
        except Exception as e:
            if not _is_not_found(e):
                raise
        if self.lookup_id(kind, name):
            raise HarnessError(f"{kind} {name!r} still exists after delete")
        self.manifest.update(entry, status=DELETED, id=obj_id, deleted_ts=time.time())
        return "deleted"

    def reset(self) -> list[dict]:
        """Delete live manifest entries in reverse creation order."""
        results = []
        for entry in reversed(self.manifest.live()):
            try:
                outcome = self.delete(entry)
            except Exception as e:
                outcome = f"error: {e}"
            results.append({"kind": entry["kind"], "name": entry["name"], "outcome": outcome})
        return results


def compare_listings(before: dict, after: dict, prefix: str) -> dict:
    """What changed in the workspace across a run, split into ours and not ours."""
    out = {"ours_left": [], "others_added": [], "removed": []}
    for kind in KINDS:
        b, a = before.get(kind, {}), after.get(kind, {})
        for oid, name in a.items():
            if oid not in b:
                (out["ours_left"] if name.startswith(prefix) else out["others_added"]).append(f"{kind} {name}")
        for oid, name in b.items():
            if oid not in a:
                out["removed"].append(f"{kind} {name}")
    out["clean"] = not out["ours_left"] and not out["removed"]
    return out
