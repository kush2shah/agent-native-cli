"""Harness executor (safety rule 7): seeds and resets the target workspace.

It holds an SDK client with the target key, so it can do things the worker's
`run_cli` never may (create runs, delete projects and agents). It is deliberately narrow:
- creates only objects named with the run's prefix, each written to the manifest first
- deletes only manifest entries, never by search or pattern
- reads (exists checks, listings) are unrestricted

The client is built with explicit target credentials; nothing comes from the
harness's process environment (rule 4).
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
KINDS = ("dataset", "project", "prompt", "agent")  # what lookups and listings cover


class HarnessError(Exception):
    pass


def target_client(cfg: Config):
    from langsmith import Client

    target = cfg.target_env
    if not target.get("LANGSMITH_API_KEY"):
        raise HarnessError("TARGET_API_KEY is not set")
    return Client(
        api_url=target.get("LANGSMITH_ENDPOINT") or DEFAULT_API_URL,
        api_key=target["LANGSMITH_API_KEY"],
        workspace_id=target.get("LANGSMITH_WORKSPACE_ID") or "",  # "" stops the SDK's fallback to process vars
        auto_batch_tracing=False,  # synchronous posts: we know when seeding is done
    )


def _is_not_found(e: Exception) -> bool:
    return type(e).__name__ in ("LangSmithNotFoundError", "NotFoundError") or "404" in str(e) or "not found" in str(e).lower()


def _dotted(start: datetime, run_id: uuid.UUID) -> str:
    return start.strftime("%Y%m%dT%H%M%S%fZ") + str(run_id)


def _lookup_kind(kind: str) -> str:
    return "project" if kind == "experiment" else kind  # an experiment is a project tied to a dataset


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
        kind = _lookup_kind(kind)
        try:
            if kind == "dataset":
                return str(self.client.read_dataset(dataset_name=name).id)
            if kind == "project":
                return str(self.client.read_project(project_name=name).id)
            if kind == "prompt":
                p = self.client.get_prompt(name)
                return str(p.id) if p else None
            if kind == "agent":
                return str(self.client.agents.retrieve(name).id)
        except Exception as e:
            if _is_not_found(e):
                return None
            raise
        raise HarnessError(f"unknown kind {kind!r}")

    def exists(self, name: str, kinds: tuple[str, ...] = KINDS) -> bool:
        return any(self.lookup_id(k, name) for k in kinds)

    def listing(self) -> dict[str, dict[str, str]]:
        """Every dataset, project, private prompt, and agent in the workspace: {kind: {id: name}}."""
        out: dict[str, dict[str, str]] = {
            "dataset": {str(d.id): d.name for d in self.client.list_datasets()},
            "project": {str(p.id): p.name for p in self.client.list_projects()},
            "prompt": {},
            "agent": {str(a.id): str(a.id) for a in self.client.agents.list()},
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

    def _seeded(self, kind: str, name: str) -> bool:
        return bool(self.manifest.find(kind, name).get("seeded"))

    def _start_posting(self, kind: str, name: str) -> None:
        """Runs can't be deleted one by one, so a step that failed partway through posting
        can't be retried without duplicating runs: the run has to be reset instead."""
        entry = self.manifest.find(kind, name)
        if entry.get("posting"):
            raise HarnessError(f"{kind} {name!r} was partly seeded before a failure; retrying would duplicate "
                               f"runs. Reset this run and start a new one")
        self.manifest.update(entry, posting=True)

    def _mark_seeded(self, kind: str, name: str, **extra) -> None:
        self.manifest.update(self.manifest.find(kind, name), seeded=True, **extra)

    def create_dataset(self, name: str, data: dict, created_by: str = "seed") -> str:
        def make() -> str:
            ds = self.client.create_dataset(name, description=data.get("description") or None,
                                            metadata={"cli_tester_run": self.prefix[:-1]})
            return str(ds.id)

        ds_id = self._create("dataset", name, created_by, make)
        if not self._seeded("dataset", name):
            examples = [{"inputs": json.loads(e["inputs_json"]), "outputs": json.loads(e["outputs_json"])}
                        for e in data["examples"]]
            ids: list[str] = []
            if examples:
                resp = self.client.create_examples(dataset_id=ds_id, examples=examples)
                raw = resp.get("example_ids") if isinstance(resp, dict) else getattr(resp, "example_ids", None)
                ids = [str(i) for i in raw or []]
            self._mark_seeded("dataset", name, example_ids=ids)
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
        if self._seeded("project", name):
            return pid
        self._start_posting("project", name)
        posted = self._post_runs(data["runs"], {"project_name": name})
        self._wait_visible({"project_name": name}, sum(n for _, n in posted), timeout_s)
        self._post_feedback(posted)
        self._mark_seeded("project", name)
        return pid

    def create_agent_with_runs(self, name: str, data: dict, created_by: str = "seed", timeout_s: float = 180) -> str:
        """An agent (all four environments exist) with runs traced to the environments the data lists."""
        from langsmith.address import agent_address

        def make() -> str:
            return str(self.client.agents.create(id=name, name=data.get("display_name") or name).id)

        aid = self._create("agent", name, created_by, make)
        if self._seeded("agent", name):
            return aid
        self._start_posting("agent", name)
        sessions = {str(e.environment).lower(): str(e.tracer_session_id)
                    for e in self.client.agents.retrieve(name).environments}
        for stage in data["environments"]:
            if not stage["runs"]:
                continue
            posted = self._post_runs(stage["runs"], {"address": agent_address(name, stage["environment"])})
            self._wait_visible({"project_id": sessions[stage["environment"]]}, sum(n for _, n in posted), timeout_s)
            self._post_feedback(posted)
        self._mark_seeded("agent", name)
        return aid

    def create_experiment(self, name: str, data: dict, dataset_name: str, created_by: str = "seed",
                          timeout_s: float = 180) -> str:
        """An experiment: a project tied to a seeded dataset, one run per row, with evaluator scores."""
        ds_entry = self.manifest.find("dataset", dataset_name)
        if not ds_entry or not ds_entry.get("seeded"):
            raise HarnessError(f"experiment {name!r}: dataset {dataset_name!r} isn't seeded")
        examples = {str(e.id): e for e in self.client.list_examples(dataset_id=ds_entry["id"])}
        example_ids = ds_entry.get("example_ids") or []
        if sorted(example_ids) != sorted(examples):  # the create response had no IDs; fall back to listing order
            example_ids = list(examples)

        def make() -> str:
            p = self.client.create_project(name, description=data.get("description") or None,
                                           reference_dataset_id=ds_entry["id"],
                                           metadata={"cli_tester_run": self.prefix[:-1]})
            return str(p.id)

        pid = self._create("experiment", name, created_by, make)
        if self._seeded("experiment", name):
            return pid
        self._start_posting("experiment", name)
        now = datetime.now(timezone.utc)
        posted = []
        for i, row in enumerate(data["rows"]):
            ex_id = example_ids[row["example_index"]]
            run = {"name": "target", "run_type": "chain", "inputs_json": json.dumps(examples[ex_id].inputs),
                   "outputs_json": row["outputs_json"], "error": row.get("error", ""),
                   "latency_s": row["latency_s"], "feedback": row.get("scores", [])}
            rid = uuid.uuid4()
            start = now - timedelta(seconds=600 - i)
            self._post_run({"project_name": name}, run, rid, rid, None, start, _dotted(start, rid),
                           reference_example_id=ex_id)
            posted.append(((rid, run), 1))
        self._wait_visible({"project_name": name}, len(posted), timeout_s)
        self._post_feedback(posted)
        self._mark_seeded("experiment", name)
        return pid

    def _post_runs(self, runs: list[dict], dest: dict) -> list[tuple[tuple, int]]:
        """Post root runs (with repeats) and their children. Returns ((root_id, run), runs posted) per root."""
        now = datetime.now(timezone.utc)
        posted = []
        for run in runs:
            for k in range(max(1, int(run.get("repeat", 1)))):
                root_id = uuid.uuid4()
                start = now - timedelta(seconds=run["start_offset_s"] + k * run.get("repeat_spacing_s", 60))
                dotted = _dotted(start, root_id)
                self._post_run(dest, run, root_id, root_id, None, start, dotted)
                for child in run.get("children", []):
                    cid = uuid.uuid4()
                    cstart = start + timedelta(seconds=child["start_after_s"])
                    self._post_run(dest, child, cid, root_id, root_id, cstart, f"{dotted}.{_dotted(cstart, cid)}")
                posted.append(((root_id, run), 1 + len(run.get("children", []))))
        return posted

    def _post_run(self, dest: dict, run: dict, run_id, trace_id, parent_id, start: datetime, dotted: str,
                  **extra) -> None:
        # Optional fields are left out rather than passed as None: the SDK mutates `extra` in place.
        optional = {
            "outputs": json.loads(run["outputs_json"]) if run.get("outputs_json") else None,
            "error": run.get("error") or None,
            "tags": run.get("tags") or None,
            "extra": {"metadata": json.loads(run["metadata_json"])} if run.get("metadata_json") else None,
        }
        self.client.create_run(
            name=run["name"], inputs=json.loads(run["inputs_json"]), run_type=run["run_type"],
            id=run_id, trace_id=trace_id, parent_run_id=parent_id, dotted_order=dotted,
            start_time=start, end_time=start + timedelta(seconds=run["latency_s"]),
            **{k: v for k, v in optional.items() if v is not None}, **dest, **extra,
        )

    def _post_feedback(self, posted: list[tuple[tuple, int]]) -> None:
        for (root_id, run), _ in posted:
            for fb in run.get("feedback", []):
                self.client.create_feedback(run_id=root_id, key=fb["key"], score=fb["score"],
                                            comment=fb.get("comment") or None)

    def _wait_visible(self, where: dict, expected: int, timeout_s: float) -> None:
        """Runs are ingested asynchronously; tasks must not start before they're queryable."""
        deadline = time.monotonic() + timeout_s
        seen = 0
        while time.monotonic() < deadline:
            seen = sum(1 for _ in self.client.list_runs(**where))
            if seen >= expected:
                return
            self.sleep(3)
        raise HarnessError(f"only {seen} of {expected} seeded runs in {where} became visible in {timeout_s:.0f}s")

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
            elif kind in ("project", "experiment"):
                self.client.delete_project(project_id=obj_id)
            elif kind == "prompt":
                self.client.delete_prompt(name)
            elif kind == "agent":
                self.client.agents.delete(name)  # also deletes its environments' traces
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
