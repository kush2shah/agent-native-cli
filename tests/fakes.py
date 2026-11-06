"""Fakes for offline M3 tests: an SDK client, a chat model, and the docs MCP server."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage


class LangSmithNotFoundError(Exception):
    pass


class FakeClient:
    """In-memory stand-in for langsmith.Client, covering what the harness executor calls.

    Datasets the fake CLI creates (via a file:// endpoint) are picked up from `cli_state`.
    """

    def __init__(self, cli_state: Path | None = None, runs_visible_after: int = 0):
        self.objs: dict[str, dict[str, SimpleNamespace]] = {"dataset": {}, "project": {}, "prompt": {}}
        self.examples: dict[str, list] = {}
        self.runs: list[dict] = []
        self.calls: list[str] = []
        self.cli_state = cli_state
        self._seen_cli: set[str] = set()
        self.runs_visible_after = runs_visible_after
        self._list_runs_calls = 0
        self.on_create = lambda kind, name: None

    def _sync_cli(self):
        if self.cli_state and self.cli_state.exists():
            for d in json.loads(self.cli_state.read_text()):
                if d["id"] not in self._seen_cli:
                    self._seen_cli.add(d["id"])
                    self.objs["dataset"][d["name"]] = SimpleNamespace(id=d["id"], name=d["name"])

    def add(self, kind, name):
        obj = SimpleNamespace(id=str(uuid.uuid4()), name=name, repo_handle=name)
        self.objs[kind][name] = obj
        return obj

    def _get(self, kind, name=None, oid=None):
        self._sync_cli()
        for o in self.objs[kind].values():
            if (name is not None and o.name == name) or (oid is not None and o.id == str(oid)):
                return o
        raise LangSmithNotFoundError(f"{kind} not found")

    # reads
    def read_dataset(self, dataset_name=None, dataset_id=None):
        return self._get("dataset", dataset_name, dataset_id)

    def read_project(self, project_name=None, project_id=None):
        return self._get("project", project_name, project_id)

    def get_prompt(self, name):
        try:
            return self._get("prompt", name)
        except LangSmithNotFoundError:
            return None

    def list_datasets(self):
        self._sync_cli()
        return list(self.objs["dataset"].values())

    def list_projects(self):
        return list(self.objs["project"].values())

    def list_prompts(self, limit=100, offset=0, is_public=None):
        repos = list(self.objs["prompt"].values())
        return SimpleNamespace(repos=repos[offset:offset + limit], total=len(repos))

    def list_runs(self, project_name=None):
        self._list_runs_calls += 1
        if self._list_runs_calls <= self.runs_visible_after:
            return []
        return [r for r in self.runs if r["project_name"] == project_name]

    # writes
    def create_dataset(self, name, description=None, metadata=None):
        self.calls.append(f"create_dataset {name}")
        self.on_create("dataset", name)
        return self.add("dataset", name)

    def create_examples(self, dataset_id, examples):
        self.examples.setdefault(str(dataset_id), []).extend(examples)

    def push_prompt(self, name, object=None, description=None):
        self.calls.append(f"push_prompt {name}")
        self.on_create("prompt", name)
        self.add("prompt", name)

    def create_project(self, name, description=None, metadata=None):
        self.calls.append(f"create_project {name}")
        self.on_create("project", name)
        return self.add("project", name)

    def create_run(self, **kw):
        self.runs.append(kw)

    def delete_dataset(self, dataset_id):
        self.calls.append(f"delete_dataset {dataset_id}")
        o = self._get("dataset", oid=dataset_id)
        del self.objs["dataset"][o.name]

    def delete_project(self, project_id):
        self.calls.append(f"delete_project {project_id}")
        o = self._get("project", oid=project_id)
        del self.objs["project"][o.name]

    def delete_prompt(self, name):
        self.calls.append(f"delete_prompt {name}")
        del self.objs["prompt"][self._get("prompt", name).name]


class FakeDocsServer:
    """`post` callable for DocsMCP: lists three tools, answers searches."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, url, headers, body):
        self.calls.append(body)
        method = body["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif method == "tools/list":
            result = {"tools": [
                {"name": n, "description": f"{n} tool", "inputSchema": {"type": "object", "properties": {
                    "query": {"type": "string"}}}}
                for n in ("search_docs_by_lang_chain", "query_docs_filesystem_docs_by_lang_chain", "submit_feedback")]}
        else:
            result = {"content": [{"type": "text", "text": f"docs about {body['params']['arguments']}"}]}
        return {}, "event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result})


def tool_call(name, args, i=0):
    return {"name": name, "args": args, "id": f"call_{i}_{uuid.uuid4().hex[:6]}", "type": "tool_call"}


def ai(content="", calls=(), tokens=(100, 10)):
    return AIMessage(content=content, tool_calls=list(calls),
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                     "total_tokens": sum(tokens)})


class FakeModel:
    """A chat model. `agent(messages, tools)` scripts worker turns; `structured`
    maps a schema name to the data returned for with_structured_output."""

    def __init__(self, agent=None, structured=None):
        self.agent = agent
        self.structured = structured or {}
        self.tools: list = []
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def with_structured_output(self, schema, **kwargs):
        model = self

        class Structured:
            def invoke(self, messages):
                return {"parsed": schema.model_validate(model.structured[schema.__name__]),
                        "raw": ai(tokens=(500, 50)), "parsing_error": None}

        return Structured()

    def invoke(self, messages):
        self.calls += 1
        return self.agent(messages, self.tools)


def tool_outputs(messages) -> list[str]:
    return [m.content for m in messages if isinstance(m, ToolMessage)]
