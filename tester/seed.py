"""Seed data: turn each plan seed step's prose `spec` into exact data, then create it.

The plan describes seed objects in words so a human can review them. Before
anything is created, `materialize` asks the model for the exact examples, runs,
or prompt messages (one structured call per step) and writes them to
`seed_data.json`, which the reviewer confirms (and can edit) before seeding.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from tester.harness import HarnessExecutor

# JSON payloads are strings: free-form dicts don't fit strict structured output.


class PromptMessage(BaseModel):
    role: Literal["system", "human", "ai"]
    content: str = Field(description="Message text; use {variable} for input variables")


class PromptData(BaseModel):
    description: str
    messages: list[PromptMessage]


class ExampleData(BaseModel):
    inputs_json: str = Field(description='JSON object, e.g. {"question": "..."}')
    outputs_json: str = Field(description="JSON object with the reference output")


class DatasetData(BaseModel):
    description: str
    examples: list[ExampleData]


class Score(BaseModel):
    key: str = Field(description="Feedback key, e.g. correctness, user_rating")
    score: float
    comment: str = Field(default="", description='Optional comment, or ""')


class ChildRunData(BaseModel):
    name: str
    run_type: Literal["tool", "chain", "llm", "retriever"]
    inputs_json: str
    outputs_json: str = Field(description='JSON object, or "" if the run errored')
    error: str = Field(description='Error message, or "" if it succeeded')
    start_after_s: float = Field(description="Seconds after the parent run started")
    latency_s: float


class RunData(BaseModel):
    name: str
    run_type: Literal["chain", "llm", "tool"]
    inputs_json: str
    outputs_json: str = Field(description='JSON object, or "" if the run errored')
    error: str = Field(description='Error message, or "" if it succeeded')
    start_offset_s: float = Field(description="How many seconds before now the run started")
    latency_s: float
    children: list[ChildRunData] = Field(default_factory=list)
    metadata_json: str = Field(default="", description='JSON object of run metadata, or ""')
    tags: list[str] = Field(default_factory=list)
    feedback: list[Score] = Field(default_factory=list, description="Feedback scores attached to this root run")
    repeat: int = Field(default=1, description="Create this many copies (for volume); copy k starts "
                                               "k * repeat_spacing_s seconds earlier than the first")
    repeat_spacing_s: float = Field(default=60.0)


class ProjectData(BaseModel):
    description: str
    runs: list[RunData]


class EnvironmentData(BaseModel):
    environment: Literal["local", "development", "staging", "production"]
    runs: list[RunData] = Field(description="Root runs traced to this environment (may be empty)")


class AgentData(BaseModel):
    display_name: str = Field(description="Human-readable agent name")
    environments: list[EnvironmentData] = Field(description="Environments that receive traces; all four exist anyway")


class ExperimentRow(BaseModel):
    example_index: int = Field(description="0-based index into the dataset's examples, in seed order")
    outputs_json: str = Field(description='What the system under test answered (JSON object), or "" on error')
    error: str = Field(default="", description='Error message, or ""')
    latency_s: float
    scores: list[Score] = Field(default_factory=list, description="Evaluator scores for this row")


class ExperimentData(BaseModel):
    description: str
    dataset: str = Field(description="Name of the seeded dataset this experiment ran against (with <run_id>-)")
    rows: list[ExperimentRow]


SCHEMAS = {"prompt": PromptData, "dataset": DatasetData, "project_with_runs": ProjectData,
           "agent": AgentData, "experiment": ExperimentData}
MAX_RUNS_PER_STEP = 300  # counting repeats and child runs; keeps seeding to a few minutes
MAX_RUN_AGE_S = 23 * 3600  # LangSmith rejects runs that start more than 24 hours in the past

MATERIALIZE_SYSTEM = """You turn a test plan's description of a seed object into the exact data that will be \
created in a LangSmith workspace. Follow the description exactly: the same number of items, the same names, \
values, error messages, latencies, and ordering in time. Tasks check their answers against these details, so \
don't add, drop, or embellish anything the description specifies. Where it leaves something open, choose \
plain, realistic values. JSON fields must be valid JSON objects (or "" where allowed)."""


def materialize_step(model, step: dict, plan: dict) -> tuple[dict, object]:
    """Returns (data, raw model message) for one seed step."""
    schema = SCHEMAS[step["kind"]]
    related = [t for t in plan.get("tasks", []) if step["name"] in t.get("prompt", "")] or plan.get("tasks", [])
    others = "\n".join(f"- {o['kind']} {o['name']}: {o['spec'][:300]}" for o in plan.get("seed", []) if o is not step)
    checks = "\n".join(f"- {t['id']}: {t['prompt']}\n  checks: " + "; ".join(
        f"{c['method']} {c.get('values')}" for c in t.get("hard_checks", [])) for t in related)
    user = (f"Seed object: {step['kind']} named {step['name']}\n\nDescription:\n{step['spec']}\n\n"
            f"Other seed objects in this plan (for references, e.g. an experiment's dataset):\n{others or '(none)'}\n\n"
            f"Tasks that use it (their checks must hold against your data):\n{checks or '(none)'}")
    out = model.with_structured_output(schema, method="json_schema", include_raw=True).invoke(
        [("system", MATERIALIZE_SYSTEM), ("user", user)])
    if out.get("parsed") is None:
        raise ValueError(f"seed {step['id']}: model output didn't parse: {out.get('parsing_error')}")
    return out["parsed"].model_dump(), out.get("raw")


def all_runs(kind: str, data: dict) -> list[dict]:
    if kind == "project_with_runs":
        return data["runs"]
    if kind == "agent":
        return [r for env in data["environments"] for r in env["runs"]]
    return []


def run_count(kind: str, data: dict) -> int:
    if kind == "experiment":
        return len(data["rows"])
    return sum(r.get("repeat", 1) * (1 + len(r.get("children", []))) for r in all_runs(kind, data))


def validate_seed_data(seed_data: list[dict]) -> list[str]:
    """Problems that block seeding (bad JSON, empty objects, dangling references)."""
    errors = []
    datasets = {s["name"]: s for s in seed_data if s["kind"] == "dataset"}
    order = [s["name"] for s in seed_data]
    for s in seed_data:
        where = f"seed {s['id']} ({s['name']})"
        try:
            data = SCHEMAS[s["kind"]].model_validate(s["data"]).model_dump()
        except Exception as e:
            errors.append(f"{where}: {e}")
            continue
        if s["kind"] == "experiment":
            ds = datasets.get(data["dataset"])
            if not ds:
                errors.append(f"{where}: dataset {data['dataset']!r} isn't a seeded dataset")
            elif order.index(ds["name"]) > order.index(s["name"]):
                errors.append(f"{where}: dataset {data['dataset']!r} must be seeded before the experiment")
            else:
                n = len(ds["data"]["examples"])
                bad = sorted({r["example_index"] for r in data["rows"] if not 0 <= r["example_index"] < n})
                if bad:
                    errors.append(f"{where}: example_index {bad} out of range (dataset has {n} examples)")
        items = (data.get("examples") or data.get("messages") or data.get("rows")
                 or all_runs(s["kind"], data) or [])
        if not items and s["kind"] != "agent":
            errors.append(f"{where}: no examples, runs, rows, or messages")
        oldest = max((r["start_offset_s"] + (max(1, r.get("repeat", 1)) - 1) * r.get("repeat_spacing_s", 60)
                      for r in all_runs(s["kind"], data)), default=0)
        if oldest > MAX_RUN_AGE_S:
            errors.append(f"{where}: a run starts {oldest / 3600:.1f} hours ago; LangSmith only accepts runs from "
                          f"the last 24 hours, so keep every run within {MAX_RUN_AGE_S // 3600} hours")
        if run_count(s["kind"], data) > MAX_RUNS_PER_STEP:
            errors.append(f"{where}: {run_count(s['kind'], data)} runs; at most {MAX_RUNS_PER_STEP} per seed step")
        payloads = []
        for item in items:
            payloads += [item.get("inputs_json"), item.get("outputs_json"), item.get("metadata_json")]
            for child in item.get("children", []):
                payloads += [child["inputs_json"], child["outputs_json"]]
        for p in payloads:
            if p is None or p == "":
                continue
            try:
                if not isinstance(json.loads(p), dict):
                    raise ValueError
            except ValueError:
                errors.append(f"{where}: not a JSON object: {p[:80]!r}")
    return errors


def check_warnings(seed_data: list[dict], plan: dict) -> list[str]:
    """Hard-check values that appear nowhere in the seed data. Not blocking: some
    values describe the answer ("no experiments"), not the data."""
    blob = json.dumps([[s["name"], s["data"]] for s in seed_data]).lower()
    warnings = []
    for t in plan.get("tasks", []):
        for c in t.get("hard_checks", []):
            if not c["method"].startswith("output_"):
                continue
            hits = [v for v in c["values"] if v.lower().strip() in blob]
            if (c["method"] == "output_contains_all" and len(hits) < len(c["values"])) or not hits:
                missing = [v for v in c["values"] if v not in hits]
                warnings.append(f"{t['id']}: check '{c['describe']}' looks for {missing}, not found in seed data")
    return warnings


def apply_seed(ex: HarnessExecutor, seed_data: list[dict], run_id: str, log=print) -> None:
    real = lambda n: n.replace("<run_id>-", f"{run_id}-")  # noqa: E731
    for s in seed_data:
        name = real(s["name"])
        log(f"  seeding {s['kind']} {name} ...")
        if s["kind"] == "dataset":
            ex.create_dataset(name, s["data"])
        elif s["kind"] == "prompt":
            ex.create_prompt(name, s["data"])
        elif s["kind"] == "project_with_runs":
            ex.create_project_with_runs(name, s["data"])
        elif s["kind"] == "agent":
            ex.create_agent_with_runs(name, s["data"])
        elif s["kind"] == "experiment":
            ex.create_experiment(name, s["data"], real(s["data"]["dataset"]))
