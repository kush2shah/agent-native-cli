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


class ProjectData(BaseModel):
    description: str
    runs: list[RunData]


SCHEMAS = {"prompt": PromptData, "dataset": DatasetData, "project_with_runs": ProjectData}

MATERIALIZE_SYSTEM = """You turn a test plan's description of a seed object into the exact data that will be \
created in a LangSmith workspace. Follow the description exactly: the same number of items, the same names, \
values, error messages, latencies, and ordering in time. Tasks check their answers against these details, so \
don't add, drop, or embellish anything the description specifies. Where it leaves something open, choose \
plain, realistic values. JSON fields must be valid JSON objects (or "" where allowed)."""


def materialize_step(model, step: dict, plan: dict) -> tuple[dict, object]:
    """Returns (data, raw model message) for one seed step."""
    schema = SCHEMAS[step["kind"]]
    related = [t for t in plan.get("tasks", []) if step["name"] in t.get("prompt", "")]
    checks = "\n".join(f"- {t['id']}: {t['prompt']}\n  checks: " + "; ".join(
        f"{c['method']} {c.get('values')}" for c in t.get("hard_checks", [])) for t in related)
    user = (f"Seed object: {step['kind']} named {step['name']}\n\nDescription:\n{step['spec']}\n\n"
            f"Tasks that use it (their checks must hold against your data):\n{checks or '(none)'}")
    out = model.with_structured_output(schema, method="json_schema", include_raw=True).invoke(
        [("system", MATERIALIZE_SYSTEM), ("user", user)])
    if out.get("parsed") is None:
        raise ValueError(f"seed {step['id']}: model output didn't parse: {out.get('parsing_error')}")
    return out["parsed"].model_dump(), out.get("raw")


def validate_seed_data(seed_data: list[dict]) -> list[str]:
    """Problems that block seeding (bad JSON, empty objects)."""
    errors = []
    for s in seed_data:
        where = f"seed {s['id']} ({s['name']})"
        try:
            data = SCHEMAS[s["kind"]].model_validate(s["data"]).model_dump()
        except Exception as e:
            errors.append(f"{where}: {e}")
            continue
        items = data.get("examples") or data.get("runs") or data.get("messages") or []
        if not items:
            errors.append(f"{where}: no examples, runs, or messages")
        payloads = []
        for item in items:
            payloads += [item.get("inputs_json"), item.get("outputs_json")]
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
    blob = json.dumps([s["data"] for s in seed_data]).lower()
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
    for s in seed_data:
        name = s["name"].replace("<run_id>-", f"{run_id}-")
        log(f"  seeding {s['kind']} {name} ...")
        if s["kind"] == "dataset":
            ex.create_dataset(name, s["data"])
        elif s["kind"] == "prompt":
            ex.create_prompt(name, s["data"])
        elif s["kind"] == "project_with_runs":
            ex.create_project_with_runs(name, s["data"])
