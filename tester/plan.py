"""Plan schema (what `propose` must produce) and validation (enforced in code, not the prompt).

Object names in a plan use the literal `<run_id>-` prefix. It is replaced with the
real run ID when the plan runs (M3), so a plan can be reviewed before it has one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from tester.classify import WRITE

if TYPE_CHECKING:
    from tester.inventory import Inventory

RUN_PREFIX = "<run_id>-"

# Creates a worker may run: the flag that holds the object's name and the kind of
# object, so run_cli can enforce the name prefix and write the manifest entry first.
CREATES = {"datasets create": {"name_flag": "--name", "kind": "dataset"}}

SEED_KINDS = ("dataset", "project_with_runs", "prompt")
CHECK_METHODS = ("output_contains_all", "output_contains_any", "object_exists", "object_absent")
MIN_TASKS, MAX_TASKS = 3, 8

_NAME_REF = re.compile(r"<run_id>-[\w-]*\w")
_CLI_SYNTAX = re.compile(r"\blangsmith\s+[a-z]|(?<![\w-])--[a-z][\w-]*")


class HardCheck(BaseModel):
    describe: str = Field(description="What the check verifies, in plain words")
    method: Literal[CHECK_METHODS] = Field(
        description="output_contains_all/any: the agent's final answer contains the values. "
        "object_exists/object_absent: a workspace object named values[0] exists / doesn't."
    )
    values: list[str] = Field(default_factory=list)


class SeedStep(BaseModel):
    id: str = Field(description="s01, s02, ...")
    kind: Literal[SEED_KINDS]
    name: str = Field(description="Object name; must start with <run_id>-")
    spec: str = Field(description="What the object contains, concretely enough to create it (examples, runs, errors, ...)")


class Task(BaseModel):
    id: str = Field(description="t01, t02, ...")
    tier: Literal["easy", "medium", "hard", "trap"]
    prompt: str = Field(description="The goal as a user would say it, in the product's terms. No CLI syntax.")
    rationale: str = Field(description="Which agent-usability risk this task probes, in one sentence")
    allowed_writes: list[str] = Field(default_factory=list, description="Command paths this task may run that write")
    expected: Literal["complete", "ask_or_refuse", "blind_spot"]
    hard_checks: list[HardCheck] = Field(default_factory=list)
    docs_refs: list[str] = Field(default_factory=list, description="URLs of the docs pages the task's concepts come from")


class Plan(BaseModel):
    summary: str = Field(description="Two or three sentences: what this plan covers and why")
    seed: list[SeedStep]
    tasks: list[Task]
    reset_notes: str = Field(description="Anything reset needs beyond deleting manifest entries in reverse order")


def validate_plan(plan: dict, inventory: Inventory, docs_urls: set[str] | None = None) -> list[str]:
    """Problems that block approval. An empty list means the plan is runnable."""
    try:
        p = Plan.model_validate(plan)
    except Exception as e:
        return [f"plan doesn't match the schema: {e}"]

    errors: list[str] = []
    if not MIN_TASKS <= len(p.tasks) <= MAX_TASKS:
        errors.append(f"plan has {len(p.tasks)} tasks; want {MIN_TASKS}-{MAX_TASKS}")
    for label, ids in (("task", [t.id for t in p.tasks]), ("seed", [s.id for s in p.seed])):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            errors.append(f"duplicate {label} ids: {', '.join(dupes)}")

    seeded = set()
    for s in p.seed:
        if not s.name.startswith(RUN_PREFIX) or len(s.name) <= len(RUN_PREFIX):
            errors.append(f"seed {s.id}: name {s.name!r} must start with {RUN_PREFIX!r}")
        seeded.add(s.name)

    for t in p.tasks:
        where = f"task {t.id}"
        for path in t.allowed_writes:
            node = inventory.get(path)
            if path not in CREATES:
                errors.append(f"{where}: allowed_writes {path!r} isn't supported (only {', '.join(sorted(CREATES))})")
            elif not node or node.get("class") != WRITE:
                errors.append(f"{where}: allowed_writes {path!r} isn't a write command in the inventory")
        if m := _CLI_SYNTAX.search(t.prompt):
            errors.append(f"{where}: prompt contains CLI syntax ({m.group(0)!r}); write the goal, not the command")
        for ref in _NAME_REF.findall(t.prompt):
            if ref not in seeded and not t.allowed_writes:
                errors.append(f"{where}: prompt refers to {ref!r}, which no seed step creates")
        if t.expected == "complete" and not t.hard_checks:
            errors.append(f"{where}: expected=complete needs at least one hard check")
        for c in t.hard_checks:
            if c.method.startswith("object_") and len(c.values) != 1:
                errors.append(f"{where}: {c.method} takes exactly one object name in values")
            if c.method.startswith("output_") and not c.values:
                errors.append(f"{where}: {c.method} needs values")
            if c.method.startswith("object_") and c.values and not c.values[0].startswith(RUN_PREFIX):
                errors.append(f"{where}: {c.method} must check a {RUN_PREFIX!r} object, not pre-existing data")
        if docs_urls is not None:
            for ref in t.docs_refs:
                if ref not in docs_urls:
                    errors.append(f"{where}: docs_refs {ref!r} isn't a page in the docs snapshot")
    return errors


def dump_plan(plan: dict, meta: dict) -> str:
    header = "".join(f"# {k}: {v}\n" for k, v in meta.items())
    return header + yaml.safe_dump(plan, sort_keys=False, allow_unicode=True, width=100)


def load_plan(text: str) -> dict:
    return yaml.safe_load(text) or {}


# --- binding names to a run ------------------------------------------------------


def worker_prefix(run_id: str, task_id: str, arm: str, repeat: int) -> str:
    """Prefix for objects one worker run creates, so repeats and arms never collide."""
    return f"{run_id}-{task_id}-{arm}-r{repeat}-"


def bind(text: str, run_id: str, seeded: set[str], wprefix: str | None = None) -> str:
    """Replace `<run_id>-x` with real names: seeded objects get the run prefix,
    anything else (an object the task creates) gets the worker's prefix."""

    def sub(m: re.Match) -> str:
        ref = m.group(0)
        rest = ref[len(RUN_PREFIX):]
        if ref in seeded or wprefix is None:
            return f"{run_id}-{rest}"
        return f"{wprefix}{rest}"

    return _NAME_REF.sub(sub, text)


def name_from_args(path: str, args: list[str]) -> str | None:
    flag = CREATES[path]["name_flag"]
    for i, tok in enumerate(args):
        if tok == flag and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None
