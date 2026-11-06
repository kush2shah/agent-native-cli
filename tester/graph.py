"""Top-level graph, M2 slice: prepare → intake → propose ⇄ review.

    prepare → intake (interrupt) → propose ─errors, auto-fix─┐
                                      ▲   ◀──────────────────┘
                                      │ feedback
                                   review (interrupt) ─approve→ END
                                      ▲ │ edited on disk
                                      └─┘

Interrupt nodes re-run from the top on resume, so nothing with side effects
happens before `interrupt()` in them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from tester.classify import DESTRUCTIVE, UNKNOWN
from tester.config import Config
from tester.docs import DocsIndex
from tester.inventory import Inventory
from tester.plan import NAME_ARG, Plan, dump_plan, load_plan, validate_plan

PROMPTS = Path(__file__).parent / "prompts"
MAX_AUTO_FIX = 2  # propose retries on validation errors before showing the reviewer

INTAKE_QUESTIONS = [
    {
        "key": "flows",
        "question": "Which flows matter most? (comma-separated)",
        "default": "debugging failing agent runs in a tracing project, building an evaluation dataset, "
        "comparing experiments, finding and reading prompts",
    },
    {
        "key": "good",
        "question": "What does 'good' look like for an agent using the CLI?",
        "default": "finds the right command without guessing, uses JSON output, recovers from errors on its own, "
        "few help lookups",
    },
    {
        "key": "writes",
        "question": "May tasks create objects? (reads-only / create-datasets)",
        "default": "create-datasets",
    },
    {
        "key": "notes",
        "question": "Anything to focus on or avoid?",
        "default": "",
    },
]


class PlanState(TypedDict, total=False):
    run_id: str
    inventory_path: str
    snapshot_dir: str
    intake: dict
    plan: dict
    errors: list[str]
    feedback: list[str]  # every reviewer comment so far, oldest first
    revision: int
    attempts: int  # propose calls since the last human feedback
    approved: bool
    next: str  # routing after review: propose | review | done
    notice: str  # message for the reviewer (e.g. why approve was refused)


@dataclass
class Deps:
    cfg: Config
    model: Any  # chat model with .with_structured_output()


@lru_cache(maxsize=4)
def _docs_index(snapshot_dir: str) -> DocsIndex:
    return DocsIndex(Path(snapshot_dir))


def run_dir(deps: Deps, state: PlanState) -> Path:
    d = deps.cfg.runs_dir / state["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- context for propose -------------------------------------------------------


def inventory_digest(inv: Inventory) -> str:
    lines = []
    for c in inv.data["commands"]:
        if not c["path"] or not c.get("runnable"):
            continue
        cls = c.get("class", UNKNOWN)
        tag = "DENIED" if cls in (DESTRUCTIVE, UNKNOWN) else cls
        probe = f" (probe: {c['probe']})" if c.get("probe") not in (None, "not_probed") else ""
        lines.append(f"- `{c['path']}` [{tag}]{probe}: {(c.get('summary') or '')[:110]}")
    return "\n".join(lines)


def docs_catalog(ix: DocsIndex) -> str:
    return "\n".join(f"- {p['title']}: {p['description'][:90]} <{p['url']}>" for p in ix.catalog())


def docs_excerpts(ix: DocsIndex, intake: dict, per_query: int = 3, max_queries: int = 6) -> str:
    queries = [q.strip() for q in re.split(r"[,;\n]", intake.get("flows", "")) if q.strip()][:max_queries]
    if intake.get("notes"):
        queries.append(intake["notes"])
    seen, parts = set(), []
    for q in queries:
        for hit in ix.search(q, k=per_query):
            key = (hit["url"], hit["heading"])
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"### {hit['title']} › {hit['heading']}\n<{hit['url']}>\n\n{hit['text'][:1200]}")
    return "\n\n".join(parts)


def build_context(state: PlanState, inv: Inventory, ix: DocsIndex) -> str:
    intake = state.get("intake", {})
    sections = [
        "## Intake\n" + "\n".join(f"- {q['question']} {intake.get(q['key']) or '(no answer)'}" for q in INTAKE_QUESTIONS),
        f"## CLI commands ({inv.data.get('cli_version')}; the worker never sees this list)\n{inventory_digest(inv)}",
        f"Write commands a task may allow: {', '.join(sorted(NAME_ARG))}",
        f"## Docs catalog\n{docs_catalog(ix)}",
        f"## Docs sections matching the intake's flows\n{docs_excerpts(ix, intake)}",
    ]
    if state.get("plan"):
        sections.append(f"## Previous plan (revision {state.get('revision', 0)})\n```yaml\n{dump_plan(state['plan'], {})}```")
    if state.get("feedback"):
        sections.append("## Reviewer feedback (latest last)\n" + "\n".join(f"- {f}" for f in state["feedback"]))
    if state.get("errors"):
        sections.append("## Validation errors to fix\n" + "\n".join(f"- {e}" for e in state["errors"]))
    return "\n\n".join(sections)


# --- graph -------------------------------------------------------------------------


def build_graph(deps: Deps, checkpointer=None):
    system = (PROMPTS / "generator.md").read_text()

    def prepare(state: PlanState) -> PlanState:
        missing = [k for k in ("inventory_path", "snapshot_dir") if not Path(state.get(k) or "/nonexistent").exists()]
        if missing:
            raise ValueError(f"missing {', '.join(missing)}; the CLI driver resolves these before starting")
        return {}

    def intake(state: PlanState) -> PlanState:
        answers = state.get("intake") or interrupt({"type": "intake", "questions": INTAKE_QUESTIONS})
        (run_dir(deps, state) / "intake.json").write_text(json.dumps(answers, indent=2))
        return {"intake": answers}

    def propose(state: PlanState) -> PlanState:
        inv = Inventory.load(Path(state["inventory_path"]))
        ix = _docs_index(state["snapshot_dir"])
        messages = [("system", system), ("user", build_context(state, inv, ix))]
        structured = deps.model.with_structured_output(Plan, method="json_schema", include_raw=True)
        out = structured.invoke(messages)
        _log_usage(run_dir(deps, state), state, out.get("raw"))

        revision = state.get("revision", 0) + 1
        if out.get("parsed") is None:
            plan = state.get("plan") or {}
            errors = [f"model output didn't parse: {out.get('parsing_error')}"]
        else:
            plan = out["parsed"].model_dump()
            urls = {p["url"] for p in ix.catalog()}
            errors = validate_plan(plan, inv, urls)
        meta = {"run_id": state["run_id"], "revision": revision, "inventory": state["inventory_path"],
                "docs_snapshot": state["snapshot_dir"], "status": "draft"}
        (run_dir(deps, state) / "plan.yaml").write_text(dump_plan(plan, meta))
        return {"plan": plan, "errors": errors, "revision": revision, "attempts": state.get("attempts", 0) + 1,
                "notice": ""}

    def after_propose(state: PlanState) -> str:
        if state["errors"] and state.get("attempts", 0) <= MAX_AUTO_FIX:
            return "propose"
        return "review"

    def review(state: PlanState) -> PlanState:
        plan_path = run_dir(deps, state) / "plan.yaml"
        decision = interrupt({
            "type": "review", "plan_path": str(plan_path), "plan": state["plan"], "errors": state["errors"],
            "revision": state.get("revision", 0), "notice": state.get("notice", ""),
        })
        action = (decision or {}).get("action")
        if action == "approve":
            if state["errors"]:
                return {"next": "review", "notice": "Can't approve while the plan has validation errors."}
            meta = {"run_id": state["run_id"], "revision": state.get("revision", 0),
                    "inventory": state["inventory_path"], "docs_snapshot": state["snapshot_dir"], "status": "approved"}
            plan_path.write_text(dump_plan(state["plan"], meta))
            return {"approved": True, "next": "done", "notice": ""}
        if action == "revise" and decision.get("feedback", "").strip():
            return {"feedback": [*state.get("feedback", []), decision["feedback"].strip()], "attempts": 0,
                    "next": "propose", "notice": ""}
        if action == "edited":
            plan = load_plan(plan_path.read_text())
            inv = Inventory.load(Path(state["inventory_path"]))
            urls = {p["url"] for p in _docs_index(state["snapshot_dir"]).catalog()}
            return {"plan": plan, "errors": validate_plan(plan, inv, urls), "next": "review",
                    "notice": "Reloaded your edits from plan.yaml."}
        return {"next": "review", "notice": f"Unrecognized review action: {decision!r}"}

    def after_review(state: PlanState) -> str:
        return {"done": END, "propose": "propose"}.get(state.get("next", ""), "review")

    g = StateGraph(PlanState)
    g.add_node("prepare", prepare)
    g.add_node("intake", intake)
    g.add_node("propose", propose)
    g.add_node("review", review)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "intake")
    g.add_edge("intake", "propose")
    g.add_conditional_edges("propose", after_propose, ["propose", "review"])
    g.add_conditional_edges("review", after_review, ["propose", "review", END])
    return g.compile(checkpointer=checkpointer)


def _log_usage(run_dir: Path, state: PlanState, raw) -> None:
    usage = getattr(raw, "usage_metadata", None) or {}
    rec = {"node": "propose", "revision": state.get("revision", 0) + 1,
           "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
    with (run_dir / "llm_usage.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
