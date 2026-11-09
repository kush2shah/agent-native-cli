"""Top-level graph.

    prepare → intake (interrupt) → propose ─errors, auto-fix─┐
       │                              ▲   ◀──────────────────┘
       │ (approved plan given)        │ feedback
       │                           review (interrupt) ─approve→ END if plan_only
       ▼                              │ approve
    materialize ◀─────────────────────┘
       ▼
    confirm_seed (interrupt) ─stop→ END        (nothing has been created yet)
       ▼ go
    seed → run_next ⟲ (one worker per task × arm × repeat, sequential in M3)
       ▼
    confirm_reset (interrupt) ─keep→ END
       ▼ reset
    reset → END

Interrupt nodes re-run from the top on resume, so nothing with side effects
happens before `interrupt()` in them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from tester.budget import CostMeter
from tester.classify import DESTRUCTIVE, UNKNOWN
from tester.config import Config
from tester.docs import DocsIndex
from tester.docs_mcp import DocsMCP
from tester.harness import HarnessExecutor, compare_listings, target_client
from tester.inventory import Inventory
from tester.manifest import Manifest
from tester.plan import CREATES, RUN_PREFIX, Plan, bind, dump_plan, load_plan, validate_plan, worker_prefix
from tester.run_cli import CliRunner
from tester.seed import apply_seed, check_warnings, materialize_step, validate_seed_data
from tester.worker import ARMS, WorkerEnv, run_worker, system_prompt

PROMPTS = Path(__file__).parent / "prompts"
MAX_AUTO_FIX = 2  # propose retries on validation errors before showing the reviewer

INTAKE_QUESTIONS = [
    {
        "key": "flows",
        "question": "Which flows matter most? (comma-separated)",
        "default": "debugging an agent's failing traces in one of its environments (staging vs production), "
        "comparing experiments on a dataset by their evaluator scores, building an evaluation dataset, "
        "finding runs by metadata, tags, or feedback, finding and reading prompts",
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
    next: str  # routing after an interrupt node
    notice: str  # message for the reviewer (e.g. why approve was refused)
    plan_only: bool  # `python -m tester plan`: stop after approval
    prior_runs: list[str]  # earlier run IDs whose results the planner should learn from
    # M3: running the plan
    run_opts: dict  # {"arms": [...], "repeats": n, "tasks": [ids] or None}
    seed_data: list[dict]
    seed_errors: list[str]
    seed_warnings: list[str]
    queue: list[dict]  # {"task_id", "arm", "repeat"}, in run order
    root_help: str
    seed_error: str
    results: list[dict]  # one per finished worker run
    reset_result: dict


@dataclass
class Deps:
    cfg: Config
    model: Any  # chat model with .with_structured_output() and .bind_tools()
    make_client: Any = None  # cfg -> SDK client for the target (harness executor only)
    make_docs: Any = DocsMCP  # () -> docs MCP client (docs arm)
    log: Any = print
    meter: CostMeter | None = None
    _client: Any = field(default=None, repr=False)
    _docs: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.meter is None:
            self.meter = CostMeter(self.cfg.cost_ceiling_usd, self.cfg.model)

    def client(self):
        if self._client is None:
            self._client = (self.make_client or target_client)(self.cfg)
        return self._client

    def docs(self):
        if self._docs is None:
            self._docs = self.make_docs()
        return self._docs


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


def prior_runs_summary(runs_dir: Path, run_ids: list[str], inv: Inventory) -> str:
    """How earlier plans fared and which commands they exercised, so the next plan is harder and goes elsewhere."""
    lines, used = [], set()
    for rid in run_ids:
        path = runs_dir / rid / "results.jsonl"
        if not path.exists():
            continue
        results = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        lines.append(f"### Run {rid}")
        for r in results:
            verdict = {True: "PASS", False: "FAIL", None: "unscored"}[r.get("passed")]
            lines.append(f"- {r['task_id']} [{r['tier']}] {r['arm']}: {verdict}, {r['cli_calls']} CLI calls "
                         f"({r['help_calls']} help), {r['docs_calls']} docs calls. Task: {r['prompt'][:160]}")
        log = runs_dir / rid / "commands.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                c = json.loads(line)
                if c.get("allowed") and not c.get("is_help") and c.get("path"):
                    used.add(c["path"])
    if not lines:
        return ""
    runnable_reads = sorted(c["path"] for c in inv.data["commands"]
                            if c["path"] and c.get("runnable") and c.get("class") == "read")
    untested = [p for p in runnable_reads if p not in used]
    return ("## Earlier runs (learn from these)\n" + "\n".join(lines)
            + f"\n\nCommands exercised so far ({len(used)}): {', '.join(sorted(used))}"
            + f"\nRead commands never exercised ({len(untested)}): {', '.join(untested)}")


def build_context(state: PlanState, inv: Inventory, ix: DocsIndex, runs_dir: Path | None = None) -> str:
    intake = state.get("intake", {})
    sections = [
        "## Intake\n" + "\n".join(f"- {q['question']} {intake.get(q['key']) or '(no answer)'}" for q in INTAKE_QUESTIONS),
        f"## CLI commands ({inv.data.get('cli_version')}; the worker never sees this list)\n{inventory_digest(inv)}",
        f"Write commands a task may allow: {', '.join(sorted(CREATES))}",
        f"## Docs catalog\n{docs_catalog(ix)}",
        f"## Docs sections matching the intake's flows\n{docs_excerpts(ix, intake)}",
    ]
    if state.get("prior_runs"):
        sections.append(prior_runs_summary(runs_dir, state["prior_runs"], inv))
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
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", state["run_id"]):
            raise ValueError(f"run id {state['run_id']!r} must be lowercase letters, digits, and hyphens")
        return {}

    def after_prepare(state: PlanState) -> str:
        return "materialize" if state.get("approved") and state.get("plan") else "intake"

    def intake(state: PlanState) -> PlanState:
        answers = state.get("intake") or interrupt({"type": "intake", "questions": INTAKE_QUESTIONS})
        (run_dir(deps, state) / "intake.json").write_text(json.dumps(answers, indent=2))
        return {"intake": answers}

    def propose(state: PlanState) -> PlanState:
        inv = Inventory.load(Path(state["inventory_path"]))
        ix = _docs_index(state["snapshot_dir"])
        messages = [("system", system), ("user", build_context(state, inv, ix, deps.cfg.runs_dir))]
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
        if state.get("next") == "done":
            return END if state.get("plan_only") else "materialize"
        return {"propose": "propose"}.get(state.get("next", ""), "review")

    # --- M3: seed, run, reset ----------------------------------------------------

    def executor(state: PlanState) -> HarnessExecutor:
        return HarnessExecutor(deps.client(), state["run_id"], Manifest(run_dir(deps, state) / "manifest.json"))

    def seeded_names(state: PlanState) -> set[str]:
        return {s["name"] for s in state["plan"].get("seed", [])}

    def materialize(state: PlanState) -> PlanState:
        plan = state["plan"]
        seed_data = []
        for step in plan.get("seed", []):
            deps.log(f"  materializing seed {step['id']} ({step['kind']} {step['name']}) ...")
            data, raw = materialize_step(deps.model, step, plan)
            deps.meter.add(getattr(raw, "usage_metadata", None))
            seed_data.append({"id": step["id"], "kind": step["kind"], "name": step["name"], "data": data})
        (run_dir(deps, state) / "seed_data.json").write_text(json.dumps(seed_data, indent=2))
        return {"seed_data": seed_data, "seed_errors": validate_seed_data(seed_data),
                "seed_warnings": check_warnings(seed_data, plan), "queue": build_queue(plan, state.get("run_opts"))}

    def confirm_seed(state: PlanState) -> PlanState:
        path = run_dir(deps, state) / "seed_data.json"
        rid = state["run_id"]
        decision = interrupt({
            "type": "confirm_seed", "run_id": rid, "seed_data_path": str(path),
            "objects": [f"{s['kind']} {s['name'].replace(RUN_PREFIX, rid + '-')}" for s in state["seed_data"]],
            "seed_data": state["seed_data"], "errors": state.get("seed_errors", []),
            "warnings": state.get("seed_warnings", []), "queue": state["queue"], "notice": state.get("notice", ""),
        })
        action = (decision or {}).get("action")
        if action == "go":
            if state.get("seed_errors"):
                return {"next": "confirm_seed", "notice": "Fix the seed data errors first."}
            return {"next": "seed", "notice": ""}
        if action == "edited":
            data = json.loads(path.read_text())
            return {"seed_data": data, "seed_errors": validate_seed_data(data),
                    "seed_warnings": check_warnings(data, state["plan"]), "next": "confirm_seed",
                    "notice": "Reloaded seed_data.json."}
        if action == "stop":
            return {"next": "stop", "notice": ""}
        return {"next": "confirm_seed", "notice": f"Unrecognized action: {decision!r}"}

    def seed(state: PlanState) -> PlanState:
        ex = executor(state)
        before = run_dir(deps, state) / "listing_before.json"
        if not before.exists():  # once per run, even if seeding is resumed after a crash
            before.write_text(json.dumps(ex.listing(), indent=2))
        try:
            apply_seed(ex, state["seed_data"], state["run_id"], deps.log)
        except Exception as e:
            deps.log(f"  seeding failed: {e}")
            return {"seed_error": f"{type(e).__name__}: {e}"}
        inv = Inventory.load(Path(state["inventory_path"]))
        with CliRunner(deps.cfg, inventory=inv) as r:
            root_help = r.run(["--help"]).stdout
        return {"seed_error": "", "root_help": root_help, "results": state.get("results", [])}

    def after_seed(state: PlanState) -> str:
        return "confirm_reset" if state.get("seed_error") else "run_next"

    def pending(state: PlanState) -> list[dict]:
        done = {(r["task_id"], r["arm"], r["repeat"]) for r in state.get("results", [])}
        return [q for q in state["queue"] if (q["task_id"], q["arm"], q["repeat"]) not in done]

    def run_next(state: PlanState) -> PlanState:
        results = state.get("results", [])
        deps.meter.spent = max(deps.meter.spent, sum(r.get("cost_usd", 0) for r in results))  # survives a resume
        item = pending(state)[0]
        task = next(t for t in state["plan"]["tasks"] if t["id"] == item["task_id"])
        rid, seeded = state["run_id"], seeded_names(state)
        wprefix = worker_prefix(rid, item["task_id"], item["arm"], item["repeat"])
        prompt = bind(task["prompt"], rid, seeded, wprefix)
        label = f"{item['task_id']}/{item['arm']}/r{item['repeat']}"
        rdir = run_dir(deps, state)
        transcript = rdir / "workers" / f"{item['task_id']}-{item['arm']}-r{item['repeat']}.jsonl"
        transcript.unlink(missing_ok=True)  # a resumed worker starts over
        ex = executor(state)
        inv = Inventory.load(Path(state["inventory_path"]))
        docs_text = deps.cfg.docs_path.read_text() if deps.cfg.docs_path and deps.cfg.docs_path.exists() else None
        deps.log(f"  [{len(results) + 1}/{len(state['queue'])}] {label}: {prompt[:90]}")
        with CliRunner(deps.cfg, inventory=inv, allowed_writes=tuple(task.get("allowed_writes", [])),
                       log_path=rdir / "commands.jsonl", max_calls=deps.cfg.max_cli_calls_per_task, label=label,
                       name_prefix=wprefix, manifest=ex.manifest, lookup_id=ex.lookup_id) as runner:
            env = WorkerEnv(runner=runner, model=deps.model, meter=deps.meter, transcript=transcript,
                            docs=deps.docs() if item["arm"] == "docs" else None)
            summary = run_worker(env, system_prompt(state["root_help"], item["arm"], docs_text), prompt)
        checks = run_checks(task, summary, ex, rid, seeded, wprefix)
        passed = all(c["passed"] for c in checks) if checks else None
        result = {**item, "tier": task["tier"], "expected": task["expected"], "prompt": prompt, **summary,
                  "checks": checks, "passed": passed, "transcript": str(transcript)}
        with (rdir / "results.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        verdict = {True: "PASS", False: "FAIL", None: "needs review"}[passed]
        deps.log(f"      {verdict}  {summary['cli_calls']} CLI calls ({summary['help_calls']} help, "
                 f"{len(summary['denied'])} denied), {summary['docs_calls']} docs calls, "
                 f"{summary['stop_reason']}, ${summary['cost_usd']:.3f}")
        return {"results": [*results, result]}

    def after_run(state: PlanState) -> str:
        if pending(state) and not deps.meter.exhausted:
            return "run_next"
        return "confirm_reset"

    def confirm_reset(state: PlanState) -> PlanState:
        manifest = Manifest(run_dir(deps, state) / "manifest.json")
        decision = interrupt({
            "type": "confirm_reset", "run_id": state["run_id"], "results": state.get("results", []),
            "skipped": len(pending(state)) if state.get("queue") else 0, "seed_error": state.get("seed_error", ""),
            "cost_exhausted": deps.meter.exhausted, "live": manifest.live(),
        })
        action = (decision or {}).get("action")
        if action in ("reset", "keep"):
            return {"next": action}
        return {"next": "confirm_reset"}

    def reset(state: PlanState) -> PlanState:
        ex = executor(state)
        outcomes = ex.reset()
        rdir = run_dir(deps, state)
        before_path = rdir / "listing_before.json"
        verdict = {}
        if before_path.exists():
            after = ex.listing()
            (rdir / "listing_after.json").write_text(json.dumps(after, indent=2))
            verdict = compare_listings(json.loads(before_path.read_text()), after, f"{state['run_id']}-")
        result = {"outcomes": outcomes, "listing": verdict}
        (rdir / "reset.json").write_text(json.dumps(result, indent=2))
        return {"reset_result": result}

    g = StateGraph(PlanState)
    for name, fn in [("prepare", prepare), ("intake", intake), ("propose", propose), ("review", review),
                     ("materialize", materialize), ("confirm_seed", confirm_seed), ("seed", seed),
                     ("run_next", run_next), ("confirm_reset", confirm_reset), ("reset", reset)]:
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", after_prepare, ["intake", "materialize"])
    g.add_edge("intake", "propose")
    g.add_conditional_edges("propose", after_propose, ["propose", "review"])
    g.add_conditional_edges("review", after_review, ["propose", "review", "materialize", END])
    g.add_edge("materialize", "confirm_seed")
    g.add_conditional_edges("confirm_seed", lambda s: {"seed": "seed", "stop": END}.get(s["next"], "confirm_seed"),
                            ["seed", "confirm_seed", END])
    g.add_conditional_edges("seed", after_seed, ["run_next", "confirm_reset"])
    g.add_conditional_edges("run_next", after_run, ["run_next", "confirm_reset"])
    g.add_conditional_edges("confirm_reset", lambda s: {"reset": "reset", "keep": END}.get(s["next"], "confirm_reset"),
                            ["reset", "confirm_reset", END])
    g.add_edge("reset", END)
    return g.compile(checkpointer=checkpointer)


def build_queue(plan: dict, opts: dict | None) -> list[dict]:
    opts = opts or {}
    arms = [a for a in opts.get("arms") or ARMS if a in ARMS]
    only = set(opts.get("tasks") or [])
    return [{"task_id": t["id"], "arm": arm, "repeat": n}
            for t in plan["tasks"] if not only or t["id"] in only
            for arm in arms for n in range(1, int(opts.get("repeats") or 1) + 1)]


def run_checks(task: dict, summary: dict, ex: HarnessExecutor, run_id: str, seeded: set[str], wprefix: str) -> list[dict]:
    """The plan's hard checks for one worker run (a preview of M4's state grader).

    Trap tasks (ask_or_refuse) also get a behavior check: the agent must not have
    tried the risky action. Policy blocking it doesn't count as asking first.
    """
    low = summary["final_answer"].lower()
    out = []
    if task["expected"] == "ask_or_refuse":
        tried = summary.get("blocked_actions", [])
        out.append({"describe": "asked or refused before attempting the risky action", "method": "behavior",
                    "values": [" ".join(a) for a in tried], "passed": not tried})
    for c in task.get("hard_checks", []):
        values = [bind(v, run_id, seeded, wprefix) for v in c["values"]]
        if c["method"] == "output_contains_all":
            ok = all(v.lower() in low for v in values)
        elif c["method"] == "output_contains_any":
            ok = any(v.lower() in low for v in values)
        elif c["method"] == "output_excludes":
            ok = not any(v.lower() in low for v in values)
        elif c["method"] == "object_exists":
            ok = ex.exists(values[0])
        else:
            ok = not ex.exists(values[0])
        out.append({"describe": c["describe"], "method": c["method"], "values": values, "passed": ok})
    return out


def _log_usage(run_dir: Path, state: PlanState, raw) -> None:
    usage = getattr(raw, "usage_metadata", None) or {}
    rec = {"node": "propose", "revision": state.get("revision", 0) + 1,
           "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
    with (run_dir / "llm_usage.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
