"""CLI entry: python -m tester discover | docs | plan | run | reset"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from collections import Counter

from tester.config import ConfigError, load_config


def new_run_id() -> str:
    # Short and lowercase: it becomes a name prefix for every object we create.
    return f"ct{time.strftime('%m%d%H%M')}{secrets.token_hex(2)}"


def cmd_discover(args: argparse.Namespace) -> int:
    from tester.inventory import discover

    cfg = load_config()
    for w in cfg.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    run_id = args.run_id or new_run_id()
    print(f"run {run_id}: crawling `{cfg.cli_bin.name} --help` ...", flush=True)
    data = discover(cfg, run_id, use_llm=not args.no_llm, do_probe=not args.no_probe)
    print_summary(data, cfg.runs_dir / run_id)
    return 0


def print_summary(data: dict, run_dir) -> None:
    cmds = [c for c in data["commands"] if c["path"]]
    runnable = [c for c in cmds if c["runnable"]]
    classes = Counter(c["class"] for c in runnable)
    probes = Counter(c["probe"] for c in runnable if c["probe"] != "not_probed")

    print(f"\ninventory: {run_dir / 'inventory.json'}")
    print(f"cli: {data['cli_version']} · {len(cmds)} commands ({len(runnable)} runnable, {len(cmds) - len(runnable)} groups)")
    print("runnable by class: " + " · ".join(f"{k} {classes[k]}" for k in ("read", "write", "destructive", "unknown")))
    if probes:
        print("probes: " + " · ".join(f"{k} {v}" for k, v in probes.most_common()))

    denied = [c for c in runnable if c["probe"] == "permission_denied"]
    if denied:
        print(f"\nBLIND SPOTS ({len(denied)}): the key can't run these")
        for c in denied:
            print(f"  {c['path']}: {c['probe_detail']}")

    errors = [c for c in runnable if c["probe"] == "error"]
    if errors:
        print(f"\nprobe errors ({len(errors)}):")
        for c in errors:
            print(f"  {c['path']}: {c['probe_detail']}")

    unknown = [c for c in runnable if c["class"] == "unknown"]
    if unknown:
        suggested = sum(1 for c in unknown if c.get("suggested_class"))
        print(f"\nunknown → denied until classified ({len(unknown)}):")
        for c in unknown:
            hint = f"  [suggested: {c['suggested_class']}]" if c.get("suggested_class") else ""
            print(f"  {c['path']}{hint}")
        if suggested:
            print(f"suggestions: {run_dir / 'suggested_classifications.yaml'} (review, then copy into classifications.yaml)")

    crawl_errors = [c for c in data["commands"] if c.get("crawl_error")]
    if crawl_errors:
        print(f"\nhelp crawl errors ({len(crawl_errors)}): " + ", ".join(c["path"] or "<root>" for c in crawl_errors))
    for note in data.get("notes", []):
        print(f"note: {note}")


def latest_inventory(runs_dir) -> "Path | None":
    found = sorted(runs_dir.glob("*/inventory.json"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def ensure_snapshot(cfg, refresh: bool = False, snapshot_dir=None):
    """Today's docs snapshot, fetched once per day unless --refresh."""
    from pathlib import Path

    from tester.docs import snapshot

    snap = Path(snapshot_dir) if snapshot_dir else cfg.runs_dir / "_docs" / time.strftime("%Y-%m-%d")
    if refresh or not (snap / "index.json").exists():
        print(f"snapshotting docs from {cfg.docs_index_url} ...", flush=True)
        data = snapshot(snap, cfg.docs_index_url)
        print(f"  {len(data['pages'])} pages from {data['indexes']} indexes"
              + (f", {len(data['errors'])} fetch errors" if data["errors"] else ""))
    return snap


def cmd_docs(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from tester.docs import check_docs
    from tester.inventory import Inventory

    cfg = load_config()
    inv_path = Path(args.inventory) if args.inventory else latest_inventory(cfg.runs_dir)
    if not inv_path or not inv_path.exists():
        print("no inventory found; run `python -m tester discover` first", file=sys.stderr)
        return 1
    snap = ensure_snapshot(cfg, args.refresh, args.snapshot)
    result = check_docs(snap, Inventory.load(inv_path))
    result["inventory"] = str(inv_path)
    run_id = args.run_id or new_run_id()
    out = cfg.runs_dir / run_id / "docs_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print_docs_summary(result, out, args.limit)
    return 0


def print_docs_summary(r: dict, out, limit: int) -> None:
    print(f"\ndocs check: {out}")
    print(f"snapshot {r['fetched_at']} · {r['pages']} pages, {r['pages_with_commands']} with CLI examples · "
          f"{r['commands']} distinct commands, {r['commands_with_issues']} with issues")
    print(f"inventory {r['inventory']} ({r['cli_version']})")
    print(f"runnable commands that appear in the docs: {r['runnable_documented']} of {r['runnable_total']}")
    if r["issue_counts"]:
        print("issues: " + " · ".join(f"{k} {v}" for k, v in r["issue_counts"].items()))
    by_kind: dict[str, list] = {}
    for res in r["results"]:
        for issue in res["issues"]:
            by_kind.setdefault(issue["kind"], []).append((res, issue))
    for kind, items in by_kind.items():
        print(f"\n{kind} ({len(items)}):")
        for res, issue in items[:limit]:
            print(f"  {issue['detail']}")
            print(f"    {res['text'][:120]}")
            print(f"    {res['url']}:{res['line']}")
        if len(items) > limit:
            print(f"  ... {len(items) - limit} more in the JSON")


# --- plan (M2) -----------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    return drive(args, plan_only=True)


def cmd_run(args: argparse.Namespace) -> int:
    return drive(args, plan_only=False)


def run_opts(args: argparse.Namespace, cfg) -> dict:
    return {"arms": [a.strip() for a in args.arms.split(",") if a.strip()],
            "repeats": args.repeats or cfg.repeats,
            "tasks": [t.strip() for t in args.tasks.split(",")] if args.tasks else None}


def prior_runs(args: argparse.Namespace, cfg, exclude: str) -> list[str]:
    """Runs the planner learns from: --learn-from, else the 3 most recent runs with results."""
    if getattr(args, "learn_from", None) is not None:
        return [r.strip() for r in args.learn_from.split(",") if r.strip()]
    found = sorted((p.stat().st_mtime, p.parent.name) for p in cfg.runs_dir.glob("*/results.jsonl")
                   if p.parent.name != exclude)
    return [name for _, name in found[-3:]]


def plan_header(text: str) -> dict:
    meta = {}
    for line in text.splitlines():
        if not line.startswith("# "):
            break
        key, _, value = line[2:].partition(": ")
        meta[key.strip()] = value.strip()
    return meta


def start_from_plan_file(args, cfg, start: dict) -> bool:
    """Fill `start` from an approved plan.yaml. False (after printing why) if it doesn't validate."""
    from pathlib import Path

    from tester.docs import DocsIndex
    from tester.inventory import Inventory
    from tester.plan import load_plan, validate_plan

    text = Path(args.plan).read_text()
    meta = plan_header(text)
    inv_path = Path(args.inventory or meta.get("inventory") or latest_inventory(cfg.runs_dir) or "")
    snap = Path(args.snapshot or meta.get("docs_snapshot") or ensure_snapshot(cfg))
    plan = load_plan(text)
    errors = validate_plan(plan, Inventory.load(inv_path), {p["url"] for p in DocsIndex(snap).catalog()})
    if errors:
        print(f"{args.plan} doesn't validate:", *[f"  - {e}" for e in errors], sep="\n", file=sys.stderr)
        return False
    start.update(plan=plan, approved=True, inventory_path=str(inv_path), snapshot_dir=str(snap))
    return True


def drive(args: argparse.Namespace, plan_only: bool) -> int:
    """Run the graph from the terminal, answering each interrupt with a prompt."""
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    from tester.graph import INTAKE_QUESTIONS, Deps, build_graph

    cfg = load_config()
    for w in cfg.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    cmd = "plan" if plan_only else "run"
    run_id = args.resume or args.run_id or new_run_id()
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 1000}
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(cfg.runs_dir / "checkpoints.sqlite")) as saver:
        graph = build_graph(Deps(cfg=cfg, model=make_model(cfg)), checkpointer=saver)
        if args.resume:
            state = graph.get_state(config)
            if not state.values:
                print(f"no saved run {run_id!r}", file=sys.stderr)
                return 1
            print(f"resuming run {run_id}")
            if not plan_only and state.values.get("plan_only") and state.values.get("approved") and not state.next:
                # A plan approved with `plan`: carry on into seeding.
                graph.update_state(config, {"plan_only": False, "run_opts": run_opts(args, cfg)}, as_node="review")
                graph.invoke(None, config)
        else:
            start = {"run_id": run_id, "plan_only": plan_only}
            if not plan_only:
                start["run_opts"] = run_opts(args, cfg)
            if getattr(args, "plan", None):
                if not start_from_plan_file(args, cfg, start):
                    return 1
                print(f"run {run_id}: plan {args.plan}")
            else:
                start["prior_runs"] = prior_runs(args, cfg, exclude=run_id)
                if start["prior_runs"]:
                    print(f"learning from earlier runs: {', '.join(start['prior_runs'])}")
                inv_path = Path(args.inventory) if args.inventory else latest_inventory(cfg.runs_dir)
                if not inv_path or not inv_path.exists():
                    print("no inventory found; run `python -m tester discover` first", file=sys.stderr)
                    return 1
                start.update(inventory_path=str(inv_path),
                             snapshot_dir=str(ensure_snapshot(cfg, snapshot_dir=args.snapshot)))
                if args.defaults:
                    start["intake"] = {q["key"]: q["default"] for q in INTAKE_QUESTIONS}
                print(f"run {run_id}: inventory {inv_path}")
                print("drafting a plan ...", flush=True)
            graph.invoke(start, config)

        handlers = {"intake": ask_intake, "review": ask_review, "confirm_seed": ask_confirm_seed,
                    "confirm_reset": ask_confirm_reset}
        progress = {"revise": "drafting a plan ...", "go": "seeding and running ...", "reset": "resetting ..."}
        while True:
            state = graph.get_state(config)
            if not state.interrupts:
                break
            payload = state.interrupts[0].value
            answer = handlers[payload["type"]](payload)
            if answer is None:
                print(f"\npaused. resume with: python -m tester {cmd} --resume {run_id}")
                return 0
            note = "drafting a plan ..." if payload["type"] == "intake" else progress.get(answer.get("action"))
            if note:
                print(note, flush=True)
            graph.invoke(Command(resume=answer), config)

        final = graph.get_state(config).values
        if plan_only and final.get("approved"):
            print(f"\napproved: {cfg.runs_dir / run_id / 'plan.yaml'} (revision {final.get('revision')})")
            print(f"run it with: python -m tester run --resume {run_id}")
        if final.get("reset_result"):
            print_reset(final["reset_result"])
        elif final.get("results"):
            print(f"\nkept the seeded objects. clean up with: python -m tester reset {run_id}")
        print(f"\nrun folder: {cfg.runs_dir / run_id}")
        return 0


def summarize_seed(step: dict) -> str:
    d = step["data"]
    if step["kind"] == "dataset":
        return f"{len(d['examples'])} examples, e.g. {d['examples'][0]['inputs_json'][:80]}" if d["examples"] else "empty"
    if step["kind"] == "prompt":
        return " / ".join(f"{m['role']}: {m['content'][:60]}" for m in d["messages"])
    if step["kind"] == "experiment":
        scores: dict[str, list[float]] = {}
        for row in d["rows"]:
            for s in row.get("scores", []):
                scores.setdefault(s["key"], []).append(s["score"])
        means = ", ".join(f"{k} mean {sum(v) / len(v):.3g}" for k, v in scores.items())
        return f"{len(d['rows'])} rows on {d['dataset']}; {means or 'no scores'}"
    if step["kind"] == "agent":
        return "; ".join(f"{e['environment']}: {summarize_runs(e['runs'])}" for e in d["environments"] if e["runs"])
    return summarize_runs(d["runs"])


def summarize_runs(runs: list[dict]) -> str:
    total = sum(r.get("repeat", 1) for r in runs)
    errors = sum(r.get("repeat", 1) for r in runs if r.get("error"))
    oldest = max((r["start_offset_s"] + (r.get("repeat", 1) - 1) * r.get("repeat_spacing_s", 60) for r in runs), default=0)
    feedback = sum(r.get("repeat", 1) for r in runs if r.get("feedback"))
    return (f"{total} root runs ({errors} errored, {feedback} with feedback), oldest {oldest / 86400:.1f} days ago; "
            f"distinct errors: {sorted({r['error'][:40] for r in runs if r.get('error')}) or 'none'}")


def ask_confirm_seed(payload: dict) -> dict | None:
    print(f"\n=== about to seed run {payload['run_id']} (seed data: {payload['seed_data_path']})")
    for obj, step in zip(payload["objects"], payload["seed_data"]):
        print(f"  create {obj}: {summarize_seed(step)}")
    arms = sorted({q["arm"] for q in payload["queue"]})
    print(f"then {len(payload['queue'])} worker runs (arms: {', '.join(arms)}), one at a time")
    for w in payload["warnings"]:
        print(f"  warning: {w}")
    for e in payload["errors"]:
        print(f"  ERROR: {e}")
    if payload.get("notice"):
        print(f"note: {payload['notice']}")
    while True:
        choice = input("\n[g]o  [e]dit seed_data.json  [s]top (creates nothing)  [q]uit (resume later)\n> ").strip().lower()
        if choice in ("g", "go"):
            return {"action": "go"}
        if choice in ("e", "edit"):
            open_editor(payload["seed_data_path"])
            return {"action": "edited"}
        if choice in ("s", "stop"):
            return {"action": "stop"}
        if choice in ("q", "quit"):
            return None


def print_results(results: list[dict]) -> None:
    print(f"\n{'run':<16} {'verdict':<13} {'cli':>4} {'help':>4} {'deny':>4} {'docs':>4} {'cost':>7}  stop")
    for r in results:
        verdict = {True: "PASS", False: "FAIL", None: "needs review"}[r["passed"]]
        label = f"{r['task_id']}/{r['arm']}/r{r['repeat']}"
        print(f"{label:<16} {verdict:<13} {r['cli_calls']:>4} {r['help_calls']:>4} {len(r['denied']):>4} "
              f"{r['docs_calls']:>4} {r['cost_usd']:>7.3f}  {r['stop_reason']}")
        for c in r["checks"]:
            if not c["passed"]:
                print(f"{'':<16} failed check: {c['describe']} ({c['method']} {c['values']})")
    print(f"total worker cost ${sum(r['cost_usd'] for r in results):.2f}")


def ask_confirm_reset(payload: dict) -> dict | None:
    if payload.get("seed_error"):
        print(f"\nSEEDING FAILED: {payload['seed_error']}")
    if payload["results"]:
        print_results(payload["results"])
    if payload.get("skipped"):
        print(f"{payload['skipped']} runs skipped" + (" (cost ceiling reached)" if payload.get("cost_exhausted") else ""))
    print(f"\n{len(payload['live'])} objects in the manifest to delete:")
    for e in payload["live"]:
        print(f"  {e['kind']} {e['name']}  ({e['status']}, by {e['created_by']})")
    retry = "  [t]ry seeding again" if payload.get("seed_error") else ""
    while True:
        choice = input(f"\n[r]eset (delete them)  [k]eep for inspection{retry}  [q]uit (resume later)\n> ").strip().lower()
        if choice in ("t", "try") and retry:
            return {"action": "retry_seed"}
        if choice in ("r", "reset"):
            return {"action": "reset"}
        if choice in ("k", "keep"):
            return {"action": "keep"}
        if choice in ("q", "quit"):
            return None


def print_reset(result: dict) -> None:
    print("\nreset:")
    for o in result["outcomes"]:
        print(f"  {o['kind']} {o['name']}: {o['outcome']}")
    v = result.get("listing") or {}
    if v:
        print("workspace listing vs. before the run: " + ("CLEAN" if v["clean"] else "NOT CLEAN"))
        for key in ("ours_left", "removed", "others_added"):
            if v.get(key):
                print(f"  {key.replace('_', ' ')}: {', '.join(v[key])}")


def cmd_reset(args: argparse.Namespace) -> int:
    import json

    from tester.harness import HarnessExecutor, compare_listings, target_client
    from tester.manifest import Manifest

    cfg = load_config()
    rdir = cfg.runs_dir / args.run_id
    path = rdir / "manifest.json"
    if not path.exists():
        print(f"no manifest for run {args.run_id!r} ({path})", file=sys.stderr)
        return 1
    manifest = Manifest(path)
    live = manifest.live()
    print(f"{len(live)} objects in the manifest to delete:")
    for e in live:
        print(f"  {e['kind']} {e['name']}  ({e['status']}, by {e['created_by']})")
    if live and not args.yes and input("delete them? [y/N] ").strip().lower() != "y":
        return 0
    ex = HarnessExecutor(target_client(cfg), args.run_id, manifest)
    result = {"outcomes": ex.reset(), "listing": {}}
    before = rdir / "listing_before.json"
    if before.exists():
        after = ex.listing()
        (rdir / "listing_after.json").write_text(json.dumps(after, indent=2))
        result["listing"] = compare_listings(json.loads(before.read_text()), after, f"{args.run_id}-")
    (rdir / "reset.json").write_text(json.dumps(result, indent=2))
    print_reset(result)
    return 0


def make_model(cfg):
    from langchain.chat_models import init_chat_model

    if not cfg.model:
        raise ConfigError("MODEL is not set")
    return init_chat_model(cfg.model)


def ask_intake(payload: dict) -> dict | None:
    print("\nA few questions to shape the plan (Enter keeps the default, q to pause):")
    answers = {}
    for q in payload["questions"]:
        default = f" [{q['default']}]" if q["default"] else ""
        reply = input(f"\n{q['question']}{default}\n> ").strip()
        if reply == "q":
            return None
        answers[q["key"]] = reply or q["default"]
    return answers


def print_plan(payload: dict) -> None:
    plan = payload["plan"]
    print(f"\n=== plan, revision {payload['revision']}: {payload['plan_path']}")
    print(plan.get("summary", ""))
    if plan.get("seed"):
        print("\nseed:")
        for s in plan["seed"]:
            print(f"  {s['id']} {s['kind']:<18} {s['name']}")
            print(f"      {s['spec'][:200]}")
    print("\ntasks:")
    for t in plan.get("tasks", []):
        writes = f"  writes: {', '.join(t['allowed_writes'])}" if t.get("allowed_writes") else ""
        print(f"  {t['id']} [{t['tier']}, expect {t['expected']}]{writes}")
        print(f"      {t['prompt']}")
        print(f"      why: {t.get('rationale', '')}")
        for c in t.get("hard_checks", []):
            print(f"      check: {c['method']} {c.get('values')}  ({c['describe']})")
        for ref in t.get("docs_refs", []):
            print(f"      docs: {ref}")
    if payload["errors"]:
        print(f"\nERRORS ({len(payload['errors'])}), fix before approving:")
        for e in payload["errors"]:
            print(f"  - {e}")
    if payload.get("notice"):
        print(f"\nnote: {payload['notice']}")


def open_editor(path: str) -> None:
    import os
    import subprocess

    subprocess.call([os.environ.get("EDITOR", "vi"), path])


def ask_review(payload: dict) -> dict | None:
    print_plan(payload)
    while True:
        choice = input("\n[a]pprove  [r]evise with feedback  [e]dit plan.yaml  [q]uit (resume later)\n> ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("r", "revise"):
            feedback = input("what should change?\n> ").strip()
            if feedback:
                return {"action": "revise", "feedback": feedback}
        elif choice in ("e", "edit"):
            open_editor(payload["plan_path"])
            return {"action": "edited"}
        elif choice in ("q", "quit"):
            return None


KNOWN_VARS = [
    "TARGET_CLI_BIN", "TARGET_API_KEY", "TARGET_ENDPOINT", "TARGET_WORKSPACE_ID", "TARGET_DOCS_PATH", "DOCS_INDEX_URL",
    "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_TRACING", "MODEL", "ANTHROPIC_API_KEY",
]


def cmd_config(_args: argparse.Namespace) -> int:
    """Report which variables are set and where from. Never prints values."""
    from dotenv import dotenv_values

    from tester.config import DEFAULT_ENV_FILE

    file_vals = dotenv_values(DEFAULT_ENV_FILE) if DEFAULT_ENV_FILE.exists() else {}
    print(f"{DEFAULT_ENV_FILE}: {'found' if DEFAULT_ENV_FILE.exists() else 'MISSING'}")
    import os

    for var in KNOWN_VARS:
        if file_vals.get(var):
            where = "set in file"
        elif var in file_vals:
            where = "EMPTY in file"
        elif os.environ.get(var):
            where = "set in shell env only"
        else:
            where = "not set"
        print(f"  {var:<22} {where}")
    extra = sorted(k for k in file_vals if k not in KNOWN_VARS)
    if extra:
        print(f"  other names in file: {', '.join(extra)}")
    try:
        cfg = load_config()
        print(f"config OK · cli {cfg.cli_bin} · target key {'present' if cfg.has_target_key else 'MISSING'}")
        for w in cfg.warnings:
            print(f"WARNING: {w}")
    except ConfigError as e:
        print(f"config error: {e}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tester")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="crawl --help, classify commands, probe permissions")
    d.add_argument("--run-id", help="reuse a run ID instead of generating one")
    d.add_argument("--no-llm", action="store_true", help="skip LLM suggestions for unknown commands")
    d.add_argument("--no-probe", action="store_true", help="skip permission probes")
    d.set_defaults(func=cmd_discover)

    dc = sub.add_parser("docs", help="snapshot the docs and check their CLI examples against the inventory")
    dc.add_argument("--run-id", help="reuse a run ID instead of generating one")
    dc.add_argument("--inventory", help="inventory.json to check against (default: newest run)")
    dc.add_argument("--snapshot", help="use this snapshot dir instead of today's")
    dc.add_argument("--refresh", action="store_true", help="re-fetch the docs even if today's snapshot exists")
    dc.add_argument("--limit", type=int, default=15, help="examples to print per issue kind")
    dc.set_defaults(func=cmd_docs)

    sub.add_parser("config", help="show which settings are set (never prints values)").set_defaults(func=cmd_config)
    pl = sub.add_parser("plan", help="intake → propose → review a test plan (resumable)")
    pl.add_argument("--run-id", help="use this run ID for a new run")
    pl.add_argument("--resume", metavar="RUN_ID", help="continue a paused run")
    pl.add_argument("--inventory", help="inventory.json to plan against (default: newest run)")
    pl.add_argument("--snapshot", help="docs snapshot dir (default: today's)")
    pl.add_argument("--defaults", action="store_true", help="skip intake questions and use the defaults")
    pl.add_argument("--learn-from", metavar="RUN_IDS", help="earlier runs to learn from (default: 3 most recent; '' for none)")
    pl.set_defaults(func=cmd_plan)
    rn = sub.add_parser("run", help="plan (or take an approved plan), seed, run each task per arm, reset")
    rn.add_argument("--plan", help="start from this approved plan.yaml instead of planning")
    rn.add_argument("--run-id", help="use this run ID for a new run")
    rn.add_argument("--resume", metavar="RUN_ID", help="continue a paused run (or run a plan approved with `plan`)")
    rn.add_argument("--arms", default="help,docs", help="knowledge arms to run (default: help,docs)")
    rn.add_argument("--repeats", type=int, help="runs per task per arm (default: REPEATS)")
    rn.add_argument("--tasks", help="only these task IDs, comma-separated")
    rn.add_argument("--inventory", help="inventory.json (default: the plan's, else newest run)")
    rn.add_argument("--snapshot", help="docs snapshot dir (default: the plan's, else today's)")
    rn.add_argument("--defaults", action="store_true", help="skip intake questions and use the defaults")
    rn.add_argument("--learn-from", metavar="RUN_IDS", help="earlier runs to learn from (default: 3 most recent; '' for none)")
    rn.set_defaults(func=cmd_run)
    rs = sub.add_parser("reset", help="delete a run's manifest entries and verify by listing")
    rs.add_argument("run_id")
    rs.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    rs.set_defaults(func=cmd_reset)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
