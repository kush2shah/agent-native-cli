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
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    from tester.graph import Deps, build_graph

    cfg = load_config()
    for w in cfg.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    run_id = args.resume or args.run_id or new_run_id()
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 40}
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(cfg.runs_dir / "checkpoints.sqlite")) as saver:
        graph = build_graph(Deps(cfg=cfg, model=make_model(cfg)), checkpointer=saver)
        if args.resume:
            if not graph.get_state(config).values:
                print(f"no saved run {run_id!r}", file=sys.stderr)
                return 1
            print(f"resuming run {run_id}")
        else:
            inv_path = Path(args.inventory) if args.inventory else latest_inventory(cfg.runs_dir)
            if not inv_path or not inv_path.exists():
                print("no inventory found; run `python -m tester discover` first", file=sys.stderr)
                return 1
            start = {"run_id": run_id, "inventory_path": str(inv_path),
                     "snapshot_dir": str(ensure_snapshot(cfg, snapshot_dir=args.snapshot))}
            if args.defaults:
                from tester.graph import INTAKE_QUESTIONS

                start["intake"] = {q["key"]: q["default"] for q in INTAKE_QUESTIONS}
            print(f"run {run_id}: inventory {inv_path}")
            print("drafting a plan ...", flush=True)
            graph.invoke(start, config)

        while True:
            state = graph.get_state(config)
            if not state.interrupts:
                break
            payload = state.interrupts[0].value
            answer = ask_intake(payload) if payload["type"] == "intake" else ask_review(payload)
            if answer is None:
                print(f"\npaused. resume with: python -m tester plan --resume {run_id}")
                return 0
            if answer.get("action") == "revise" or payload["type"] == "intake":
                print("drafting a plan ...", flush=True)
            graph.invoke(Command(resume=answer), config)

        final = graph.get_state(config).values
        if final.get("approved"):
            print(f"\napproved: {cfg.runs_dir / run_id / 'plan.yaml'} (revision {final.get('revision')})")
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


def ask_review(payload: dict) -> dict | None:
    import os
    import subprocess

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
            subprocess.call([os.environ.get("EDITOR", "vi"), payload["plan_path"]])
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


def not_yet(milestone: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"not implemented yet ({milestone})", file=sys.stderr)
        return 2

    return run


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
    pl.set_defaults(func=cmd_plan)
    sub.add_parser("run", help="(M3)").set_defaults(func=not_yet("M3"))
    sub.add_parser("reset", help="(M3)").set_defaults(func=not_yet("M3"))

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
