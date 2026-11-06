"""CLI entry: python -m tester discover | plan | run | reset"""

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


KNOWN_VARS = [
    "TARGET_CLI_BIN", "TARGET_API_KEY", "TARGET_ENDPOINT", "TARGET_WORKSPACE_ID", "TARGET_DOCS_PATH",
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

    sub.add_parser("config", help="show which settings are set (never prints values)").set_defaults(func=cmd_config)
    sub.add_parser("plan", help="(M2)").set_defaults(func=not_yet("M2"))
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
