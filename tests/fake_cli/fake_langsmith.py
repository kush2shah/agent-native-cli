"""A stub of the LangSmith CLI for offline tests.

Mimics what the harness depends on: cobra-style help, one urfave-style generated
subtree, aliases, `unknown flag` errors, the exit-0 help output for an unknown
subcommand of a group, a 403 for one command, and auth via LANGSMITH_API_KEY.
`debug *` commands expose the subprocess env, cwd, and a hang for safety tests.
"""

from __future__ import annotations

import json
import os
import sys
import time

GLOBAL_FLAGS = [
    (None, "api-url", "string", "LangSmith API URL [env: LANGSMITH_ENDPOINT]"),
    (None, "format", "string", 'Output format: pretty or json (default "pretty")'),
    (None, "profile", "string", "Named profile to use [env: LANGSMITH_PROFILE]"),
    (None, "workspace", "string", "LangSmith workspace ID [env: LANGSMITH_WORKSPACE_ID]"),
]
URFAVE_GLOBALS = [
    (None, "debug", None, "Enable debug logging"),
    (None, "base-url", "string", "Override the base URL for API requests (default: url)"),
    (None, "format", "string", "The format for displaying response data"),
]
LIMIT = ("n", "limit", "int", "Maximum number of items to return (default 20)")
OUTPUT = ("o", "output", "string", "Write JSON output to a file")

TREE = {
    "summary": "Fake LangSmith CLI — query and manage LangSmith resources.",
    "children": {
        "datasets": {
            "summary": "Create, manage, and inspect evaluation datasets",
            "aliases": ["dataset"],
            "children": {
                "list": {"summary": "List all datasets in the workspace", "run": "datasets_list", "flags": [LIMIT, OUTPUT]},
                "get": {"summary": "Get dataset details by name or UUID", "args": ["NAME_OR_ID"], "run": "echo"},
                "create": {"summary": "Create a new empty dataset", "run": "datasets_create",
                           "flags": [(None, "description", "string", "Optional description"),
                                     (None, "name", "string", "Name for the new dataset (required)")]},
                "delete": {"summary": "Delete a dataset by name or UUID", "args": ["NAME_OR_ID"], "run": "echo"},
            },
        },
        "projects": {
            "summary": "List and inspect tracing projects",
            "aliases": ["project", "sessions"],
            "children": {
                "list": {"summary": "List tracing projects", "run": "forbidden", "flags": [LIMIT]},
            },
        },
        "experiments": {
            "summary": "Query evaluation experiments",
            "children": {
                "list": {"summary": "List experiments", "run": "needs_dataset", "flags": [LIMIT]},
                "compute": {"summary": "Compute experiment aggregates", "run": "echo"},
            },
        },
        "api": {
            "summary": "Browse API endpoints and make authenticated requests",
            "runnable": True,
            "args": ["[PATH]"],
            "flags": [("X", "method", "string", "HTTP method")],
            "children": {"ls": {"summary": "List available API endpoints", "run": "echo"}},
        },
        "runs": {
            "summary": "Query and export individual runs",
            "aliases": ["run"],
            "children": {
                "list": {"summary": "List runs", "run": "echo", "flags": [LIMIT]},
                "share": {
                    "style": "urfave",
                    "summary": "Manage share",
                    "children": {
                        "create": {"style": "urfave", "summary": "Creates or returns a share token for a run.",
                                   "run": "echo", "flags": [(None, "run-id", "string", "")]},
                    },
                },
            },
        },
        "debug": {
            "summary": "Test helpers",
            "children": {
                "env": {"summary": "Print the environment as JSON", "run": "env"},
                "pwd": {"summary": "Print the working directory", "run": "pwd"},
                "sleep": {"summary": "Sleep for a long time", "run": "sleep"},
            },
        },
    },
}


def walk(argv: list[str]):
    node, path, rest, flags = TREE, [], [], []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-"):
            name = tok.split("=", 1)[0]
            spec = find_flag(node, name)
            flags.append((name, spec))
            if spec and spec[2] and "=" not in tok:
                i += 1
        elif not rest and (child := find_child(node, tok)):
            path.append(child)
            node = node["children"][child]
        else:
            rest.append(tok)
        i += 1
    return node, path, rest, flags


def find_child(node: dict, tok: str) -> str | None:
    for name, ch in node.get("children", {}).items():
        if tok == name or tok in ch.get("aliases", []):
            return name
    return None


def all_flags(node: dict) -> list:
    extra = URFAVE_GLOBALS if node.get("style") == "urfave" else GLOBAL_FLAGS
    return [*node.get("flags", []), *extra, ("h", "help", None, "help")]


def find_flag(node: dict, name: str):
    for f in all_flags(node):
        if name == f"--{f[1]}" or (f[0] and name == f"-{f[0]}"):
            return f
    return None


def runnable(node: dict) -> bool:
    return "run" in node or node.get("runnable", False)


def cobra_help(node: dict, path: list[str]) -> str:
    full = " ".join(["langsmith", *path])
    out = [node["summary"], "", "Usage:"]
    if runnable(node):
        out.append(f"  {' '.join([full, *node.get('args', [])])} [flags]")
    if node.get("children"):
        out.append(f"  {full} [command]")
    if node.get("aliases"):
        out += ["", "Aliases:", f"  {', '.join([path[-1], *node['aliases']])}"]
    if node.get("children"):
        out += ["", "Available Commands:"]
        width = max(len(n) for n in node["children"]) + 1  # cobra: longest name + 1 space
        out += [f"  {n:<{width}}{c['summary']}" for n, c in node["children"].items()]
    flag_lines = [("h", "help", None, f"help for {path[-1] if path else 'langsmith'}"), *node.get("flags", [])]
    if not path:
        flag_lines += GLOBAL_FLAGS
    out += ["", "Flags:"] + [fmt_cobra_flag(f) for f in flag_lines]
    if path:
        out += ["", "Global Flags:"] + [fmt_cobra_flag(f) for f in GLOBAL_FLAGS]
    if node.get("children"):
        out += ["", f'Use "{full} [command] --help" for more information about a command.']
    return "\n".join(out) + "\n"


def fmt_cobra_flag(f) -> str:
    short, name, typ, desc = f
    head = (f"  -{short}, " if short else "      ") + f"--{name}" + (f" {typ}" if typ else "")
    return f"{head:<30}{desc}"


def urfave_help(node: dict, path: list[str]) -> str:
    # Generated commands render their path with a colon, like the real CLI.
    full = "langsmith " + ":".join(path[:2]) + (" " + " ".join(path[2:]) if len(path) > 2 else "")
    title = f"{full} - {node['summary']}" if runnable(node) else full
    usage = f"{full} [options]" if runnable(node) else f"{full} [command [command options]]"
    out = ["NAME:", f"   {title}", "", "USAGE:", f"   {usage}", ""]
    if node.get("children"):
        out += ["COMMANDS:"] + [f"   {n}  {c['summary']}" for n, c in node["children"].items()] + [""]
    opts = [*node.get("flags", []), (None, "help", None, "show help")]
    out += ["OPTIONS:"] + [fmt_urfave_flag(f) for f in opts] + [""]
    out += ["GLOBAL OPTIONS:"] + [fmt_urfave_flag(f) for f in URFAVE_GLOBALS]
    return "\n".join(out) + "\n"


def fmt_urfave_flag(f) -> str:
    short, name, typ, desc = f
    head = f"   --{name}" + (f" {typ}" if typ else "") + (", -h" if name == "help" else "")
    return f"{head:<28}{desc}"


def record_create(name: str) -> str:
    """With a file:// endpoint, creates land in a JSON file the tests' fake SDK client shares."""
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "")
    if not endpoint.startswith("file://"):
        return "d2"
    path = endpoint[len("file://"):]
    data = json.load(open(path)) if os.path.exists(path) else []
    new_id = f"cli-{len(data) + 1}"
    data.append({"id": new_id, "name": name})
    json.dump(data, open(path, "w"))
    return new_id


def flag_value(argv: list[str], flag: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def main(argv: list[str]) -> int:
    if argv == ["--version"]:
        print("langsmith version 0.0.1-fake")
        return 0
    if argv[:1] == ["help"]:
        argv = [*argv[1:], "--help"]
    node, path, rest, flags = walk(argv)
    urfave = node.get("style") == "urfave"

    if any(name in ("-h", "--help") for name, _ in flags):
        print(urfave_help(node, path) if urfave else cobra_help(node, path), end="")
        return 0
    for name, spec in flags:
        if spec is None:
            print("Tip: report CLI friction with 'langsmith report \"note\"'.", file=sys.stderr)
            print(f"flag provided but not defined: {name}" if urfave else f"unknown flag: {name}", file=sys.stderr)
            return 1
    if not runnable(node):
        if not path and rest:
            print(f'Error: unknown command "{rest[0]}" for "langsmith"', file=sys.stderr)
            return 1
        print(urfave_help(node, path) if urfave else cobra_help(node, path), end="")
        return 0  # like the real CLI: unknown subcommand of a group prints help, exit 0

    required = [a for a in node.get("args", []) if not a.startswith("[")]
    if len(rest) < len(required):
        print(f"Error: accepts {len(required)} arg(s), received {len(rest)}", file=sys.stderr)
        return 1
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("not authenticated; run 'langsmith auth login', set LANGSMITH_API_KEY, or pass --api-key", file=sys.stderr)
        return 1

    run = node.get("run", "echo")
    if os.environ["LANGSMITH_API_KEY"] == "wrong-key":
        print('listing: GET "https://api.example/api/v1/x": 403 Forbidden', file=sys.stderr)
        return 1
    if run == "datasets_list":
        print(json.dumps([{"id": "d1", "name": "fixture-dataset"}]))
    elif run == "datasets_create":
        name = flag_value(argv, "--name")
        if not name:
            print('Error: required flag(s) "name" not set', file=sys.stderr)
            return 1
        if os.environ.get("FAKE_CREATE_FAILS"):
            print("Error: 409 Conflict: dataset already exists", file=sys.stderr)
            return 1
        new_id = record_create(name)
        if flag_value(argv, "--format") == "json":
            print(json.dumps({"id": new_id, "name": name}))
        else:
            print(f"Created dataset {name}")
    elif run == "forbidden":
        print("Error: 403 Forbidden: this API key cannot access projects", file=sys.stderr)
        return 1
    elif run == "needs_dataset":
        print("Error: required flag(s) \"dataset\" not set", file=sys.stderr)
        return 1
    elif run == "env":
        print(json.dumps(dict(os.environ)))
    elif run == "pwd":
        print(os.getcwd())
    elif run == "sleep":
        time.sleep(30)
    else:
        print(json.dumps({"ok": True, "path": " ".join(path), "args": rest}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
