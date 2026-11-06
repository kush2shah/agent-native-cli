"""Command classification: hard denylist > human overrides > verb rules > unknown.

The LLM never decides a class. It only writes suggestions for `unknown` commands,
which a human may copy into classifications.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml

READ, WRITE, DESTRUCTIVE, UNKNOWN = "read", "write", "destructive", "unknown"
CLASSES = (READ, WRITE, DESTRUCTIVE, UNKNOWN)

# Command-path prefixes that are always denied. Nothing overrides these.
HARD_DENY_PREFIXES = {
    "api": "raw authenticated HTTP requests bypass command classification",
    "auth": "credential management",
    "profiles": "credential and profile management",
    "connector": "long-running service",
    "relay": "long-running service",
    "tunnel": "long-running service",
    "sandboxes": "starts remote compute and interactive sessions",
    "self-update": "modifies the CLI binary",
    "completion": "shell setup, not a workspace operation",
    "report": "sends feedback outside the workspace",
    "product-feedback": "sends feedback outside the workspace",
    "traces setup": "configures local coding agents",
    "workspaces create": "operates outside the one target workspace",
    "workspaces update": "operates outside the one target workspace",
}

# Path segments that are always denied, wherever they appear.
HARD_DENY_SEGMENTS = {
    "share": "makes data public",
    "select": "changes saved defaults",
    "login": "credential management",
    "logout": "credential management",
    "set-workspace": "changes saved defaults",
}

READ_VERBS = {
    "list", "ls", "get", "show", "view", "describe", "info", "stats", "query", "search",
    "resolve", "dashboard", "messages", "get-url", "retrieve-version", "events", "export",
    "pull", "download", "diff", "status", "whoami", "version", "compare", "preview", "logs",
    "inspect", "count", "lookup", "history", "read", "retrieve",
}
WRITE_VERBS = {
    "create", "add", "update", "upload", "push", "clone", "tag", "update-tags", "ingest-batch",
    "edit", "rename", "commit", "import", "append", "attach", "link", "patch", "set", "upsert",
    "run", "start", "trigger", "invoke", "publish", "copy", "duplicate", "enable", "restore",
    "annotate", "label", "submit",
}
DESTRUCTIVE_VERBS = {
    "delete", "rm", "remove", "archive", "purge", "clear", "reset", "drop", "unlink", "detach",
    "revoke", "unshare", "cancel", "stop", "kill", "destroy", "disable", "wipe", "prune",
    "serve", "deploy", "use", "set-default", "replace", "overwrite", "move", "mv",
}


def hard_deny_reason(path: str) -> str | None:
    tokens = path.split()
    for prefix, reason in HARD_DENY_PREFIXES.items():
        p = prefix.split()
        if tokens[: len(p)] == p:
            return f"hard denylist: `{prefix}` ({reason})"
    for tok in tokens:
        if tok in HARD_DENY_SEGMENTS:
            return f"hard denylist: `{tok}` ({HARD_DENY_SEGMENTS[tok]})"
    return None


def verb_class(path: str) -> str | None:
    """Class from the last path segment. Hyphenated verbs match on any part, and
    the most dangerous part wins: `bulk-delete` is destructive, `aggregate-stats` is read."""
    verb = path.split()[-1]
    parts = {verb, *verb.split("-")}
    for verbs, cls in ((DESTRUCTIVE_VERBS, DESTRUCTIVE), (WRITE_VERBS, WRITE), (READ_VERBS, READ)):
        if parts & verbs:
            return cls
    return None


def classify(path: str, runnable: bool, overrides: dict[str, str]) -> tuple[str, str]:
    """Return (class, source) for one command path."""
    if hard_deny_reason(path):
        return DESTRUCTIVE, "hard_deny"
    if path in overrides:
        return overrides[path], "override"
    if not runnable:
        return READ, "group"  # can only print help
    if cls := verb_class(path):
        return cls, "rule:verb"
    return UNKNOWN, "default"


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    bad = {k: v for k, v in data.items() if v not in (READ, WRITE, DESTRUCTIVE)}
    if bad:
        raise ValueError(f"{path}: classes must be read|write|destructive, got {bad}")
    return {str(k): v for k, v in data.items()}


# --- LLM suggestions (advisory only) ------------------------------------------

SUGGEST_BATCH = 40
PROMPT_PATH = Path(__file__).parent / "prompts" / "classifier.md"


def suggest_classes(commands: list[dict], model) -> dict[str, dict]:
    """Ask the model to suggest a class for each command. Returns {path: {class, reason}}.

    The caller records these as suggestions only; they never change `class`.
    """
    from typing import Literal

    from pydantic import BaseModel

    class Suggestion(BaseModel):
        path: str
        classification: Literal["read", "write", "destructive"]
        reason: str

    class Suggestions(BaseModel):
        suggestions: list[Suggestion]

    structured = model.with_structured_output(Suggestions, method="json_schema")
    system = PROMPT_PATH.read_text()
    out: dict[str, dict] = {}
    for i in range(0, len(commands), SUGGEST_BATCH):
        batch = commands[i : i + SUGGEST_BATCH]
        listing = "\n\n".join(f"### `{c['path']}`\n{c['help'][:1500]}" for c in batch)
        result = structured.invoke([("system", system), ("user", listing)])
        wanted = {c["path"] for c in batch}
        for s in result.suggestions:
            if s.path in wanted:
                out[s.path] = {"class": s.classification, "reason": s.reason}
    return out
