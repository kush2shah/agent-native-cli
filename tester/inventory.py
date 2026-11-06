"""Discover: crawl `--help`, parse it, classify every command, probe cheap reads.

The LangSmith CLI mixes two help formats: hand-written cobra commands
(`Usage:` / `Available Commands:` / `Flags:`) and generated API-resource
commands in urfave/cli style (`USAGE:` / `COMMANDS:` / `OPTIONS:`).
`parse_help` handles both.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import yaml

from tester.classify import READ, UNKNOWN, classify, load_overrides, suggest_classes
from tester.config import REPO_ROOT, Config
from tester.run_cli import CliRunner

SKIP_CRAWL = {"help"}  # cobra's built-in `help` command
MAX_NODES = 3000
PROBE_VERBS = {"list", "ls"}
PERMISSION_RE = re.compile(
    r"\b(401|403)\b|forbidden|unauthori[sz]ed|permission denied|not authorized|insufficient permission", re.I
)

_HEADER = re.compile(r"^([A-Za-z][A-Za-z ]*):\s*$")
# Cobra pads names to the longest one plus a single space, so the longest name
# has just one space before its summary.
_COBRA_CMD = re.compile(r"^  (?P<name>[\w][\w-]*)(?:\s+(?P<summary>.*))?$")
_COBRA_FLAG = re.compile(r"^\s+(?:-(?P<short>\w), )?--(?P<name>[\w-]+)(?: (?P<type>\w+))?(?:\s{2,}(?P<desc>.*))?$")
_URFAVE_CMD = re.compile(r"^ {3}(?P<names>[\w][\w-]*(?:, [\w][\w-]*)*)\s{2,}(?P<summary>.*)$")
_URFAVE_FLAG = re.compile(
    r"^\s+--(?P<name>[\w-]+)(?: (?P<type>[a-z]\w*))?(?:, -(?P<short>\w))?(?:\s{2,}(?P<desc>.*))?\s*$"
)
_PLACEHOLDER = re.compile(r"^(<[^>]+>|[A-Z][A-Z0-9_/-]*)$")  # NAME_OR_ID, OWNER/REPO, <id>


# --- parsing ----------------------------------------------------------------


def _sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Split help text into (preamble lines, {HEADER: lines})."""
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] = preamble
    for line in text.splitlines():
        if m := _HEADER.match(line):
            current = sections.setdefault(m.group(1).upper(), [])
        else:
            current.append(line)
    return preamble, sections


def _flag(m: re.Match) -> dict:
    return {"name": m["name"], "short": m["short"], "type": m["type"]}


def parse_help(text: str) -> dict:
    preamble, sec = _sections(text)
    fmt = "urfave" if "NAME" in sec else "cobra"
    usage = [l.strip() for l in sec.get("USAGE", []) if l.strip()]
    children: list[dict] = []
    aliases: list[str] = []
    flags: list[dict] = []

    if fmt == "cobra":
        summary = next((l.strip() for l in preamble if l.strip()), "")
        for header, lines in sec.items():
            if header.endswith("COMMANDS"):
                for l in lines:
                    if m := _COBRA_CMD.match(l):
                        children.append({"name": m["name"], "summary": (m["summary"] or "").strip(), "aliases": []})
        for l in sec.get("ALIASES", []):
            aliases += [a.strip() for a in l.split(",") if a.strip()]
        for header in ("FLAGS", "GLOBAL FLAGS"):
            flags += [_flag(m) for l in sec.get(header, []) if (m := _COBRA_FLAG.match(l))]
        runnable = any(not u.endswith("[command]") for u in usage)
    else:
        name_line = next((l.strip() for l in sec.get("NAME", []) if l.strip()), "")
        summary = name_line.split(" - ", 1)[1] if " - " in name_line else ""
        for l in sec.get("COMMANDS", []):
            if m := _URFAVE_CMD.match(l):
                first, *rest = [n.strip() for n in m["names"].split(",")]
                children.append({"name": first, "summary": m["summary"].strip(), "aliases": rest})
        for header in ("OPTIONS", "GLOBAL OPTIONS"):
            flags += [_flag(m) for l in sec.get(header, []) if (m := _URFAVE_FLAG.match(l))]
        runnable = not any("[command" in u for u in usage)

    required_args: list[str] = []
    for u in usage[:1]:
        toks = u.split()
        # `--path <path>` is a flag value, not a positional
        required_args = [
            t for i, t in enumerate(toks) if _PLACEHOLDER.match(t) and not (i and toks[i - 1].startswith("-"))
        ]

    return {
        "format": fmt,
        "summary": summary,
        "usage": usage,
        "runnable": runnable,
        "children": children,
        "aliases": [a for a in aliases[1:]],  # cobra lists the name itself first
        "required_args": required_args,
        "flags": flags,
    }


# --- inventory model --------------------------------------------------------


class Inventory:
    """Lookup helpers over inventory.json. The root command has path ""."""

    def __init__(self, data: dict):
        self.data = data
        self.by_path = {c["path"]: c for c in data["commands"]}
        self.global_flags = data.get("global_flags", [])
        self._child: dict[tuple[str, str], str] = {}
        for c in data["commands"]:
            for ch in c["children"]:
                for name in [ch["name"], *ch.get("aliases", [])]:
                    self._child[(c["path"], name)] = ch["name"]
            # cobra puts a command's aliases on the command itself
            parent = c["path"].rpartition(" ")[0]
            for alias in c.get("aliases", []):
                self._child.setdefault((parent, alias), c["path"].split()[-1])

    @classmethod
    def load(cls, path: Path) -> Inventory:
        return cls(json.loads(path.read_text()))

    def get(self, path: str) -> dict | None:
        return self.by_path.get(path)

    def child(self, parent: str, token: str) -> str | None:
        """Canonical child name if `token` names a subcommand (or alias) of `parent`."""
        name = self._child.get((parent, token))
        return name if name and self.has(f"{parent} {name}".strip()) else None

    def has(self, path: str) -> bool:
        return path in self.by_path

    def flag_takes_value(self, path: str, tok: str) -> bool:
        node = self.by_path.get(path) or {}
        name = tok.split("=", 1)[0]
        for f in [*node.get("flags", []), *self.global_flags]:
            if name == f"--{f['name']}" or (f["short"] and name == f"-{f['short']}"):
                return bool(f["type"]) and f["type"] != "bool"
        return False


# --- crawl, classify, probe -------------------------------------------------


def _help_node(runner: CliRunner, path: str) -> dict:
    res = runner.run([*path.split(), "--help"])
    text = res.stdout or res.stderr
    node = {"path": path, "help": text, **parse_help(text)}
    if not res.allowed or res.exit_code != 0 or not text.strip():
        node["crawl_error"] = res.decision.reason if not res.allowed else f"exit {res.exit_code}"
    return node


def crawl(runner: CliRunner, workers: int = 8) -> list[dict]:
    nodes: dict[str, dict] = {}
    with ThreadPoolExecutor(workers) as pool:
        pending = {pool.submit(_help_node, runner, ""): ""}
        seen = {""}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                path = pending.pop(fut)
                node = nodes[path] = fut.result()
                for ch in node["children"]:
                    if not path and ch["name"] in SKIP_CRAWL:
                        continue
                    child_path = f"{path} {ch['name']}".strip()
                    if child_path in seen or len(seen) >= MAX_NODES:
                        continue
                    seen.add(child_path)
                    pending[pool.submit(_help_node, runner, child_path)] = child_path

    # Generated group commands have no summary of their own; use the parent's listing.
    for node in nodes.values():
        for ch in node["children"]:
            child = nodes.get(f"{node['path']} {ch['name']}".strip())
            if child and not child["summary"]:
                child["summary"] = ch["summary"]
    return [nodes[p] for p in sorted(nodes)]


def _first_line(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("Tip:")]
    return lines[0][:300] if lines else ""


def probe(runner: CliRunner, inv: Inventory, workers: int = 8) -> None:
    """Run cheap, argument-free list commands to find what the key can't read.

    Mutates each command's `probe` / `probe_detail` in place.
    """
    targets = []
    for c in inv.data["commands"]:
        if c["path"] and c["class"] == READ and c["runnable"] and not c["children"] and not c["required_args"]:
            if c["path"].split()[-1] in PROBE_VERBS:
                targets.append(c)

    def one(c: dict) -> None:
        args = [*c["path"].split(), "--format", "json"]
        if any(f["name"] == "limit" for f in c["flags"]):
            args += ["--limit", "1"]
        res = runner.run(args)
        out = f"{res.stderr}\n{res.stdout}"
        if not res.allowed:
            c["probe"], c["probe_detail"] = "error", f"denied: {res.decision.reason}"
        elif res.exit_code == 0:
            c["probe"], c["probe_detail"] = "ok", ""
        elif PERMISSION_RE.search(out):
            c["probe"], c["probe_detail"] = "permission_denied", _first_line(res.stderr or res.stdout)
        else:
            detail = "timed out" if res.timed_out else _first_line(res.stderr or res.stdout)
            c["probe"], c["probe_detail"] = "error", detail

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, targets))


def discover(
    cfg: Config,
    run_id: str,
    *,
    use_llm: bool = True,
    do_probe: bool = True,
    overrides_path: Path = REPO_ROOT / "classifications.yaml",
    model=None,
) -> dict:
    run_dir = cfg.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(overrides_path)
    notes: list[str] = []

    with CliRunner(cfg, log_path=run_dir / "commands.jsonl", label="discover") as runner:
        version = runner.run(["--version"]).stdout.strip()
        commands = crawl(runner)

        for c in commands:
            c["class"], c["class_source"] = classify(c["path"], c["runnable"], overrides)
            c["suggested_class"] = None
            c["probe"], c["probe_detail"] = "not_probed", ""

        known = {c["path"] for c in commands}
        if stale := sorted(set(overrides) - known):
            notes.append(f"overrides for commands not in the CLI (ignored): {', '.join(stale)}")

        root = next(c for c in commands if c["path"] == "")
        data = {
            "cli_version": version,
            "cli_bin": str(cfg.cli_bin),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "global_flags": [f for f in root["flags"] if f["name"] not in ("help", "version")],
            "commands": commands,
            "notes": notes,
        }

        unknown = [c for c in commands if c["class"] == UNKNOWN]
        if use_llm and unknown and (model is not None or cfg.model):
            try:
                if model is None:
                    from langchain.chat_models import init_chat_model

                    model = init_chat_model(cfg.model)
                suggestions = suggest_classes(unknown, model)
            except Exception as e:  # suggestions are optional; never block discovery
                notes.append(f"LLM suggestions skipped: {type(e).__name__}: {e}")
                suggestions = {}
            for c in unknown:
                if s := suggestions.get(c["path"]):
                    c["suggested_class"] = s["class"]
                    c["suggested_reason"] = s["reason"]
            if suggestions:
                _write_suggestions(run_dir / "suggested_classifications.yaml", unknown)

        if do_probe:
            if cfg.has_target_key:
                runner.inventory = Inventory(data)
                probe(runner, runner.inventory)
                results = [c["probe"] for c in commands if c["probe"] != "not_probed"]
                if results and "ok" not in results and "permission_denied" in results:
                    # This CLI answers a wrong key (or wrong region) with 403, which
                    # would otherwise read as "everything is a blind spot".
                    notes.append(
                        "WARNING: every probe was refused. The key is probably wrong, for another "
                        "workspace, or for another region. Fix it before trusting any blind spots."
                    )
            else:
                notes.append("probes skipped: TARGET_API_KEY is not set")

    (run_dir / "inventory.json").write_text(json.dumps(data, indent=2))
    return data


def _write_suggestions(path: Path, unknown: list[dict]) -> None:
    lines = [
        "# LLM suggestions for commands the rules couldn't classify. NOT applied.",
        "# Review each one and copy the ones you agree with into classifications.yaml.",
        "",
    ]
    for c in unknown:
        if c.get("suggested_class"):
            lines.append(f"# {c['summary'][:100]}")
            lines.append(f"# why: {c.get('suggested_reason', '')[:160]}")
            lines.append(yaml.safe_dump({c["path"]: c["suggested_class"]}).strip())
            lines.append("")
    path.write_text("\n".join(lines))
