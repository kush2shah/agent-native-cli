"""The agent's only tool: run the target CLI under an allowlist.

`decide()` is the policy, a pure function so it's easy to test. `CliRunner` runs
allowed commands in a sandbox (temp HOME and cwd, minimal env, closed stdin,
timeout) and logs every call, allowed or denied, to commands.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from tester.classify import DESTRUCTIVE, READ, WRITE, hard_deny_reason
from tester.config import Config, build_subprocess_env

if TYPE_CHECKING:
    from tester.inventory import Inventory

# Target-switching flags, plus --debug (can log request headers, i.e. the key).
DENIED_FLAGS = {"--workspace", "--profile", "--api-url", "--api-key", "--base-url", "--debug"}
HELP_FLAGS = {"-h", "--help"}
_WORD = re.compile(r"^[a-z0-9][a-z0-9-]*$")
LOG_HEAD_CHARS = 2000


@dataclass
class Decision:
    allowed: bool
    path: str  # resolved command path, "" for the root
    cls: str
    reason: str
    is_help: bool = False


@dataclass
class CliResult:
    argv: list[str]
    decision: Decision
    exit_code: int | None = None  # None when denied or timed out
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False

    @property
    def allowed(self) -> bool:
        return self.decision.allowed

    def log_record(self) -> dict:
        return {
            "argv": self.argv,
            **{k: v for k, v in asdict(self.decision).items()},
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "stdout_chars": len(self.stdout),
            "stderr_chars": len(self.stderr),
            "stdout_head": self.stdout[:LOG_HEAD_CHARS],
            "stderr_head": self.stderr[:LOG_HEAD_CHARS],
        }


# --- policy -----------------------------------------------------------------


def _bad_arg(tok: str) -> str | None:
    if "\x00" in tok:
        return "argument contains a NUL byte"
    if tok.startswith("@") or "=@" in tok:
        return f"file references are not allowed: {tok!r}"
    values = [tok]
    if tok.startswith("-") and "=" in tok:
        values.append(tok.split("=", 1)[1])
    for v in values:
        if v.startswith(("/", "~")):
            return f"absolute or home paths are not allowed: {tok!r}"
        if ".." in Path(v).parts:
            return f"paths may not contain `..`: {tok!r}"
    return None


def _flag_name(tok: str) -> str:
    return tok.split("=", 1)[0]


def resolve(args: list[str], inv: Inventory) -> tuple[list[str], list[str], list[str]]:
    """Split argv into (command path tokens, other positionals, flag names).

    Mirrors cobra loosely: positionals extend the path while they name a known
    subcommand (aliases resolve to the canonical name); a flag known to take a
    value consumes the next token.
    """
    path: list[str] = []
    rest: list[str] = []
    flags: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            rest.extend(args[i + 1 :])
            break
        if tok.startswith("-") and tok != "-":
            flags.append(_flag_name(tok))
            if "=" not in tok and inv.flag_takes_value(" ".join(path), tok):
                i += 1  # skip the value
        elif not rest and (name := inv.child(" ".join(path), tok)):
            path.append(name)
        else:
            rest.append(tok)
        i += 1
    return path, rest, flags


def decide(args: list[str], inventory: Inventory | None, allowed_writes: Iterable[str] = ()) -> Decision:
    for tok in args:
        if reason := _bad_arg(tok):
            return Decision(False, "", "n/a", reason)
    for tok in args:
        if tok.startswith("-") and _flag_name(tok) in DENIED_FLAGS:
            return Decision(False, "", "n/a", f"target-switching flag not allowed: {_flag_name(tok)}")

    if inventory is None:
        # Bootstrap mode (help crawl): only `<words...> --help` and `--version`.
        if args == ["--version"]:
            return Decision(True, "", READ, "version", is_help=True)
        if args and args[-1] == "--help" and all(_WORD.match(t) for t in args[:-1]):
            return Decision(True, " ".join(args[:-1]), READ, "help", is_help=True)
        return Decision(False, "", "unknown", "no inventory loaded; only --help is allowed")

    path_toks, rest, flags = resolve(args, inventory)
    path = " ".join(path_toks)

    is_help = any(f in HELP_FLAGS for f in flags) or (not path_toks and rest[:1] == ["help"])
    if reason := hard_deny_reason(path):
        # `--help` on a denied command is fine, but only in its bare form, so a
        # command that passes flags through can't smuggle a real call past us.
        if is_help and not rest and len(flags) == 1:
            return Decision(True, path, READ, "help", is_help=True)
        return Decision(False, path, DESTRUCTIVE, reason)
    if is_help:
        return Decision(True, path, READ, "help", is_help=True)

    node = inventory.get(path)
    if node is None:
        return Decision(False, path, "unknown", "command not in inventory")
    if not node["runnable"]:
        if rest:
            # Don't trust the inventory to be complete: a subcommand we didn't
            # crawl could be anything, so it's denied rather than passed through.
            return Decision(False, path, "unknown", f"unknown subcommand {rest[0]!r} of `{path or 'langsmith'}`")
        return Decision(True, path, READ, "group command: prints help")

    cls = node["class"]
    if cls == READ:
        return Decision(True, path, cls, "read")
    if cls == WRITE:
        if path in set(allowed_writes):
            return Decision(True, path, cls, "write allowed by task contract")
        return Decision(False, path, cls, "write not in the task's allowed_writes")
    if cls == DESTRUCTIVE:
        return Decision(False, path, cls, "destructive commands are never run by the agent")
    return Decision(False, path, cls, "unclassified command (unknown means denied)")


# --- runner -----------------------------------------------------------------


@dataclass
class CliRunner:
    cfg: Config
    inventory: Inventory | None = None
    allowed_writes: tuple[str, ...] = ()
    log_path: Path | None = None
    max_calls: int | None = None
    label: str = ""
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="cli-tester-")
        root = Path(self._tmp.name)
        self.home = root / "home"
        self.workdir = root / "work"
        self.home.mkdir()
        self.workdir.mkdir()
        self.env = build_subprocess_env(self.cfg, self.home)
        self._lock = threading.Lock()

    def __enter__(self) -> CliRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._tmp.cleanup()

    def run(self, args: list[str]) -> CliResult:
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise TypeError("run_cli takes a list of strings")
        with self._lock:
            self.calls += 1
            over_budget = self.max_calls is not None and self.calls > self.max_calls
        if over_budget:
            decision = Decision(False, "", "n/a", f"budget: max {self.max_calls} CLI calls reached")
        else:
            decision = decide(args, self.inventory, self.allowed_writes)

        result = CliResult(argv=args, decision=decision)
        if decision.allowed:
            self._execute(result)
        self._log(result)
        return result

    def _execute(self, result: CliResult) -> None:
        start = time.monotonic()
        proc = subprocess.Popen(
            [str(self.cfg.cli_bin), *result.argv],
            env=self.env,
            cwd=self.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,  # own process group, so a timeout kills children too
        )
        try:
            result.stdout, result.stderr = proc.communicate(timeout=self.cfg.cli_timeout_s)
            result.exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            result.stdout, result.stderr = proc.communicate()
            result.timed_out = True
        result.duration_s = time.monotonic() - start

    def _log(self, result: CliResult) -> None:
        if not self.log_path:
            return
        record = {"ts": time.time(), "label": self.label, **result.log_record()}
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
