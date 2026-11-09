"""Turn run folders into report.md: what wasn't tested, where agents wasted
calls, how they behaved on risky requests, and how the arms compared.

Everything except the suggested fixes is computed from the run files
(results.jsonl, commands.jsonl, reset.json) and the inventory. The fixes
come from one structured model call over the grouped evidence, and can be
skipped with --no-llm.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field

from tester.inventory import Inventory
from tester.worker import MAX_TOOL_OUTPUT

PROMPTS = Path(__file__).parent / "prompts"

# What an agent-native CLI gets right. Findings are tagged with one of these.
TENETS = {
    "learnable": "Learnable from --help alone: an agent with no prior knowledge finds the right command without docs.",
    "fails_loudly": "Wrong input fails loudly: an invented command or flag exits non-zero, with an error on stderr that says what's wrong and how to fix it.",
    "help_is_true": "Help matches behavior: usage lines run as written, stated defaults are real, limits are documented.",
    "parseable_output": "Output is machine-readable and bounded: JSON output is valid JSON, and lists can be limited.",
    "consistent": "Consistent conventions: what an agent learns on one command transfers to the next.",
    "addressable": "Objects are addressable by a name the agent can type: identifiers it sees can be passed back as-is.",
    "concepts_map": "Commands follow the product's concepts: a request in product terms leads to the right commands.",
    "clear_permissions": "Permission errors are recognizable: \"you can't\" is distinguishable from \"you did it wrong\".",
}

TIP_LINE = re.compile(r"^Tip: report CLI friction")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
URL = re.compile(r"https?://\S+?(?=[\"'\s:]|$)")
NUMBER = re.compile(r"\b\d{5,}\b|(?<=\()[^)]*\d[^)]*(?=\))")  # long numbers and parenthesized values (the instance's own figures)
HTTP_STATUS = re.compile(r"\": (\d{3}) [A-Z]")
DETAIL = re.compile(r'"detail":"((?:[^"\\]|\\.)*)"')
PERMISSION_WORDS = re.compile(r"forbidden|unauthorized|permission denied|not authenticated", re.I)
INVENTED_FLAG = re.compile(r"unknown (shorthand )?flag|flag provided but not defined")

HARNESS_DENIAL_CLASS = "n/a"  # sandbox argument checks: the harness's doing, not the CLI's


# --- loading ----------------------------------------------------------------


@dataclass
class Run:
    run_id: str
    path: Path
    results: list[dict]
    commands: list[dict]
    reset: dict | None

    @classmethod
    def load(cls, path: Path) -> Run:
        def jsonl(name: str) -> list[dict]:
            f = path / name
            return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []

        reset = json.loads((path / "reset.json").read_text()) if (path / "reset.json").exists() else None
        return cls(path.name, path, jsonl("results.jsonl"), jsonl("commands.jsonl"), reset)

    def worker(self, label: str) -> str:
        return f"{self.run_id}:{label}"


# --- classifying calls ------------------------------------------------------


def error_lines(c: dict) -> tuple[list[str], str]:
    """The meaningful error lines of a failed call, and which stream they came from."""
    err = [ln for ln in c.get("stderr_head", "").splitlines() if ln.strip() and not TIP_LINE.match(ln)]
    if err:
        return err, "stderr"
    out = [ln for ln in c.get("stdout_head", "").splitlines() if ln.strip()]
    return out, "stdout" if out else "none"


def signature(c: dict, run_id: str) -> str:
    """A failed call's error, with names, IDs, URLs and numbers normalized so
    the same problem groups together across tasks and runs."""
    if not c.get("allowed"):
        text = c.get("reason", "")
    else:
        lines, _ = error_lines(c)
        joined = "\n".join(lines)
        detail, status = DETAIL.search(joined), HTTP_STATUS.search(joined)
        text = detail.group(1) if detail else (
            lines[0] if lines else ("timed out" if c.get("timed_out") else f"exit {c.get('exit_code')}, no output"))
        prefix = f"{status.group(1)}: " if detail and status else ""
    if not c.get("allowed"):
        prefix = ""
    if run_id:
        text = re.sub(rf"{re.escape(run_id)}[\w.-]*", "<name>", text)
    text = UUID.sub("<id>", text)
    text = URL.sub("<url>", text)
    text = NUMBER.sub("…", text)
    return (prefix + text.strip())[:200]


def is_permission_error(c: dict) -> bool:
    joined = "\n".join(error_lines(c)[0])
    status = HTTP_STATUS.search(joined)
    return (status.group(1) in ("401", "403")) if status else bool(PERMISSION_WORDS.search(joined))


def kind(c: dict) -> str:
    """help | ok | cli_error | permission | invented_subcommand | policy | harness | budget"""
    if c.get("allowed"):
        if c.get("is_help"):
            return "help"
        if c.get("exit_code") == 0 and not c.get("timed_out"):
            return "ok"
        return "permission" if is_permission_error(c) else "cli_error"
    reason = c.get("reason", "")
    if reason.startswith("budget"):
        return "budget"
    if reason.startswith("unknown subcommand") or reason == "command not in inventory":
        return "invented_subcommand"
    if c.get("cls") == HARNESS_DENIAL_CLASS:
        return "harness"
    return "policy"


# --- sections ---------------------------------------------------------------


@dataclass
class Group:
    path: str
    signature: str
    calls: list[tuple[Run, dict]] = field(default_factory=list)
    recovered: int = 0  # failed calls after which the same worker later ran this path successfully
    stream: str = ""

    @property
    def workers(self) -> set[str]:
        return {run.worker(c["label"]) for run, c in self.calls}

    @property
    def tasks(self) -> set[str]:
        return {f"{run.run_id}:{c['label'].split('/')[0]}" for run, c in self.calls}

    def examples(self, n: int = 2) -> list[dict]:
        seen, out = set(), []
        for run, c in self.calls:
            key = tuple(c["argv"])
            if key in seen:
                continue
            seen.add(key)
            lines, _ = error_lines(c) if c.get("allowed") else ([c.get("reason", "")], "")
            out.append({"argv": c["argv"], "error": "\n".join(lines)[:400], "worker": run.worker(c["label"])})
            if len(out) == n:
                break
        return out


def friction_groups(runs: list[Run]) -> list[Group]:
    """Failed CLI calls (errors and invented subcommands), grouped by command and error."""
    groups: dict[tuple[str, str], Group] = {}
    for run in runs:
        by_worker = defaultdict(list)
        for c in run.commands:
            by_worker[c["label"]].append(c)
        for calls in by_worker.values():
            for i, c in enumerate(calls):
                if kind(c) not in ("cli_error", "invented_subcommand"):
                    continue
                path = c.get("path") or " ".join(t for t in c["argv"] if not t.startswith("-"))[:40]
                g = groups.setdefault((path, signature(c, run.run_id)), Group(path, signature(c, run.run_id)))
                g.calls.append((run, c))
                if c.get("allowed"):
                    g.stream = error_lines(c)[1]
                if any(kind(later) == "ok" and later.get("path") == c.get("path") for later in calls[i + 1:]):
                    g.recovered += 1
    return sorted(groups.values(), key=lambda g: (-len(g.calls), -len(g.workers), g.path))


def oversized(runs: list[Run]) -> list[dict]:
    """Successful calls whose output didn't fit in the agent's tool-output window."""
    out = []
    for run in runs:
        for c in run.commands:
            if kind(c) == "ok" and c.get("stdout_chars", 0) > MAX_TOOL_OUTPUT:
                out.append({"path": c["path"], "chars": c["stdout_chars"], "argv": c["argv"], "worker": run.worker(c["label"])})
    return sorted(out, key=lambda o: -o["chars"])


def baselines(results: list[dict]) -> dict[str, int]:
    """Fewest CLI calls any passing worker needed, per task: the reference for wasted calls."""
    best: dict[str, int] = {}
    for r in results:
        if r.get("passed") and r.get("expected") == "complete":
            best[r["task_id"]] = min(best.get(r["task_id"], r["cli_calls"]), r["cli_calls"])
    return best


def failed_calls(run: Run, r: dict) -> int:
    label = f"{r['task_id']}/{r['arm']}/r{r['repeat']}"
    return sum(1 for c in run.commands if c["label"] == label and kind(c) in ("cli_error", "invented_subcommand"))


def risky_attempts(run: Run, r: dict) -> list[list[str]]:
    """Writes and destructive commands the worker tried that policy refused (from the log, since
    older results lack blocked_actions)."""
    label = f"{r['task_id']}/{r['arm']}/r{r['repeat']}"
    return [c["argv"] for c in run.commands if c["label"] == label and kind(c) == "policy"]


def not_tested(inv: Inventory, runs: list[Run]) -> dict:
    used = {c["path"] for run in runs for c in run.commands if kind(c) in ("ok", "cli_error", "permission")}
    runnable = [c for c in inv.data["commands"] if c["path"] and c.get("runnable")]
    blind = sorted({c["path"] for c in runnable if c.get("probe") == "permission_denied"}
                   | {c["path"] for run in runs for c in run.commands if kind(c) == "permission"})
    reads = sorted(c["path"] for c in runnable if c.get("class") == "read")
    writes = sorted(c["path"] for c in runnable if c.get("class") == "write")
    return {
        "blind_spots": blind,
        "reads_total": len(reads),
        "reads_untested": [p for p in reads if p not in used],
        "writes_total": len(writes),
        "writes_untested": [p for p in writes if p not in used],
        "destructive_total": sum(1 for c in runnable if c.get("class") == "destructive"),
    }


# --- suggested fixes (one model call) ---------------------------------------


class Finding(BaseModel):
    group: int = Field(description="Index of the evidence group this finding explains")
    fix_id: str = Field(description="Short kebab-case slug for the underlying fix; findings one change would solve share it")
    tenet: str = Field(description="One tenet id from the list")
    problem: str = Field(description="What the CLI (or docs) did that cost the agent calls, in one or two sentences")
    change: str = Field(description="The concrete change that would prevent it")
    owner: Literal["cli", "docs", "api", "harness"]


class Findings(BaseModel):
    findings: list[Finding]


def evidence(groups: list[Group], big: list[dict], inv: Inventory) -> str:
    parts = []
    for i, g in enumerate(groups):
        node = inv.get(g.path) or {}
        flags = ", ".join(f"--{f['name']}" for f in node.get("flags", []) if f["name"] not in ("help",))
        ex = "\n".join(f"    $ langsmith {' '.join(e['argv'])}\n    {e['error']}" for e in g.examples())
        parts.append(
            f"## Group {i}: `{g.path}`: {g.signature}\n"
            f"{len(g.calls)} failed calls by {len(g.workers)} workers; recovered afterwards in {g.recovered}. "
            f"Error printed on: {g.stream or 'n/a (denied before running)'}.\n"
            f"Help summary: {node.get('summary', '(not in inventory)')}\n"
            f"Usage: {'; '.join(node.get('usage', []))}\nFlags: {flags}\nExamples:\n{ex}")
    if big:
        parts.append("## Oversized outputs (not a group; mention in a finding only if relevant)\n" + "\n".join(
            f"- `{o['path']}`: {o['chars']:,} chars (agent window {MAX_TOOL_OUTPUT:,})" for o in big[:5]))
    return "\n\n".join(parts)


def suggest_fixes(model, groups: list[Group], big: list[dict], inv: Inventory) -> list[Finding]:
    if not groups:
        return []
    system = (PROMPTS / "report.md").read_text().format(
        tenets="\n".join(f"- `{k}`: {v}" for k, v in TENETS.items()))
    out = model.with_structured_output(Findings, method="json_schema", include_raw=True).invoke(
        [("system", system), ("user", evidence(groups, big, inv))])
    parsed = out["parsed"] if isinstance(out, dict) else out
    return [f for f in parsed.findings if 0 <= f.group < len(groups)]


# --- report data (one dict, rendered as markdown and as HTML) ---------------

MAX_EVENT_CHARS = 6000  # per transcript event in the HTML view; the full text stays in workers/*.jsonl


def label_of(r: dict) -> str:
    return f"{r['task_id']}/{r['arm']}/r{r['repeat']}"


def transcript_events(path: Path) -> list[dict]:
    """A worker transcript as display events: task, thinking, text, tool (with status), answer."""
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "start":
            events.append({"type": "task", "text": e.get("prompt", "")})
        elif e["event"] == "model":
            content = e.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for b in blocks:
                if b.get("type") in ("thinking", "text") and (b.get(b["type"]) or b.get("text")):
                    events.append({"type": b["type"], "text": (b.get(b["type"]) or b.get("text"))[:MAX_EVENT_CHARS]})
        elif e["event"] == "tool":
            out = str(e.get("output", ""))
            if out.startswith("DENIED"):
                status = "denied"
            elif out.startswith("exit code: 0"):
                status = "ok"
            else:
                status = "error"
            args = e.get("args", "")
            try:
                parsed = ast.literal_eval(args) if isinstance(args, str) else args
                args = parsed.get("args", parsed) if isinstance(parsed, dict) else parsed
            except (ValueError, SyntaxError):
                pass
            events.append({"type": "tool", "name": e.get("name", ""), "args": args, "status": status,
                           "output": out[:MAX_EVENT_CHARS], "truncated": len(out) > MAX_EVENT_CHARS})
        elif e["event"] == "end":
            events.append({"type": "answer", "text": e.get("final_answer", ""), "stop": e.get("stop_reason", "")})
    return events


def compute(runs: list[Run], inv: Inventory, groups: list[Group], big: list[dict],
            findings: list[Finding] | None, transcripts: bool = True) -> dict:
    results = [(run, r) for run in runs for r in run.results]
    all_calls = [c for run in runs for c in run.commands]
    counts = defaultdict(int)
    for c in all_calls:
        counts[kind(c)] += 1
    failed = counts["cli_error"] + counts["invented_subcommand"]

    friction = [{
        "n": i + 1, "path": g.path, "signature": g.signature, "stream": g.stream, "recovered": g.recovered,
        "workers": len(g.workers), "invented_flag": bool(INVENTED_FLAG.search(g.signature)),
        "calls": [{"worker": run.worker(c["label"]), "argv": c["argv"],
                   "error": "\n".join(error_lines(c)[0] if c.get("allowed") else [c.get("reason", "")])[:600]}
                  for run, c in g.calls],
    } for i, g in enumerate(groups)]

    fixes = None
    if findings is not None:
        merged: dict[str, list[Finding]] = defaultdict(list)  # one entry per fix, however many groups it covers
        for f in findings:
            merged[f.fix_id.strip().lower()].append(f)
        ranked = sorted(merged.values(), key=lambda fs: -sum(len(groups[f.group].calls) for f in fs))
        fixes = []
        for n, fs in enumerate(ranked, 1):
            fs.sort(key=lambda f: -len(groups[f.group].calls))
            fixes.append({
                "n": n, "change": fs[0].change, "owner": fs[0].owner, "tenet": fs[0].tenet,
                "tenet_name": TENETS.get(fs[0].tenet, fs[0].tenet).split(":")[0],
                "groups": [f.group + 1 for f in fs],
                "calls": sum(len(groups[f.group].calls) for f in fs),
                "workers": len(set().union(*(groups[f.group].workers for f in fs))),
                "problems": [{"group": f.group + 1, "problem": f.problem, "argv": groups[f.group].examples(1)[0]["argv"],
                              "signature": groups[f.group].signature} for f in fs],
            })

    traps = [{"worker": run.worker(label_of(r)), "run": run.run_id, "task": r["task_id"], "arm": r["arm"],
              "passed": bool(r.get("passed")), "tried": risky_attempts(run, r), "answer": r.get("final_answer", ""),
              "prompt": r.get("prompt", "")}
             for run, r in results if r.get("expected") == "ask_or_refuse"]

    scorecard = []
    for run in runs:
        best = baselines(run.results)
        for r in sorted(run.results, key=lambda r: (r["task_id"], r["arm"], r["repeat"])):
            scorecard.append({
                "worker": run.worker(label_of(r)), "run": run.run_id, "task": r["task_id"], "tier": r["tier"],
                "arm": r["arm"], "repeat": r["repeat"], "expected": r.get("expected"), "passed": r.get("passed"),
                "calls": r["cli_calls"], "help": r["help_calls"], "failed": failed_calls(run, r),
                "excess": r["cli_calls"] - best[r["task_id"]] if r["task_id"] in best and r.get("passed") else None,
                "stop": r["stop_reason"], "cost": r.get("cost_usd", 0), "docs_calls": r.get("docs_calls", 0),
                "prompt": r.get("prompt", ""), "checks": r.get("checks", []),
            })

    by_arm = defaultdict(list)
    for _, r in results:
        by_arm[r["arm"]].append(r)
    arms = [{"arm": arm, "passed": sum(1 for r in rs if r.get("passed")), "total": len(rs),
             "mean_calls": round(mean(r["cli_calls"] for r in rs), 1),
             "mean_help": round(mean(r["help_calls"] for r in rs), 1),
             "docs_calls": sum(r.get("docs_calls", 0) for r in rs),
             "cost": round(sum(r.get("cost_usd", 0) for r in rs), 4)} for arm, rs in sorted(by_arm.items())]
    split = defaultdict(dict)
    for run, r in results:
        split[(run.run_id, r["task_id"])].setdefault(r["arm"], []).append(bool(r.get("passed")))
    disagree = [f"{rid} {t}" for (rid, t), v in split.items() if len({all(x) for x in v.values()}) > 1]

    caveats = []
    if counts["harness"]:
        reasons = sorted({signature(c, "") for c in all_calls if kind(c) == "harness"})
        caveats.append(f"{counts['harness']} calls were denied by the tester's own sandbox, not the CLI: "
                       + "; ".join(reasons[:3]) + ". Check these aren't false positives.")
    if counts["budget"]:
        caveats.append(f"{counts['budget']} calls hit the per-task call budget.")
    stops = [f"{run.run_id} {r['task_id']}/{r['arm']} ({r['stop_reason']})" for run, r in results
             if r.get("stop_reason") != "answered"]
    if stops:
        caveats.append("Stopped before answering: " + ", ".join(stops) + ".")
    for run in runs:
        if run.reset is None:
            caveats.append(f"{run.run_id} has no reset.json: the workspace may still hold its objects.")
        elif not run.reset.get("listing", {}).get("clean"):
            caveats.append(f"{run.run_id}'s reset didn't leave the workspace clean: {json.dumps(run.reset.get('listing'))}.")
    caveats.append("The plan generator and the worker are the same model; results that look clean may share its blind spots.")

    data = {
        "runs": [r.run_id for r in runs],
        "cli_version": inv.data.get("cli_version", "unknown CLI version"),
        "summary": {
            "workers": len(results), "calls": len(all_calls), "help": counts["help"], "failed": failed,
            "work_calls": sum(counts[k] for k in ("ok", "cli_error", "invented_subcommand", "permission")),
            "cost": round(sum(r.get("cost_usd", 0) for _, r in results), 4),
            "invented_subcommands": counts["invented_subcommand"],
            "invented_flags": sum(len(f["calls"]) for f in friction if f["invented_flag"]),
            "stdout_groups": [f["n"] for f in friction if f["stream"] == "stdout"],
            "traps_acted": sum(1 for t in traps if not t["passed"]),
        },
        "not_tested": not_tested(inv, runs),
        "friction": friction,
        "oversized": big,
        "fixes": fixes,
        "findings": [f.model_dump() for f in findings] if findings is not None else None,
        "traps": traps,
        "scorecard": scorecard,
        "arms": arms,
        "docs_runs": len(by_arm.get("docs", [])),
        "docs_unused": bool(by_arm.get("docs")) and not any(r.get("docs_calls") for r in by_arm["docs"]),
        "disagree": disagree,
        "caveats": caveats,
        "tenets": TENETS,
        "max_tool_output": MAX_TOOL_OUTPUT,
    }
    if transcripts:
        data["transcripts"] = {
            run.worker(label_of(r)): transcript_events(
                run.path / "workers" / f"{r['task_id']}-{r['arm']}-r{r['repeat']}.jsonl")
            for run, r in results}
    return data


# --- markdown ---------------------------------------------------------------


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _cmd(argv: list[str]) -> str:
    return "`langsmith " + " ".join(argv).replace("|", "\\|") + "`"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _by_group(paths: list[str]) -> str:
    grouped = defaultdict(list)
    for p in paths:
        top, _, rest = p.partition(" ")
        grouped[top].append(rest or "(itself)")
    return "\n".join(f"- **{top}**: {', '.join(rest)}" for top, rest in sorted(grouped.items()))


def render_md(d: dict) -> str:
    s, nt = d["summary"], d["not_tested"]
    lines = [
        f"# CLI agent-usability report: {', '.join(d['runs'])}", "",
        f"{d['cli_version']}. {s['workers']} worker runs, {s['calls']} CLI calls ({s['help']} help lookups), "
        f"{s['failed']} failed, ${s['cost']:.2f} in worker model cost.", "",
    ]

    lines += ["## 1. Not tested", ""]
    if nt["blind_spots"]:
        lines += ["**Blind spots** (the key couldn't run these, so nothing here says whether they work):",
                  _by_group(nt["blind_spots"]), ""]
    else:
        lines += ["**Blind spots:** none. Every command the workers tried was permitted for the key.", ""]
    lines += [
        f"**Coverage:** {nt['reads_total'] - len(nt['reads_untested'])} of {nt['reads_total']} read commands and "
        f"{nt['writes_total'] - len(nt['writes_untested'])} of {nt['writes_total']} write commands were exercised. "
        f"The {nt['destructive_total']} destructive commands are never run by workers.", "",
        f"<details><summary>{len(nt['reads_untested'])} read commands never exercised</summary>",
        "", _by_group(nt["reads_untested"]), "", "</details>", "",
    ]

    lines += ["## 2. Friction: where agents wasted calls", ""]
    if d["friction"]:
        lines += [
            f"{s['failed']} of {s['work_calls']} non-help calls failed ({s['failed'] / max(s['work_calls'], 1):.0%}). "
            "Grouped by command and error, with names and numbers normalized:", "",
            "| # | Failed calls | Workers | Same command later worked | Command | Error |",
            "|---|---|---|---|---|---|",
        ]
        for f in d["friction"]:
            lines.append(f"| {f['n']} | {len(f['calls'])} | {f['workers']} | {f['recovered']}/{len(f['calls'])} | "
                         f"`{f['path']}` | {_cell(f['signature'])} |")
        lines.append("")
        notes = []
        if s["invented_subcommands"]:
            n = s["invented_subcommands"]
            notes.append(f"{n} {_plural(n, 'call')} used a subcommand that doesn't exist "
                         "(the real CLI prints help and exits 0 for these, so the agent can't tell).")
        if s["invented_flags"]:
            n = s["invented_flags"]
            notes.append(f"{n} {_plural(n, 'call')} used a flag the CLI rejected.")
        if s["stdout_groups"]:
            which = ", ".join(map(str, s["stdout_groups"]))
            notes.append(f"{'Groups' if len(s['stdout_groups']) > 1 else 'Group'} {which} printed the error on stdout; "
                         "stderr held only a \"report CLI friction\" tip.")
        lines += [f"- {n}" for n in notes] + ([""] if notes else [])
    else:
        lines += ["No failed calls.", ""]
    if d["oversized"]:
        lines += [f"**Oversized output:** {len(d['oversized'])} successful calls returned more than the agent's "
                  f"{d['max_tool_output']:,}-character window, so it saw a truncated result:", ""]
        lines += [f"- {_cmd(o['argv'])}: {o['chars']:,} chars" for o in d["oversized"][:5]] + [""]

    lines += ["## 3. Suggested changes", ""]
    if d["fixes"] is None:
        lines += ["Skipped (`--no-llm`).", ""]
    elif not d["fixes"]:
        lines += ["Nothing to suggest.", ""]
    for fx in d["fixes"] or []:
        nums = ", ".join(map(str, fx["groups"]))
        lines += [f"**{fx['n']}. {fx['change']}**", "",
                  f"Owner: {fx['owner']}. Tenet: {fx['tenet_name']}. Friction "
                  f"{'groups' if len(fx['groups']) > 1 else 'group'} {nums}: {fx['calls']} failed "
                  f"{_plural(fx['calls'], 'call')} by {fx['workers']} {_plural(fx['workers'], 'worker')}.", ""]
        for p in fx["problems"]:
            lines += [f"- {p['problem']}", f"  - e.g. {_cmd(p['argv'])} → {_cell(p['signature'])}"]
        lines.append("")

    lines += ["## 4. Behavior on risky requests", ""]
    if d["traps"]:
        lines += [f"{s['traps_acted']} of {len(d['traps'])} trap runs acted without asking first.", ""]
        for t in d["traps"]:
            verdict = "asked or refused" if t["passed"] else "acted"
            tried = "; ".join(_cmd(a) for a in t["tried"]) or "nothing risky"
            lines += [f"- **{t['run']} {t['task']} / {t['arm']}: {verdict}.** Tried: {tried}.",
                      f"  > {_cell(t['answer'][:240])}…"]
        lines.append("")
    else:
        lines += ["No trap tasks in these runs.", ""]

    lines += ["## 5. Scorecard", "",
              "Calls are CLI calls; **excess** is calls beyond the fewest any passing worker needed for that task "
              "(a weak baseline with one repeat per arm).", "",
              "| Run | Task | Tier | Arm | Verdict | Calls | Help | Failed | Excess | Stop |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["scorecard"]:
        verdict = {True: "pass", False: "**FAIL**", None: "unscored"}[r["passed"]]
        excess = "–" if r["excess"] is None else r["excess"]
        lines.append(f"| {r['run']} | {r['task']} | {r['tier']} | {r['arm']} | {verdict} | {r['calls']} | "
                     f"{r['help']} | {r['failed']} | {excess} | {r['stop']} |")
    lines.append("")
    for r in d["scorecard"]:
        for ch in r["checks"]:
            if not ch.get("passed"):
                lines.append(f"- {r['run']} {r['task']}/{r['arm']} failed check: {ch['describe']}")
    lines.append("")

    lines += ["## 6. Arms", "", "| Arm | Passed | Mean calls | Mean help | Docs calls | Cost |", "|---|---|---|---|---|---|"]
    for a in d["arms"]:
        lines.append(f"| {a['arm']} | {a['passed']}/{a['total']} | {a['mean_calls']:.1f} | {a['mean_help']:.1f} | "
                     f"{a['docs_calls']} | ${a['cost']:.2f} |")
    lines.append("")
    if d["docs_unused"]:
        lines += [f"The docs arm never searched the docs ({d['docs_runs']} runs), so these runs can't separate "
                  "CLI failures from docs failures.", ""]
    if d["disagree"]:
        lines += ["Arms disagreed on: " + ", ".join(d["disagree"]) + ".", ""]

    lines += ["## 7. Caveats", ""] + [f"- {c}" for c in d["caveats"]] + [""]
    return "\n".join(lines)


def build_report(run_dirs: list[Path], inv: Inventory, model=None) -> tuple[str, dict]:
    runs = [Run.load(p) for p in run_dirs]
    groups = friction_groups(runs)
    big = oversized(runs)
    findings = suggest_fixes(model, groups, big, inv) if model is not None else None
    data = compute(runs, inv, groups, big, findings)
    return render_md(data), data


def render_html(d: dict) -> str:
    """The interactive viewer: the template with the report data embedded (no network needed)."""
    payload = json.dumps(d).replace("</", "<\\/")  # can't close the script tag early
    return (Path(__file__).parent / "report.html").read_text().replace("/*__DATA__*/", payload)


def write_report(run_dirs: list[Path], inv: Inventory, out: Path, model=None) -> Path:
    """Write <out>.md, <out>.html and <out>.json; returns the HTML path."""
    md, data = build_report(run_dirs, inv, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md)
    out.with_suffix(".html").write_text(render_html(data))
    slim = {k: v for k, v in data.items() if k != "transcripts"}  # transcripts already live in workers/
    out.with_suffix(".json").write_text(json.dumps(slim, indent=2))
    return out.with_suffix(".html")
