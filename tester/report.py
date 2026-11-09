"""Turn run folders into report.md: what wasn't tested, where agents wasted
calls, how they behaved on risky requests, and how the arms compared.

Everything except the suggested fixes is computed from the run files
(results.jsonl, commands.jsonl, reset.json) and the inventory. The fixes
come from one structured model call over the grouped evidence, and can be
skipped with --no-llm.
"""

from __future__ import annotations

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


# --- rendering --------------------------------------------------------------


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


def render(runs: list[Run], inv: Inventory, groups: list[Group], big: list[dict],
           findings: list[Finding] | None) -> str:
    results = [(run, r) for run in runs for r in run.results]
    workers = len(results)
    cost = sum(r.get("cost_usd", 0) for _, r in results)
    all_calls = [c for run in runs for c in run.commands]
    counts = defaultdict(int)
    for c in all_calls:
        counts[kind(c)] += 1
    failed = counts["cli_error"] + counts["invented_subcommand"]
    work_calls = sum(counts[k] for k in ("ok", "cli_error", "invented_subcommand", "permission"))
    nt = not_tested(inv, runs)
    lines = [
        f"# CLI agent-usability report: {', '.join(r.run_id for r in runs)}",
        "",
        f"{inv.data.get('cli_version', 'unknown CLI version')}. {workers} worker runs, {len(all_calls)} CLI calls "
        f"({counts['help']} help lookups), {failed} failed, ${cost:.2f} in worker model cost.",
        "",
    ]

    # 1. Not tested
    lines += ["## 1. Not tested", ""]
    if nt["blind_spots"]:
        lines += ["**Blind spots** (the key couldn't run these, so nothing here says whether they work):",
                  _by_group(nt["blind_spots"]), ""]
    else:
        lines += ["**Blind spots:** none. Every command the workers tried was permitted for the key.", ""]
    lines += [
        f"**Coverage:** {nt['reads_total'] - len(nt['reads_untested'])} of {nt['reads_total']} read commands and "
        f"{nt['writes_total'] - len(nt['writes_untested'])} of {nt['writes_total']} write commands were exercised. "
        f"The {nt['destructive_total']} destructive commands are never run by workers.",
        "",
        f"<details><summary>{len(nt['reads_untested'])} read commands never exercised</summary>",
        "", _by_group(nt["reads_untested"]), "", "</details>", "",
    ]

    # 2. Friction
    lines += ["## 2. Friction: where agents wasted calls", ""]
    if groups:
        lines += [
            f"{failed} of {work_calls} non-help calls failed ({failed / max(work_calls, 1):.0%}). "
            "Grouped by command and error, with names and numbers normalized:", "",
            "| # | Failed calls | Workers | Same command later worked | Command | Error |",
            "|---|---|---|---|---|---|",
        ]
        for i, g in enumerate(groups):
            lines.append(f"| {i + 1} | {len(g.calls)} | {len(g.workers)} | {g.recovered}/{len(g.calls)} | "
                         f"`{g.path}` | {_cell(g.signature)} |")
        lines.append("")
        invented = [g for g in groups if INVENTED_FLAG.search(g.signature)]
        on_stdout = [g for g in groups if g.stream == "stdout"]
        notes = []
        if counts["invented_subcommand"]:
            notes.append(f"{counts['invented_subcommand']} {_plural(counts['invented_subcommand'], 'call')} used a subcommand that doesn't exist "
                         "(the real CLI prints help and exits 0 for these, so the agent can't tell).")
        if invented:
            n = sum(len(g.calls) for g in invented)
            notes.append(f"{n} {_plural(n, 'call')} used a flag the CLI rejected.")
        if on_stdout:
            which = ", ".join(str(groups.index(g) + 1) for g in on_stdout)
            notes.append(f"{'Groups' if len(on_stdout) > 1 else 'Group'} {which} printed the error on stdout; "
                         "stderr held only a \"report CLI friction\" tip.")
        lines += [f"- {n}" for n in notes] + ([""] if notes else [])
    else:
        lines += ["No failed calls.", ""]
    if big:
        lines += [f"**Oversized output:** {len(big)} successful calls returned more than the agent's "
                  f"{MAX_TOOL_OUTPUT:,}-character window, so it saw a truncated result:", ""]
        lines += [f"- {_cmd(o['argv'])}: {o['chars']:,} chars" for o in big[:5]] + [""]

    # 3. Suggested changes
    lines += ["## 3. Suggested changes", ""]
    if findings is None:
        lines += ["Skipped (`--no-llm`).", ""]
    elif not findings:
        lines += ["Nothing to suggest.", ""]
    else:
        merged: dict[str, list[Finding]] = defaultdict(list)  # one entry per fix, however many groups it covers
        for f in findings:
            merged[f.fix_id.strip().lower()].append(f)
        ranked = sorted(merged.items(), key=lambda kv: -sum(len(groups[f.group].calls) for f in kv[1]))
        for n, (_, fs) in enumerate(ranked, 1):
            fs.sort(key=lambda f: -len(groups[f.group].calls))
            change = fs[0].change
            calls = sum(len(groups[f.group].calls) for f in fs)
            workers = len(set().union(*(groups[f.group].workers for f in fs)))
            tenet = TENETS.get(fs[0].tenet, fs[0].tenet).split(":")[0]
            nums = ", ".join(str(f.group + 1) for f in fs)
            lines += [f"**{n}. {change}**", "",
                      f"Owner: {fs[0].owner}. Tenet: {tenet}. Friction {'groups' if len(fs) > 1 else 'group'} "
                      f"{nums}: {calls} failed {_plural(calls, 'call')} by {workers} {_plural(workers, 'worker')}.", ""]
            for f in fs:
                g = groups[f.group]
                ex = g.examples(1)[0]
                lines += [f"- {f.problem}", f"  - e.g. {_cmd(ex['argv'])} → {_cell(g.signature)}"]
            lines.append("")

    # 4. Behavior on risky requests
    traps = [(run, r) for run, r in results if r.get("expected") == "ask_or_refuse"]
    lines += ["## 4. Behavior on risky requests", ""]
    if traps:
        failed_traps = [x for x in traps if not x[1].get("passed")]
        lines += [f"{len(failed_traps)} of {len(traps)} trap runs acted without asking first.", ""]
        for run, r in traps:
            verdict = "asked or refused" if r.get("passed") else "acted"
            tried = "; ".join(_cmd(a) for a in risky_attempts(run, r)) or "nothing risky"
            lines += [f"- **{run.run_id} {r['task_id']} / {r['arm']}: {verdict}.** Tried: {tried}.",
                      f"  > {_cell(r.get('final_answer', '')[:240])}…"]
        lines.append("")
    else:
        lines += ["No trap tasks in these runs.", ""]

    # 5. Scorecard
    lines += ["## 5. Scorecard", "",
              "Calls are CLI calls; **excess** is calls beyond the fewest any passing worker needed for that task "
              "(a weak baseline with one repeat per arm).", "",
              "| Run | Task | Tier | Arm | Verdict | Calls | Help | Failed | Excess | Stop |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        best = baselines(run.results)
        for r in sorted(run.results, key=lambda r: (r["task_id"], r["arm"], r["repeat"])):
            verdict = {True: "pass", False: "**FAIL**", None: "unscored"}[r.get("passed")]
            excess = r["cli_calls"] - best[r["task_id"]] if r["task_id"] in best and r.get("passed") else "–"
            lines.append(f"| {run.run_id} | {r['task_id']} | {r['tier']} | {r['arm']} | {verdict} | {r['cli_calls']} | "
                         f"{r['help_calls']} | {failed_calls(run, r)} | {excess} | {r['stop_reason']} |")
    lines.append("")
    for run, r in results:
        for ch in r.get("checks", []):
            if not ch.get("passed"):
                lines.append(f"- {run.run_id} {r['task_id']}/{r['arm']} failed check: {ch['describe']}")
    lines.append("")

    # 6. Arms
    lines += ["## 6. Arms", "", "| Arm | Passed | Mean calls | Mean help | Docs calls | Cost |", "|---|---|---|---|---|---|"]
    arms = defaultdict(list)
    for _, r in results:
        arms[r["arm"]].append(r)
    for arm, rs in sorted(arms.items()):
        passed = sum(1 for r in rs if r.get("passed"))
        lines.append(f"| {arm} | {passed}/{len(rs)} | {mean(r['cli_calls'] for r in rs):.1f} | "
                     f"{mean(r['help_calls'] for r in rs):.1f} | {sum(r.get('docs_calls', 0) for r in rs)} | "
                     f"${sum(r.get('cost_usd', 0) for r in rs):.2f} |")
    lines.append("")
    docs_runs = arms.get("docs", [])
    if docs_runs and not any(r.get("docs_calls") for r in docs_runs):
        lines += [f"The docs arm never searched the docs ({len(docs_runs)} runs), so these runs can't separate "
                  "CLI failures from docs failures.", ""]
    split = defaultdict(dict)
    for run, r in results:
        split[(run.run_id, r["task_id"])].setdefault(r["arm"], []).append(bool(r.get("passed")))
    disagree = [k for k, v in split.items() if len({all(x) for x in v.values()}) > 1]
    if disagree:
        lines += ["Arms disagreed on: " + ", ".join(f"{rid} {t}" for rid, t in disagree) + ".", ""]

    # 7. Caveats
    lines += ["## 7. Caveats", ""]
    caveats = []
    if counts["harness"]:
        harness = [c for c in all_calls if kind(c) == "harness"]
        reasons = sorted({signature(c, "") for c in harness})
        caveats.append(f"{counts['harness']} calls were denied by the tester's own sandbox, not the CLI: "
                       + "; ".join(reasons[:3]) + ". Check these aren't false positives.")
    if counts["budget"]:
        caveats.append(f"{counts['budget']} calls hit the per-task call budget.")
    stops = [f"{run.run_id} {r['task_id']}/{r['arm']} ({r['stop_reason']})" for run, r in results
             if r.get("stop_reason") != "answered"]
    if stops:
        caveats.append("Stopped before answering: " + ", ".join(stops) + ".")
    for run in runs:
        listing = (run.reset or {}).get("listing", {})
        if run.reset is None:
            caveats.append(f"{run.run_id} has no reset.json: the workspace may still hold its objects.")
        elif not listing.get("clean"):
            caveats.append(f"{run.run_id}'s reset didn't leave the workspace clean: {json.dumps(listing)}.")
    caveats.append("The plan generator and the worker are the same model; results that look clean may share its blind spots.")
    lines += [f"- {c}" for c in caveats] + [""]
    return "\n".join(lines)


def build_report(run_dirs: list[Path], inv: Inventory, model=None) -> tuple[str, dict]:
    runs = [Run.load(p) for p in run_dirs]
    groups = friction_groups(runs)
    big = oversized(runs)
    findings = suggest_fixes(model, groups, big, inv) if model is not None else None
    md = render(runs, inv, groups, big, findings)
    data = {
        "runs": [r.run_id for r in runs],
        "not_tested": not_tested(inv, runs),
        "friction": [{"path": g.path, "signature": g.signature, "calls": len(g.calls), "workers": len(g.workers),
                      "recovered": g.recovered, "stream": g.stream, "examples": g.examples()} for g in groups],
        "oversized": big,
        "findings": [f.model_dump() for f in findings] if findings is not None else None,
    }
    return md, data
