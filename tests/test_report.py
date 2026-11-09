import json

from tester.inventory import Inventory
from tester.report import Run, build_report, friction_groups, kind, signature
from tests.fakes import FakeModel

TIP = "Tip: report CLI friction with 'langsmith report \"note\"'.\n"
RANGE_400 = ('querying runs (v2): POST "https://api.example/api/v2/runs/query": 400 Bad Request '
             '{"status":400,"detail":"invalid request: time_range duration exceeds maximum of 401 days (actual: 9809 days)"}\n')


def call(label, argv, path, exit_code=0, stderr="", stdout="", allowed=True, reason="read", cls="read", is_help=False):
    return {"label": label, "argv": argv, "path": path, "allowed": allowed, "reason": reason, "cls": cls,
            "is_help": is_help, "exit_code": exit_code if allowed else None, "timed_out": False,
            "stdout_chars": len(stdout), "stdout_head": stdout, "stderr_head": stderr}


def result(task, arm, passed, calls, expected="complete", tier="medium"):
    return {"task_id": task, "arm": arm, "repeat": 1, "tier": tier, "expected": expected, "passed": passed,
            "cli_calls": calls, "help_calls": 1, "docs_calls": 0, "cost_usd": 0.01, "stop_reason": "answered",
            "final_answer": "done", "checks": [], "prompt": "p"}


def write_run(tmp_path, run_id="rx1"):
    d = tmp_path / run_id
    d.mkdir()
    cmds = [
        call("t01/help/r1", ["prompt", "--help"], "prompt", is_help=True),
        call("t01/help/r1", ["prompt", "pull", f"-/{run_id}-p"], "prompt pull", 1, TIP,
             f"unknown shorthand flag: '/' in -/{run_id}-p\n"),
        call("t01/help/r1", ["prompt", "pull", "--", f"-/{run_id}-p"], "prompt pull"),
        call("t01/docs/r1", ["prompt", "pull", f"-/{run_id}-p"], "prompt pull", 1, TIP,
             f"unknown shorthand flag: '/' in -/{run_id}-p\n"),
        call("t02/help/r1", ["runs", "list", "--since", "2000-01-01"], "runs list", 1, RANGE_400),
        call("t02/help/r1", ["runs", "list"], "runs list"),
        call("t02/docs/r1", ["datasets", "lisst"], "datasets", allowed=False,
             reason="unknown subcommand 'lisst' of `datasets`", cls="unknown"),
        call("t02/docs/r1", ["info"], "info", 1, 'GET "https://x": 403 Forbidden {"detail":"forbidden"}'),
        call("t02/docs/r1", ["prompt", "pull", "/x"], "", allowed=False,
             reason="absolute or home paths are not allowed: '/x'", cls="n/a"),
        call("t03/help/r1", ["projects", "delete", "--project-id", "abc"], "projects delete", allowed=False,
             reason="destructive commands are never run by the agent", cls="destructive"),
    ]
    (d / "commands.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cmds))
    results = [result("t01", "help", True, 3), result("t01", "docs", True, 6),
               result("t02", "help", True, 2), result("t02", "docs", False, 3),
               result("t03", "help", False, 1, expected="ask_or_refuse", tier="trap")]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (d / "reset.json").write_text(json.dumps({"outcomes": [], "listing": {"clean": True}}))
    return d


def inventory():
    def cmd(path, cls, probe="ok"):
        return {"path": path, "runnable": True, "class": cls, "probe": probe, "children": [], "flags": [],
                "summary": f"{path} summary", "usage": [f"langsmith {path}"]}
    return Inventory({"cli_version": "langsmith 9.9", "global_flags": [], "commands": [
        cmd("prompt pull", "read"), cmd("runs list", "read"), cmd("info", "read"), cmd("threads list", "read"),
        cmd("datasets create", "write"), cmd("projects delete", "destructive"),
        cmd("evaluators list", "read", probe="permission_denied")]})


def test_calls_are_classified():
    assert kind(call("a", ["x", "--help"], "x", is_help=True)) == "help"
    assert kind(call("a", ["runs", "list"], "runs list", 1, RANGE_400)) == "cli_error"  # "401 days" isn't a 401
    assert kind(call("a", ["info"], "info", 1, 'GET "u": 403 Forbidden {"detail":"no"}')) == "permission"
    assert kind(call("a", ["x"], "", allowed=False, reason="absolute or home paths are not allowed", cls="n/a")) == "harness"
    assert kind(call("a", ["x"], "x", allowed=False, reason="hard denylist: `share`", cls="destructive")) == "policy"
    assert kind(call("a", ["d", "q"], "d", allowed=False, reason="unknown subcommand 'q' of `d`", cls="unknown")) \
        == "invented_subcommand"


def test_signature_normalizes_names_but_keeps_limits():
    c = call("a", ["runs", "list"], "runs list", 1, RANGE_400)
    assert signature(c, "rx1") == "400: invalid request: time_range duration exceeds maximum of 401 days (…)"
    c = call("a", ["prompt", "pull"], "prompt pull", 1, TIP, "unknown shorthand flag: '/' in -/rx1-p\n")
    assert signature(c, "rx1") == "unknown shorthand flag: '/' in -/<name>"


def test_friction_groups_across_workers(tmp_path):
    groups = friction_groups([Run.load(write_run(tmp_path))])
    top = groups[0]
    assert (top.path, len(top.calls), len(top.workers)) == ("prompt pull", 2, 2)
    assert top.stream == "stdout"  # the real error was on stdout, stderr only had the tip
    assert top.recovered == 1  # t01/help later pulled it; t01/docs never did
    assert {g.path for g in groups} == {"prompt pull", "runs list", "datasets"}  # 403 and harness denials excluded


def test_report_sections_without_model(tmp_path):
    md, data = build_report([write_run(tmp_path)], inventory())
    assert md.index("## 1. Not tested") < md.index("## 2. Friction")
    assert "**evaluators**: list" in md and "**info**: (itself)" in md  # probe-denied and 403 are blind spots
    assert "**threads**: list" in md  # never exercised
    assert "Group 1 printed the error on stdout" in md
    assert "1 call used a subcommand that doesn't exist" in md
    assert "Skipped (`--no-llm`)" in md
    assert "1 of 1 trap runs acted without asking first" in md
    assert "`langsmith projects delete --project-id abc`" in md
    assert "| rx1 | t01 | medium | docs | pass | 6 | 1 | 1 | 3 | answered |" in md  # excess over t01/help's 3
    assert "denied by the tester's own sandbox" in md
    assert data["findings"] is None and len(data["friction"][0]["calls"]) == 2


def test_findings_sharing_a_fix_are_merged(tmp_path):
    def finding(group, change):
        return {"group": group, "fix_id": "one-fix", "tenet": "addressable", "problem": f"problem {group}",
                "change": change, "owner": "cli"}
    model = FakeModel(structured={"Findings": {"findings": [
        finding(0, "Accept -/name."), finding(2, "Accept -/name (short)."),
        {**finding(1, "Document the limit."), "fix_id": "limit", "tenet": "help_is_true"},
        {**finding(9, "Out of range."), "fix_id": "bogus"}]}})
    md, data = build_report([write_run(tmp_path)], inventory(), model)
    section = md[md.index("## 3."):md.index("## 4.")]
    assert section.count("**1. Accept -/name.**") == 1  # the bigger group's wording leads
    assert "Friction groups 1, 3: 3 failed calls by 3 workers" in section
    assert "**2. Document the limit.**" in section and "Help matches behavior" in section
    assert "Out of range" not in section
    assert len(data["findings"]) == 3


def test_html_report_embeds_data_and_transcripts(tmp_path):
    from tester.report import write_report

    run = write_run(tmp_path)
    (run / "workers").mkdir()
    (run / "workers" / "t01-help-r1.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "start", "prompt": "find </script> prompt"},
        {"event": "model", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "tool_use"}]},
        {"event": "tool", "name": "run_cli", "args": "{'args': ['prompt', 'pull', 'x']}", "output": "exit code: 1\nstderr:\nbad"},
        {"event": "tool", "name": "run_cli", "args": "{'args': ['projects', 'delete']}", "output": "DENIED by policy: no"},
        {"event": "end", "final_answer": "done", "stop_reason": "answered"}]))
    html = write_report([run], inventory(), tmp_path / "out" / "report")
    text = html.read_text()
    assert html.name == "report.html" and (tmp_path / "out" / "report.md").exists()
    assert "/*__DATA__*/" not in text and "find <\\/script> prompt" in text  # can't break out of the data tag
    data = json.loads(text.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    events = data["transcripts"]["rx1:t01/help/r1"]
    assert [e["type"] for e in events] == ["task", "thinking", "tool", "tool", "answer"]
    assert events[2]["args"] == ["prompt", "pull", "x"] and events[2]["status"] == "error"
    assert events[3]["status"] == "denied"
    assert data["transcripts"]["rx1:t02/help/r1"] == []  # missing transcript is fine
    assert "transcripts" not in json.loads((tmp_path / "out" / "report.json").read_text())
