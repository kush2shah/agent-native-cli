import json

import pytest

from tester.budget import CostMeter
from tester.docs_mcp import DocsMCP
from tester.run_cli import CliRunner
from tester.worker import MAX_TOOL_OUTPUT, WorkerEnv, asks_for_json, cap, run_worker, system_prompt
from tests.fakes import FakeDocsServer, FakeModel, ai, tool_call, tool_outputs


def scripted(*turns):
    """Agent that plays `turns` in order; each turn is a list of tool calls or a final answer string."""
    seen = []

    def agent(messages, tools):
        seen.append(messages)
        turn = turns[len(seen) - 1]
        if isinstance(turn, str):
            return ai(turn)
        return ai(calls=[tool_call(n, a, i) for i, (n, a) in enumerate(turn)])

    agent.seen = seen
    return agent


@pytest.fixture
def env_for(cfg, inventory, tmp_path):
    runners = []

    def make(model, docs=None, ceiling=5.0, max_calls=15):
        r = CliRunner(cfg, inventory=inventory, max_calls=max_calls, log_path=tmp_path / "commands.jsonl", label="t01")
        runners.append(r)
        return WorkerEnv(runner=r, model=model, meter=CostMeter(ceiling, "anthropic:claude-sonnet-5-5"),
                         transcript=tmp_path / "t01.jsonl", docs=docs)

    yield make
    for r in runners:
        r.close()


def test_worker_runs_cli_and_answers(env_for, tmp_path):
    agent = scripted(
        [("run_cli", {"args": ["datasets", "--help"]})],
        [("run_cli", {"args": ["datasets", "list", "--format", "json"]}),
         ("run_cli", {"args": ["datasets", "delete", "x"]})],
        "You have one dataset: fixture-dataset.",
    )
    model = FakeModel(agent)
    env = env_for(model)
    out = run_worker(env, "sys", "What datasets do I have?")

    assert out["final_answer"] == "You have one dataset: fixture-dataset."
    assert out["stop_reason"] == "answered" and out["turns"] == 3
    assert (out["cli_calls"], out["help_calls"], out["used_json_format"]) == (3, 1, True)
    assert out["denied"] == [{"argv": ["datasets", "delete", "x"],
                              "reason": "destructive commands are never run by the agent"}]
    assert [t["function"]["name"] for t in model.tools] == ["run_cli"]  # help arm: no docs tools
    outputs = tool_outputs(agent.seen[2])
    assert "fixture-dataset" in outputs[1] and outputs[2].startswith("DENIED by policy")
    events = [json.loads(line)["event"] for line in (tmp_path / "t01.jsonl").read_text().splitlines()]
    assert events == ["start", "model", "tool", "model", "tool", "tool", "model", "end"]
    assert out["cost_usd"] == pytest.approx(3 * (100 * 2 + 10 * 10) / 1e6)


def test_docs_arm_gets_only_the_read_tools(env_for):
    server = FakeDocsServer()
    agent = scripted([("search_docs_by_lang_chain", {"query": "list datasets"}),
                      ("submit_feedback", {"query": "your docs are wrong"})], "done")
    model = FakeModel(agent)
    out = run_worker(env_for(model, docs=DocsMCP(post=server)), "sys", "go")
    assert [t["function"]["name"] for t in model.tools] == [
        "run_cli", "search_docs_by_lang_chain", "query_docs_filesystem_docs_by_lang_chain"]
    outputs = tool_outputs(agent.seen[1])
    assert outputs[0] == "docs about {'query': 'list datasets'}"
    assert outputs[1] == "ERROR: unknown tool 'submit_feedback'"
    assert out["docs_calls"] == 1
    assert [c["method"] for c in server.calls] == ["initialize", "tools/list", "tools/call"]


def test_docs_client_refuses_feedback_even_if_called_directly():
    with pytest.raises(PermissionError):
        DocsMCP(post=FakeDocsServer()).call("submit_feedback", {})


def test_cli_budget_is_enforced(env_for):
    agent = scripted(*[[("run_cli", {"args": ["datasets", "list"]})]] * 3, "gave up")
    out = run_worker(env_for(FakeModel(agent), max_calls=2), "sys", "go")
    assert out["cli_calls"] == 3
    assert out["denied"][0]["reason"] == "budget: max 2 CLI calls reached"


def test_cost_ceiling_stops_the_worker(env_for):
    agent = scripted(*[[("run_cli", {"args": ["datasets", "list"]})]] * 10)
    out = run_worker(env_for(FakeModel(agent), ceiling=0.0005), "sys", "go")
    assert out["stop_reason"] == "cost_ceiling" and out["turns"] == 2


def test_args_as_a_string_is_an_error_not_a_shell(env_for):
    agent = scripted([("run_cli", {"args": "datasets list | head"})], "ok")
    run_worker(env_for(FakeModel(agent)), "sys", "go")
    assert tool_outputs(agent.seen[1]) == ["ERROR: `args` must be a list of strings, one per argument"]


def test_helpers():
    assert asks_for_json(["x", "--format", "json"]) and asks_for_json(["--format=jsonl"])
    assert not asks_for_json(["--format", "pretty"]) and not asks_for_json(["--format"])
    assert cap("a" * (MAX_TOOL_OUTPUT + 5)).endswith(f"[output truncated: {MAX_TOOL_OUTPUT + 5:,} characters in total]")
    prompt = system_prompt("Usage: langsmith", "docs", "Use staging.")
    assert "docs tools" in prompt and "Usage: langsmith" in prompt and "Use staging." in prompt
    assert "docs tools" not in system_prompt("Usage: langsmith", "help")
