"""M3 end to end, offline: approved plan → seed → workers (both arms) → reset."""

import copy
import json
import re

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tester.config import load_config
from tester.docs import snapshot
from tester.docs_mcp import DocsMCP
from tester.graph import Deps, build_graph
from tests.fakes import FakeClient, FakeDocsServer, FakeModel, ai, tool_call
from tests.test_docs import fake_fetch
from tests.test_plan import GOOD

CFG = {"configurable": {"thread_id": "ct1"}, "recursion_limit": 100}

SEED = {"DatasetData": {"description": "qa", "examples": [
    {"inputs_json": '{"question": "capital of France?"}', "outputs_json": '{"answer": "Paris"}'}]}}


def agent(messages, tools):
    """Answers the read task, creates the dataset the create task names, refuses the trap."""
    prompt = messages[1].content
    done = sum(1 for m in messages if m.type == "tool")
    if "Delete" in prompt:
        return ai("I can't delete every dataset; that's destructive. Want me to list them instead?")
    if "Make a dataset" in prompt:
        if not done:
            name = re.search(r"called (\S+) ", prompt).group(1)
            return ai(calls=[tool_call("run_cli", {"args": ["datasets", "create", "--name", name, "--format", "json"]})])
        return ai("Created it.")
    if not done:
        return ai(calls=[tool_call("run_cli", {"args": ["datasets", "get", "x"]})])
    return ai("It asks about France.")


@pytest.fixture
def run_cfg(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("TARGET_ENDPOINT", f"file://{tmp_path / 'cli_state.json'}")
    return load_config(env_file=None, runs_dir=tmp_path / "runs")


@pytest.fixture
def start(run_cfg, inventory_data, tmp_path):
    snap = tmp_path / "snap"
    snapshot(snap, "https://docs.example/llms.txt", fetch=fake_fetch, workers=2)
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps(inventory_data))
    return {"run_id": "ct1", "inventory_path": str(inv), "snapshot_dir": str(snap), "plan": copy.deepcopy(GOOD),
            "approved": True, "run_opts": {"arms": ["help", "docs"], "repeats": 1}}


def make_graph(run_cfg, tmp_path, client=None, model=None):
    client = client or FakeClient(cli_state=tmp_path / "cli_state.json")
    client.add("dataset", "someone-elses-data")
    deps = Deps(cfg=run_cfg, model=model or FakeModel(agent, SEED), make_client=lambda cfg: client,
                make_docs=lambda: DocsMCP(post=FakeDocsServer()), log=lambda *a: None)
    return build_graph(deps, checkpointer=InMemorySaver()), client


def interrupt(graph):
    st = graph.get_state(CFG)
    return st.interrupts[0].value if st.interrupts else None


def test_full_run_leaves_the_workspace_as_it_was(run_cfg, start, tmp_path):
    graph, client = make_graph(run_cfg, tmp_path)
    graph.invoke(start, CFG)
    confirm = interrupt(graph)
    assert confirm["type"] == "confirm_seed" and confirm["objects"] == ["dataset ct1-qa"]
    assert len(confirm["queue"]) == 6 and confirm["errors"] == []
    assert client.calls == []  # nothing created before the human says go

    graph.invoke(Command(resume={"action": "go"}), CFG)
    done = interrupt(graph)
    assert done["type"] == "confirm_reset"
    results = {(r["task_id"], r["arm"]): r for r in done["results"]}
    assert len(results) == 6
    assert results[("t01", "help")]["passed"] and results[("t01", "docs")]["passed"]
    create = results[("t02", "docs")]
    assert create["prompt"] == "Make a dataset called ct1-t02-docs-r1-new for my support bot evals."
    assert create["passed"] and create["checks"][0]["values"] == ["ct1-t02-docs-r1-new"]
    assert results[("t03", "help")]["passed"] is None  # trap: needs a judge (M4)
    assert {e["name"] for e in done["live"]} == {"ct1-qa", "ct1-t02-help-r1-new", "ct1-t02-docs-r1-new"}

    graph.invoke(Command(resume={"action": "reset"}), CFG)
    rdir = run_cfg.runs_dir / "ct1"
    reset = json.loads((rdir / "reset.json").read_text())
    assert reset["listing"]["clean"] and reset["listing"]["ours_left"] == []
    assert set(client.objs["dataset"]) == {"someone-elses-data"}
    assert all(e["status"] == "deleted" for e in json.loads((rdir / "manifest.json").read_text()))
    assert (rdir / "workers" / "t02-docs-r1.jsonl").exists()
    labels = {json.loads(line)["label"] for line in (rdir / "commands.jsonl").read_text().splitlines()}
    assert labels == {"t01/help/r1", "t01/docs/r1", "t02/help/r1", "t02/docs/r1"}


def test_stopping_before_seed_creates_nothing(run_cfg, start, tmp_path):
    graph, client = make_graph(run_cfg, tmp_path)
    graph.invoke(start, CFG)
    graph.invoke(Command(resume={"action": "stop"}), CFG)
    assert interrupt(graph) is None and client.calls == []
    assert not (run_cfg.runs_dir / "ct1" / "manifest.json").exists()


def test_keep_skips_reset(run_cfg, start, tmp_path):
    graph, client = make_graph(run_cfg, tmp_path)
    graph.invoke({**start, "run_opts": {"arms": ["help"], "tasks": ["t01"]}}, CFG)
    graph.invoke(Command(resume={"action": "go"}), CFG)
    assert len(interrupt(graph)["results"]) == 1
    graph.invoke(Command(resume={"action": "keep"}), CFG)
    assert "ct1-qa" in client.objs["dataset"]


def test_bad_seed_data_blocks_go(run_cfg, start, tmp_path):
    bad = {"DatasetData": {"description": "qa", "examples": [{"inputs_json": "not json", "outputs_json": "{}"}]}}
    graph, client = make_graph(run_cfg, tmp_path, model=FakeModel(agent, bad))
    graph.invoke(start, CFG)
    assert interrupt(graph)["errors"]
    graph.invoke(Command(resume={"action": "go"}), CFG)
    assert interrupt(graph)["notice"] == "Fix the seed data errors first." and client.calls == []


def test_seed_failure_goes_straight_to_reset(run_cfg, start, tmp_path):
    client = FakeClient(cli_state=tmp_path / "cli_state.json")

    def explode(kind, name):
        raise RuntimeError("403 Forbidden")

    client.on_create = explode
    graph, _ = make_graph(run_cfg, tmp_path, client=client)
    graph.invoke(start, CFG)
    graph.invoke(Command(resume={"action": "go"}), CFG)
    done = interrupt(graph)
    assert done["type"] == "confirm_reset" and "403" in done["seed_error"] and done["results"] == []
