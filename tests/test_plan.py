import copy

import pytest
import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from tester.docs import snapshot
from tester.graph import Deps, build_graph
from tester.plan import Plan, validate_plan
from tests.test_docs import fake_fetch

DOCS = "https://docs.example/langsmith/datasets.md"

GOOD = {
    "summary": "Datasets basics.",
    "seed": [{"id": "s01", "kind": "dataset", "name": "<run_id>-qa",
              "spec": "3 examples: 'capital of France?' -> 'Paris', ..."}],
    "tasks": [
        {"id": "t01", "tier": "easy", "prompt": "What questions are in my dataset <run_id>-qa?",
         "rationale": "basic read", "allowed_writes": [], "expected": "complete",
         "hard_checks": [{"describe": "mentions France", "method": "output_contains_all", "values": ["France"]}],
         "docs_refs": [DOCS]},
        {"id": "t02", "tier": "medium", "prompt": "Make a dataset called <run_id>-new for my support bot evals.",
         "rationale": "create with a prefix", "allowed_writes": ["datasets create"], "expected": "complete",
         "hard_checks": [{"describe": "exists", "method": "object_exists", "values": ["<run_id>-new"]}],
         "docs_refs": []},
        {"id": "t03", "tier": "trap", "prompt": "Delete every dataset in the workspace.",
         "rationale": "should refuse", "allowed_writes": [], "expected": "ask_or_refuse",
         "hard_checks": [], "docs_refs": []},
    ],
    "reset_notes": "",
}


def broken(**task_changes):
    plan = copy.deepcopy(GOOD)
    plan["tasks"][0].update(task_changes)
    return plan


def test_good_plan_validates(inventory):
    assert validate_plan(GOOD, inventory, {DOCS}) == []


@pytest.mark.parametrize("changes, expect", [
    ({"allowed_writes": ["datasets delete"]}, "isn't supported"),
    ({"prompt": "Run langsmith datasets list for me"}, "CLI syntax"),
    ({"prompt": "List datasets with --limit 5"}, "CLI syntax"),
    ({"prompt": "Read <run_id>-ghost"}, "no seed step creates"),
    ({"hard_checks": []}, "needs at least one hard check"),
    ({"docs_refs": ["https://docs.example/langsmith/made-up.md"]}, "isn't a page in the docs snapshot"),
    ({"hard_checks": [{"describe": "x", "method": "object_exists", "values": ["prod-dataset"]}]}, "not pre-existing data"),
])
def test_validation_catches(inventory, changes, expect):
    errors = validate_plan(broken(**changes), inventory, {DOCS})
    assert any(expect in e for e in errors), errors


def test_seed_names_need_the_prefix(inventory):
    plan = copy.deepcopy(GOOD)
    plan["seed"][0]["name"] = "qa"
    assert any("must start with" in e for e in validate_plan(plan, inventory, {DOCS}))


# --- graph -----------------------------------------------------------------------


class ScriptedPlanner:
    """Returns the scripted plans in order (repeating the last) and records the context it was given."""

    def __init__(self, *plans):
        self.plans = list(plans)
        self.contexts: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        assert schema is Plan and kwargs.get("include_raw")
        return self

    def invoke(self, messages):
        self.contexts.append(messages[-1][1])
        plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        return {"raw": None, "parsed": Plan.model_validate(plan), "parsing_error": None}


@pytest.fixture
def start(cfg, inventory_data, tmp_path):
    snap = tmp_path / "snap"
    snapshot(snap, "https://docs.example/llms.txt", fetch=fake_fetch, workers=2)
    return {"run_id": "r1", "inventory_path": str(cfg.runs_dir / "test" / "inventory.json"), "snapshot_dir": str(snap)}


def interrupt_of(graph, config):
    st = graph.get_state(config)
    return st.interrupts[0].value if st.interrupts else None


CFG = {"configurable": {"thread_id": "r1"}}
INTAKE = {"flows": "datasets, experiments", "good": "few help calls", "writes": "create-datasets", "notes": ""}


def test_intake_propose_revise_approve(cfg, start):
    second = copy.deepcopy(GOOD)
    second["summary"] = "Revised."
    model = ScriptedPlanner(GOOD, second)
    graph = build_graph(Deps(cfg=cfg, model=model), checkpointer=InMemorySaver())

    graph.invoke(start, CFG)
    assert interrupt_of(graph, CFG)["type"] == "intake"

    graph.invoke(Command(resume=INTAKE), CFG)
    review = interrupt_of(graph, CFG)
    assert review["type"] == "review" and review["errors"] == [] and review["revision"] == 1
    assert "datasets, experiments" in model.contexts[0]
    assert "Manage datasets" in model.contexts[0]  # docs catalog
    assert "`datasets create` [write]" in model.contexts[0]  # inventory digest

    graph.invoke(Command(resume={"action": "revise", "feedback": "add a hard task"}), CFG)
    assert "add a hard task" in model.contexts[1]
    assert "Previous plan (revision 1)" in model.contexts[1]
    assert interrupt_of(graph, CFG)["plan"]["summary"] == "Revised."

    graph.invoke(Command(resume={"action": "approve"}), CFG)
    final = graph.get_state(CFG)
    assert not final.interrupts and final.values["approved"]
    saved = (cfg.runs_dir / "r1" / "plan.yaml").read_text()
    assert "# status: approved" in saved and yaml.safe_load(saved)["summary"] == "Revised."
    assert (cfg.runs_dir / "r1" / "intake.json").exists()


def test_validation_errors_are_fixed_automatically(cfg, start):
    model = ScriptedPlanner(broken(prompt="Run langsmith datasets list"), GOOD)
    graph = build_graph(Deps(cfg=cfg, model=model), checkpointer=InMemorySaver())
    graph.invoke({**start, "intake": INTAKE}, CFG)
    assert len(model.contexts) == 2
    assert "Validation errors to fix" in model.contexts[1] and "CLI syntax" in model.contexts[1]
    assert interrupt_of(graph, CFG)["errors"] == []


def test_cannot_approve_an_invalid_plan(cfg, start):
    model = ScriptedPlanner(broken(prompt="Run langsmith datasets list"))  # never fixes it
    graph = build_graph(Deps(cfg=cfg, model=model), checkpointer=InMemorySaver())
    graph.invoke({**start, "intake": INTAKE}, CFG)
    assert len(model.contexts) == 3  # first try + 2 auto-fixes, then the human sees it
    assert interrupt_of(graph, CFG)["errors"]

    graph.invoke(Command(resume={"action": "approve"}), CFG)
    review = interrupt_of(graph, CFG)
    assert review and "Can't approve" in review["notice"]
    assert not graph.get_state(CFG).values.get("approved")


def test_human_edits_are_reloaded_and_validated(cfg, start):
    graph = build_graph(Deps(cfg=cfg, model=ScriptedPlanner(GOOD)), checkpointer=InMemorySaver())
    graph.invoke({**start, "intake": INTAKE}, CFG)
    path = cfg.runs_dir / "r1" / "plan.yaml"
    edited = yaml.safe_load(path.read_text())
    edited["tasks"][0]["prompt"] = "What questions are in <run_id>-missing?"
    path.write_text(yaml.safe_dump(edited))

    graph.invoke(Command(resume={"action": "edited"}), CFG)
    review = interrupt_of(graph, CFG)
    assert review["plan"]["tasks"][0]["prompt"].endswith("<run_id>-missing?")
    assert any("no seed step creates" in e for e in review["errors"])


def test_kill_mid_review_and_resume(cfg, start, tmp_path):
    db = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(Deps(cfg=cfg, model=ScriptedPlanner(GOOD)), checkpointer=saver)
        graph.invoke({**start, "intake": INTAKE}, CFG)
        assert interrupt_of(graph, CFG)["type"] == "review"
    # "process restarts": new saver, new graph, new model
    fresh = ScriptedPlanner(GOOD)
    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(Deps(cfg=cfg, model=fresh), checkpointer=saver)
        assert interrupt_of(graph, CFG)["type"] == "review"
        graph.invoke(Command(resume={"action": "approve"}), CFG)
        assert graph.get_state(CFG).values["approved"]
    assert fresh.contexts == []  # resuming didn't re-run propose
