import json

import pytest

from tester.harness import HarnessError, HarnessExecutor, compare_listings
from tester.manifest import Manifest
from tests.fakes import FakeClient

DATASET = {"description": "qa", "examples": [{"inputs_json": '{"q": "capital of France?"}',
                                              "outputs_json": '{"a": "Paris"}'}]}
PROJECT = {"description": "bot", "runs": [
    {"name": "agent", "run_type": "chain", "inputs_json": '{"t": 1}', "outputs_json": "", "error": "Boom",
     "start_offset_s": 60, "latency_s": 30.0, "children": [
         {"name": "lookup", "run_type": "tool", "inputs_json": "{}", "outputs_json": "", "error": "Boom",
          "start_after_s": 1, "latency_s": 29}]},
    {"name": "agent", "run_type": "chain", "inputs_json": '{"t": 2}', "outputs_json": '{"ok": true}', "error": "",
     "start_offset_s": 300, "latency_s": 2.0, "children": []},
]}


@pytest.fixture
def ex(tmp_path):
    return HarnessExecutor(FakeClient(), "ct1", Manifest(tmp_path / "manifest.json"), sleep=lambda s: None)


def test_manifest_entry_is_written_before_the_create(ex):
    seen = []
    ex.client.on_create = lambda kind, name: seen.append(json.loads(ex.manifest.path.read_text()))
    ex.create_dataset("ct1-qa", DATASET)
    assert seen[0][0]["name"] == "ct1-qa" and seen[0][0]["status"] == "intended"
    [entry] = ex.manifest.entries
    assert entry["status"] == "created" and entry["id"] and entry["seeded"]
    [row] = ex.client.examples[entry["id"]]
    assert (row["inputs"], row["outputs"]) == ({"q": "capital of France?"}, {"a": "Paris"})
    assert entry["example_ids"] == [row["id"]]  # recorded, though not in seed order in general


def test_harness_only_creates_prefixed_names(ex):
    with pytest.raises(HarnessError, match="must start with 'ct1-'"):
        ex.create_dataset("prod-data", DATASET)
    assert ex.client.calls == [] and ex.manifest.entries == []


def test_seeding_is_idempotent_across_a_resume(ex):
    ex.create_dataset("ct1-qa", DATASET)
    ex.create_dataset("ct1-qa", DATASET)
    assert ex.client.calls == ["create_dataset ct1-qa"]
    assert len(ex.manifest.entries) == 1


def test_project_runs_are_posted_with_trees_and_waited_for(tmp_path):
    client = FakeClient(runs_visible_after=2)
    ex = HarnessExecutor(client, "ct1", Manifest(tmp_path / "m.json"), sleep=lambda s: None)
    ex.create_project_with_runs("ct1-bot", PROJECT)
    assert len(client.runs) == 3 and client._list_runs_calls == 3
    root, child = client.runs[0], client.runs[1]
    assert child["parent_run_id"] == root["id"] == child["trace_id"]
    assert child["dotted_order"].startswith(root["dotted_order"] + ".")
    assert (root["end_time"] - root["start_time"]).total_seconds() == 30
    assert root["error"] == "Boom" and "outputs" not in root and "extra" not in root and "tags" not in root


def test_runs_that_never_show_up_fail_loudly(tmp_path):
    ex = HarnessExecutor(FakeClient(runs_visible_after=10**6), "ct1", Manifest(tmp_path / "m.json"),
                         sleep=lambda s: None)
    with pytest.raises(HarnessError, match="became visible"):
        ex.create_project_with_runs("ct1-bot", PROJECT, timeout_s=0.01)


def test_reset_ignores_objects_not_in_the_manifest(ex):
    ex.client.add("dataset", "someone-elses-data")
    ex.client.add("dataset", "ct1-lookalike")  # has our prefix but we never created it
    ex.create_dataset("ct1-qa", DATASET)
    ex.create_prompt("ct1-prompt", {"description": "", "messages": [{"role": "system", "content": "hi"}]})
    outcomes = ex.reset()
    assert [o["name"] for o in outcomes] == ["ct1-prompt", "ct1-qa"]  # reverse creation order
    assert set(ex.client.objs["dataset"]) == {"someone-elses-data", "ct1-lookalike"}
    assert all(e["status"] == "deleted" for e in ex.manifest.entries)
    assert not any("lookalike" in c or "someone" in c for c in ex.client.calls)


def test_reset_resolves_an_intended_entry_by_its_exact_name(ex):
    ex.client.add("dataset", "ct1-t02-help-r1-new")  # created by a worker; the CLI printed no ID
    ex.manifest.intend("dataset", "ct1-t02-help-r1-new", "t02/help/r1")
    ex.manifest.intend("dataset", "ct1-t02-help-r2-new", "t02/help/r2")  # create failed; never existed
    outcomes = {o["name"]: o["outcome"] for o in ex.reset()}
    assert outcomes == {"ct1-t02-help-r1-new": "deleted", "ct1-t02-help-r2-new": "never created"}
    assert "ct1-t02-help-r1-new" not in ex.client.objs["dataset"]


def test_delete_refuses_entries_outside_the_manifest(ex):
    obj = ex.client.add("dataset", "ct1-x")
    with pytest.raises(HarnessError, match="isn't in the manifest"):
        ex.delete({"kind": "dataset", "name": "ct1-x", "id": obj.id, "status": "created"})
    assert "ct1-x" in ex.client.objs["dataset"]


def test_listing_comparison_splits_ours_from_others():
    before = {"dataset": {"1": "theirs"}, "project": {}, "prompt": {"9": "p"}}
    after = {"dataset": {"1": "theirs", "2": "ct1-left", "3": "new-theirs"}, "project": {}, "prompt": {}}
    cmp = compare_listings(before, after, "ct1-")
    assert cmp == {"ours_left": ["dataset ct1-left"], "others_added": ["dataset new-theirs"],
                   "removed": ["prompt p"], "clean": False}


AGENT = {"display_name": "Checkout", "environments": [
    {"environment": "staging", "runs": [
        {"name": "checkout", "run_type": "chain", "inputs_json": "{}", "outputs_json": "", "error": "CardDeclined",
         "start_offset_s": 100, "latency_s": 1.0, "children": [], "metadata_json": '{"release": "v2"}',
         "tags": ["canary"], "feedback": [{"key": "user_rating", "score": 0, "comment": ""}],
         "repeat": 3, "repeat_spacing_s": 3600}]},
    {"environment": "production", "runs": []},
]}


def test_agent_runs_go_to_their_environment_with_repeats_metadata_and_feedback(ex):
    ex.create_agent_with_runs("ct1-checkout", AGENT)
    runs = ex.client.runs
    assert len(runs) == 3 and all(r["address"] == {"kind": "AGENT", "id": "ct1-checkout", "environment": "STAGING"}
                                  for r in runs)
    gaps = sorted((runs[0]["start_time"] - r["start_time"]).total_seconds() for r in runs)
    assert gaps == [0, 3600, 7200]
    assert runs[0]["extra"] == {"metadata": {"release": "v2"}} and runs[0]["tags"] == ["canary"]
    assert [f["key"] for f in ex.client.feedback] == ["user_rating"] * 3
    assert ex.manifest.find("agent", "ct1-checkout")["seeded"]
    assert ex.reset() == [{"kind": "agent", "name": "ct1-checkout", "outcome": "deleted"}]
    assert ex.client.agents.items == {}


def test_experiment_links_runs_to_examples_and_scores_them(ex):
    ds = {"description": "", "examples": [
        {"inputs_json": '{"q": "a"}', "outputs_json": '{"a": 1}'},
        {"inputs_json": '{"q": "b"}', "outputs_json": '{"a": 2}'}]}
    ex.create_dataset("ct1-qa", ds)
    exp = {"description": "baseline", "dataset": "<run_id>-qa", "rows": [
        {"example_index": 1, "outputs_json": '{"a": 2}', "error": "", "latency_s": 1.5,
         "scores": [{"key": "correctness", "score": 1, "comment": ""}]},
        {"example_index": 0, "outputs_json": '{"a": 9}', "error": "", "latency_s": 2.0,
         "scores": [{"key": "correctness", "score": 0, "comment": "wrong"}]}]}
    ex.create_experiment("ct1-baseline", exp, "ct1-qa", [{"q": "a"}, {"q": "b"}])
    by_q = {e["inputs"]["q"]: e["id"] for e in ex.client.examples[ex.manifest.find("dataset", "ct1-qa")["id"]]}
    assert [r["reference_example_id"] for r in ex.client.runs] == [by_q["b"], by_q["a"]]
    assert [r["inputs"] for r in ex.client.runs] == [{"q": "b"}, {"q": "a"}]
    assert ex.client.objs["project"]["ct1-baseline"].reference_dataset_id == ex.manifest.find("dataset", "ct1-qa")["id"]
    assert [f["score"] for f in ex.client.feedback] == [1, 0]
    outcomes = ex.reset()
    assert [o["name"] for o in outcomes] == ["ct1-baseline", "ct1-qa"] and all(o["outcome"] == "deleted" for o in outcomes)


def test_experiment_needs_its_dataset_seeded_first(ex):
    with pytest.raises(HarnessError, match="isn't seeded"):
        ex.create_experiment("ct1-baseline", {"description": "", "dataset": "<run_id>-qa", "rows": []}, "ct1-qa", [])


def test_a_partly_seeded_step_cannot_be_retried(tmp_path):
    client = FakeClient()
    posts = {"n": 0}
    real = client.create_run

    def flaky(**kw):
        posts["n"] += 1
        if posts["n"] == 2:
            raise RuntimeError("422 start_time out of range")
        real(**kw)

    client.create_run = flaky
    ex = HarnessExecutor(client, "ct1", Manifest(tmp_path / "m.json"), sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        ex.create_project_with_runs("ct1-bot", PROJECT)
    with pytest.raises(HarnessError, match="partly seeded"):
        ex.create_project_with_runs("ct1-bot", PROJECT)
    assert len(client.runs) == 1  # no duplicates
