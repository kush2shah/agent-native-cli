from tester.seed import check_warnings, run_count, validate_seed_data

DS = {"id": "s01", "kind": "dataset", "name": "<run_id>-qa",
      "data": {"description": "", "examples": [{"inputs_json": '{"q": "a"}', "outputs_json": '{"a": "Paris"}'}]}}


def experiment(dataset="<run_id>-qa", index=0):
    return {"id": "s02", "kind": "experiment", "name": "<run_id>-baseline",
            "data": {"description": "", "dataset": dataset, "rows": [
                {"example_index": index, "outputs_json": "{}", "error": "", "latency_s": 1, "scores": []}]}}


def run(repeat=1, children=0):
    return {"name": "r", "run_type": "chain", "inputs_json": "{}", "outputs_json": "{}", "error": "",
            "start_offset_s": 1, "latency_s": 1, "repeat": repeat,
            "children": [{"name": "c", "run_type": "tool", "inputs_json": "{}", "outputs_json": "{}", "error": "",
                          "start_after_s": 0, "latency_s": 1}] * children}


def test_valid_experiment_passes():
    assert validate_seed_data([DS, experiment()]) == []


def test_experiment_must_reference_an_earlier_seeded_dataset():
    assert "isn't a seeded dataset" in validate_seed_data([DS, experiment("<run_id>-other")])[0]
    assert "seeded before" in validate_seed_data([experiment(), DS])[0]
    assert "out of range" in validate_seed_data([DS, experiment(index=3)])[0]


def test_run_volume_is_capped_counting_repeats_and_children():
    agent = {"id": "s03", "kind": "agent", "name": "<run_id>-bot", "data": {"display_name": "Bot", "environments": [
        {"environment": "staging", "runs": [run(repeat=120, children=2)]}]}}
    assert run_count("agent", agent["data"]) == 360
    assert "at most 300" in validate_seed_data([agent])[0]


def test_bad_metadata_json_is_caught():
    bad = run()
    bad["metadata_json"] = "release=v2"
    proj = {"id": "s04", "kind": "project_with_runs", "name": "<run_id>-p", "data": {"description": "", "runs": [bad]}}
    assert "not a JSON object" in validate_seed_data([proj])[0]


def test_warns_when_a_check_value_is_missing_from_the_data():
    plan = {"tasks": [{"id": "t01", "hard_checks": [
        {"describe": "names the city", "method": "output_contains_all", "values": ["Paris", "Lyon"]}]}]}
    assert check_warnings([DS], plan) == ["t01: check 'names the city' looks for ['Lyon'], not found in seed data"]


def test_runs_older_than_a_day_are_rejected():
    old = run(repeat=3)
    old["start_offset_s"] = 20 * 3600
    old["repeat_spacing_s"] = 2 * 3600  # third copy starts 24h ago
    proj = {"id": "s05", "kind": "project_with_runs", "name": "<run_id>-p", "data": {"description": "", "runs": [old]}}
    assert "last 24 hours" in validate_seed_data([proj])[0]
