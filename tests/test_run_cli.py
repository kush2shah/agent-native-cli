"""Safety tests for run_cli. These must pass before anything touches a real workspace."""

import json

import pytest

from tester.inventory import Inventory, discover
from tester.run_cli import CliRunner, decide

from .conftest import TARGET_KEY


@pytest.fixture
def runner(cfg, inventory, tmp_path):
    with CliRunner(cfg, inventory=inventory, log_path=tmp_path / "commands.jsonl", label="t") as r:
        yield r


def log_lines(runner):
    return [json.loads(l) for l in runner.log_path.read_text().splitlines()]


def inventory_with(cfg, tmp_path, overrides_yaml: str, run_id: str) -> Inventory:
    overrides = tmp_path / f"{run_id}.yaml"
    overrides.write_text(overrides_yaml)
    return Inventory(discover(cfg, run_id, use_llm=False, do_probe=False, overrides_path=overrides))


# --- classification ---------------------------------------------------------


def test_unclassified_command_is_denied_and_not_executed(runner):
    res = runner.run(["experiments", "compute"])  # `compute` matches no verb rule
    assert not res.allowed
    assert res.decision.cls == "unknown"
    assert res.exit_code is None
    entry = log_lines(runner)[-1]
    assert entry["allowed"] is False and entry["exit_code"] is None


def test_unknown_verb_denied(inventory):
    d = decide(["debug", "env"], inventory)
    assert d.path == "debug env" and not d.allowed


def test_disallowed_write_is_denied_and_logged(runner):
    res = runner.run(["datasets", "create", "ct-x"])
    assert not res.allowed
    assert "allowed_writes" in res.decision.reason
    assert log_lines(runner)[-1]["reason"] == res.decision.reason


def test_write_in_contract_runs(cfg, inventory):
    with CliRunner(cfg, inventory=inventory, allowed_writes=("datasets create",)) as r:
        res = r.run(["datasets", "create", "ct-x", "--format", "json"])
    assert res.allowed and res.exit_code == 0
    assert json.loads(res.stdout)["name"] == "ct-x"


def test_destructive_denied_even_if_listed_as_allowed_write(cfg, inventory):
    with CliRunner(cfg, inventory=inventory, allowed_writes=("datasets delete",)) as r:
        res = r.run(["datasets", "delete", "ct-x"])
    assert not res.allowed and res.decision.cls == "destructive"


def test_read_runs(runner):
    res = runner.run(["datasets", "list", "--format", "json"])
    assert res.allowed and res.exit_code == 0
    assert json.loads(res.stdout)[0]["name"] == "fixture-dataset"


def test_alias_resolves_to_canonical_path(runner):
    res = runner.run(["dataset", "list"])
    assert res.allowed and res.decision.path == "datasets list"


def test_global_flag_value_before_subcommand(runner):
    res = runner.run(["--format", "json", "datasets", "list"])
    assert res.allowed and res.decision.path == "datasets list"


@pytest.mark.parametrize(
    "argv",
    [
        ["api", "sessions"],
        ["api", "sessions", "-X", "DELETE"],
        ["api", "ls"],
        ["runs", "share", "create", "--run-id", "r1"],
    ],
)
def test_hard_denylist(runner, argv):
    res = runner.run(argv)
    assert not res.allowed and "hard denylist" in res.decision.reason


def test_hard_deny_beats_human_override(cfg, tmp_path):
    inv = inventory_with(cfg, tmp_path, "api: read\nruns share create: read\n", "ov1")
    assert inv.get("api")["class"] == "destructive"
    assert not decide(["api", "sessions"], inv).allowed
    assert not decide(["runs", "share", "create"], inv).allowed


def test_override_applies(cfg, tmp_path):
    inv = inventory_with(cfg, tmp_path, "experiments compute: read\n", "ov2")
    assert decide(["experiments", "compute"], inv).allowed


def test_bad_override_class_rejected(cfg, tmp_path):
    with pytest.raises(ValueError):
        inventory_with(cfg, tmp_path, "experiments compute: maybe\n", "ov3")


# --- argument checks --------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["datasets", "list", "--workspace", "other-ws"],
        ["datasets", "list", "--workspace=other-ws"],
        ["--profile", "prod", "datasets", "list"],
        ["datasets", "list", "--api-url", "https://evil.example"],
        ["runs", "list", "--api-key", "k"],
        ["runs", "list", "--base-url", "https://evil.example"],
        ["runs", "list", "--debug"],
    ],
)
def test_target_switching_flags_denied(runner, argv):
    res = runner.run(argv)
    assert not res.allowed and "flag not allowed" in res.decision.reason


@pytest.mark.parametrize(
    "argv",
    [
        ["datasets", "create", "x", "--description", "@../.env"],
        ["datasets", "create", "x", "--description=@secrets"],
        ["api", "x", "-F", "key=@/etc/passwd"],
        ["datasets", "list", "-o", "/tmp/out.json"],
        ["datasets", "list", "--output=/tmp/out.json"],
        ["datasets", "list", "-o", "~/out.json"],
        ["datasets", "list", "-o", "../out.json"],
        ["datasets", "list", "-o", "a/../../out.json"],
    ],
)
def test_file_references_and_escaping_paths_denied(cfg, inventory, argv):
    with CliRunner(cfg, inventory=inventory, allowed_writes=("datasets create",)) as r:
        res = r.run(argv)
    assert not res.allowed
    assert res.exit_code is None


def test_relative_output_path_allowed(runner):
    res = runner.run(["datasets", "list", "-o", "out.json"])
    assert res.allowed


# --- help -------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["datasets", "--help"], ["datasets", "delete", "--help"], ["api", "--help"], ["help", "datasets"]],
)
def test_help_always_allowed(runner, argv):
    res = runner.run(argv)
    assert res.allowed and res.decision.is_help and res.exit_code == 0


def test_help_cannot_smuggle_a_denied_call(inventory):
    assert not decide(["api", "sessions", "-X", "DELETE", "--help"], inventory).allowed


def test_invented_subcommand_is_denied(runner):
    # The real CLI would print help and exit 0. We deny instead: an incomplete
    # crawl must never turn into "anything under this group runs".
    res = runner.run(["datasets", "frobnicate"])
    assert not res.allowed and "unknown subcommand 'frobnicate'" in res.decision.reason
    res = runner.run(["frobnicate", "delete"])
    assert not res.allowed and res.exit_code is None


def test_bare_group_prints_help(runner):
    res = runner.run(["datasets"])
    assert res.allowed and res.exit_code == 0 and "Available Commands" in res.stdout


def test_invented_flag_reaches_cli(runner):
    res = runner.run(["datasets", "list", "--frobnicate"])
    assert res.allowed and res.exit_code == 1
    assert "unknown flag: --frobnicate" in res.stderr


def test_no_inventory_only_allows_help():
    assert decide(["datasets", "list", "--help"], None).allowed
    assert decide(["--version"], None).allowed
    assert not decide(["datasets", "list"], None).allowed
    assert not decide(["datasets", "--limit", "1", "--help"], None).allowed


# --- subprocess sandbox -----------------------------------------------------


def run_debug(cfg, inventory, sub):
    data = json.loads(json.dumps(inventory.data))
    for c in data["commands"]:
        if c["path"].startswith("debug "):
            c["class"] = "read"
    with CliRunner(cfg, inventory=Inventory(data)) as r:
        return r, r.run(["debug", sub])


def test_subprocess_env_is_exactly_path_home_and_target_key(cfg, inventory):
    r, res = run_debug(cfg, inventory, "env")
    env = json.loads(res.stdout)
    # macOS may inject a few CoreFoundation/locale vars into Python itself; nothing else may appear.
    env = {k: v for k, v in env.items() if not k.startswith("__CF") and k != "LC_CTYPE"}
    assert set(env) == {"PATH", "HOME", "LANGSMITH_API_KEY"}
    assert env["LANGSMITH_API_KEY"] == TARGET_KEY
    assert env["HOME"] == str(r.home)


def test_subprocess_runs_in_temp_workdir(cfg, inventory):
    r, res = run_debug(cfg, inventory, "pwd")
    assert res.stdout.strip() == str(r.workdir.resolve())


def test_hanging_command_is_killed_at_timeout(cfg, inventory):
    _, res = run_debug(cfg, inventory, "sleep")
    assert res.timed_out and res.exit_code is None
    assert res.duration_s < 10


def test_max_calls_budget(cfg, inventory):
    with CliRunner(cfg, inventory=inventory, max_calls=2) as r:
        assert r.run(["datasets", "list"]).allowed
        assert r.run(["datasets", "list"]).allowed
        res = r.run(["datasets", "list"])
    assert not res.allowed and "budget" in res.decision.reason


def test_rejects_non_list_args(runner):
    with pytest.raises(TypeError):
        runner.run("datasets list")
