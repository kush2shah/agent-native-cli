from types import SimpleNamespace

from tester.classify import classify, hard_deny_reason
from tester.inventory import Inventory, discover, parse_help

# Trimmed from the real CLI (v0.3.13): a group whose command summary wraps onto
# unindented lines, and a generated urfave-style leaf.
COBRA_GROUP = """\
List and inspect tracing projects (sessions)

Usage:
  langsmith projects [command]

Aliases:
  projects, project, sessions

Available Commands:
  delete-everything Is the longest name, so cobra leaves one space
  delete      Permanently delete a tracing project and all of its traces
  resolve     Returns the tracing project (session) an address names. An address is an AGENT
(`id` and `environment`), an EXPERIMENT (`id`), or an EVALUATOR (no `id`:
evaluator traces share one project per workspace).
  update      Update a project.

Flags:
  -h, --help   help for projects

Global Flags:
      --api-url string     LangSmith API URL [env: LANGSMITH_ENDPOINT]
      --format string      Output format: pretty or json (default "pretty")

Use "langsmith projects [command] --help" for more information about a command.
"""

COBRA_LEAF = """\
Get dataset details by name or UUID

Usage:
  langsmith datasets get NAME_OR_ID [flags]

Flags:
  -h, --help            help for get
  -o, --output string   Write JSON output to a file
"""

URFAVE_LEAF = """\
NAME:
   langsmith runs:share create - Creates or returns a share token for a run.

USAGE:
   langsmith runs:share create [options]

OPTIONS:
   --run-id string
   --session-id string  session_id is the tracing project UUID containing the trace.
   --help, -h           show help

GLOBAL OPTIONS:
   --debug                   Enable debug logging
   --base-url string         Override the base URL for API requests (default: url)
   --raw-output, -r          If the result is a string, print it without JSON quotes.
"""

URFAVE_GROUP = """\
NAME:
   langsmith runs:streaming

USAGE:
   langsmith runs:streaming [command [command options]]

COMMANDS:
   query-v2  Returns a paginated list of runs for the given projects within min/max
             start_time. Supports filters.

OPTIONS:
   --help, -h  show help
"""


def test_parse_cobra_group():
    h = parse_help(COBRA_GROUP)
    assert h["format"] == "cobra" and not h["runnable"]
    assert [c["name"] for c in h["children"]] == ["delete-everything", "delete", "resolve", "update"]
    assert h["aliases"] == ["project", "sessions"]
    assert {"name": "format", "short": None, "type": "string"} in h["flags"]


def test_parse_cobra_leaf():
    h = parse_help(COBRA_LEAF)
    assert h["runnable"] and h["required_args"] == ["NAME_OR_ID"]
    assert {"name": "output", "short": "o", "type": "string"} in h["flags"]
    assert {"name": "help", "short": "h", "type": None} in h["flags"]


def test_parse_slash_placeholder():
    h = parse_help("List tags\n\nUsage:\n  langsmith prompt tag list OWNER/REPO [flags]\n")
    assert h["required_args"] == ["OWNER/REPO"]


def test_parse_urfave_leaf():
    h = parse_help(URFAVE_LEAF)
    assert h["format"] == "urfave" and h["runnable"]
    assert h["summary"] == "Creates or returns a share token for a run."
    flags = {f["name"]: f for f in h["flags"]}
    assert flags["run-id"]["type"] == "string"
    assert flags["help"]["short"] == "h"
    assert flags["raw-output"] == {"name": "raw-output", "short": "r", "type": None}
    assert flags["base-url"]["type"] == "string"


def test_parse_urfave_group():
    h = parse_help(URFAVE_GROUP)
    assert not h["runnable"]
    assert [c["name"] for c in h["children"]] == ["query-v2"]


def test_classify_precedence():
    assert classify("api", True, {"api": "read"}) == ("destructive", "hard_deny")
    assert classify("runs share create", True, {}) == ("destructive", "hard_deny")
    assert classify("datasets get", True, {"datasets get": "write"}) == ("write", "override")
    assert classify("datasets", False, {}) == ("read", "group")
    assert classify("datasets list", True, {}) == ("read", "rule:verb")
    assert classify("datasets delete", True, {}) == ("destructive", "rule:verb")
    assert classify("datasets create", True, {}) == ("write", "rule:verb")
    assert classify("experiments compute", True, {}) == ("unknown", "default")
    assert classify("online-evaluators bulk-delete", True, {}) == ("destructive", "rule:verb")
    assert classify("examples delete-all", True, {}) == ("destructive", "rule:verb")
    assert classify("threads aggregate-stats", True, {}) == ("read", "rule:verb")
    assert classify("datasets update-tags", True, {}) == ("write", "rule:verb")
    assert classify("datasets retrieve-version", True, {}) == ("read", "rule:verb")


def test_hard_deny_matches_whole_tokens():
    assert hard_deny_reason("api")
    assert hard_deny_reason("api ls")
    assert not hard_deny_reason("apps list")  # `apps` is not `api`
    assert hard_deny_reason("datasets share")
    assert not hard_deny_reason("datasets shared-list")


def test_discover_classifies_everything(inventory_data):
    cmds = {c["path"]: c for c in inventory_data["commands"]}
    assert inventory_data["cli_version"] == "langsmith version 0.0.1-fake"
    assert all(c["class"] in ("read", "write", "destructive", "unknown") for c in cmds.values())
    assert not [c for c in cmds.values() if c.get("crawl_error")]
    assert cmds["datasets list"]["class"] == "read"
    assert cmds["experiments compute"]["class"] == "unknown"
    assert cmds["runs share create"]["format"] == "urfave"
    assert "help" not in cmds


def test_discover_probes(inventory_data):
    cmds = {c["path"]: c for c in inventory_data["commands"]}
    assert cmds["datasets list"]["probe"] == "ok"
    assert cmds["projects list"]["probe"] == "permission_denied"
    assert "403" in cmds["projects list"]["probe_detail"]
    assert cmds["experiments list"]["probe"] == "error"
    assert cmds["datasets get"]["probe"] == "not_probed"  # needs an argument


def test_wrong_key_warns_instead_of_reporting_blind_spots(cfg, no_overrides):
    from dataclasses import replace

    # Like the real CLI, the fake answers a wrong key with 403 everywhere.
    wrong = replace(cfg, target_env={"LANGSMITH_API_KEY": "wrong-key"})
    data = discover(wrong, "wrongkey", use_llm=False, overrides_path=no_overrides)
    assert not any(c["probe"] == "ok" for c in data["commands"])
    assert any("every probe was refused" in n for n in data["notes"])


def test_probes_skipped_without_target_key(cfg, no_overrides, monkeypatch):
    from dataclasses import replace

    data = discover(replace(cfg, target_env={}), "nokey", use_llm=False, overrides_path=no_overrides)
    assert all(c["probe"] == "not_probed" for c in data["commands"])
    assert any("probes skipped" in n for n in data["notes"])


def test_discover_writes_files(cfg, inventory_data):
    run_dir = cfg.runs_dir / "test"
    assert Inventory.load(run_dir / "inventory.json").has("datasets list")
    assert (run_dir / "commands.jsonl").read_text().count("\n") > 10


class FakeModel:
    """Suggests `read` for everything, including things that are clearly not reads."""

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self

    def invoke(self, messages):
        user = messages[-1][1]
        paths = [line[5:-1] for line in user.splitlines() if line.startswith("### `")]
        return SimpleNamespace(
            suggestions=[SimpleNamespace(path=p, classification="read", reason="looks harmless") for p in paths]
        )


def test_llm_suggestions_are_never_applied(cfg, no_overrides):
    data = discover(cfg, "llm", overrides_path=no_overrides, model=FakeModel(), do_probe=False)
    compute = next(c for c in data["commands"] if c["path"] == "experiments compute")
    assert compute["class"] == "unknown"
    assert compute["suggested_class"] == "read"
    suggestions = (cfg.runs_dir / "llm" / "suggested_classifications.yaml").read_text()
    assert "experiments compute: read" in suggestions
