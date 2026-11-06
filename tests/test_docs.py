import json

from tester.docs import DocsIndex, check_command, check_docs, extract_commands, snapshot, split_shell
from tester.inventory import parse_help

PAGE = """\
# Datasets

List them with `langsmith datasets list`, or see `langsmith dataset`.

```bash theme={"x": 1}
$ langsmith datasets list --limit 5 \\
    --format json
cd work && langsmith datasets get my-ds | jq .
langsmith runs list --limit 'a && b'
```

```python
import langsmith
langsmith = Client()
```

```hcl
langsmith = {
  source = "x"
}
```
"""


def texts(md):
    return [(c["where"], c["text"]) for c in extract_commands(md)]


def test_extract_commands_from_shell_blocks_and_inline_code():
    found = texts(PAGE)
    assert ("inline", "langsmith datasets list") in found
    assert ("inline", "langsmith dataset") in found
    assert ("code", "langsmith datasets list --limit 5 --format json") in found  # continuation joined, prompt stripped
    assert ("code", "langsmith datasets get my-ds") in found  # split on && and |
    assert ("code", "langsmith runs list --limit 'a && b'") in found  # && inside quotes is not a separator
    assert not any("Client" in t or "source" in t or t == "langsmith = {" for _, t in found)  # python / hcl skipped


def test_split_shell_respects_quotes():
    assert split_shell("a && b | c; d") == ["a", "b", "c", "d"]
    assert split_shell("x -c 'cd /tmp && y' && z") == ["x -c 'cd /tmp && y'", "z"]


def test_check_command_matches(inventory):
    assert check_command("langsmith datasets list --limit 5 --format json", inventory) == []
    assert check_command("langsmith dataset get my-ds", inventory) == []  # alias
    assert check_command("langsmith datasets get <name>", inventory) == []
    assert check_command("langsmith datasets --help", inventory) == []


def kinds(text, inv, where="code"):
    return [i["kind"] for i in check_command(text, inv, where)]


def test_check_command_findings(inventory):
    assert kinds("langsmith datasets list --project-id x", inventory) == ["unknown_flag"]
    assert kinds("langsmith datasets frobnicate", inventory) == ["unknown_command"]
    assert kinds("langsmith fleet skills pull", inventory) == ["unknown_command"]
    assert kinds("langsmith datasets get", inventory) == ["missing_args"]
    assert kinds("langsmith datasets get", inventory, where="inline") == []  # inline mentions name commands
    assert kinds("langsmith datasets 'unclosed", inventory) == ["unparseable"]


def test_placeholders_are_not_unknown_commands(inventory):
    for text in ("langsmith <command> --help", "langsmith datasets $CMD", "langsmith datasets ..."):
        assert kinds(text, inventory) == [], text


def test_hidden_api_key_value_is_not_read_as_a_command(inventory):
    # Without special handling the key would be taken as a positional and
    # everything after it misread.
    assert kinds("langsmith --api-key lsv2_x datasets list --limit 1", inventory) == ["hidden_flag"]
    assert kinds("langsmith --api-key=lsv2_x datasets list", inventory) == ["hidden_flag"]


def test_flag_value_placeholder_is_not_a_required_positional():
    help_text = "Do it\n\nUsage:\n  langsmith sandboxes tunnel <name> --remote-port <port> [flags]\n"
    assert parse_help(help_text)["required_args"] == ["<name>"]


ROOT = """\
# Docs
- [LangSmith (3 pages)](https://docs.example/_llms/ls.txt): LangSmith
- [REST (500 pages)](https://docs.example/_llms/lang-smith-rest-api.txt): skipped
"""
LS_INDEX = """\
### Datasets
- [Manage datasets](https://docs.example/langsmith/datasets.md): Create and list datasets.
- [Changelog](https://docs.example/langsmith/self-hosted-changelog.md): Releases.
- [OSS page](https://docs.example/oss/python/x.md): not LangSmith
- [Relative](/langsmith/evaluation.md): Run evals on a dataset.
"""
PAGES = {
    "https://docs.example/llms.txt": ROOT,
    "https://docs.example/_llms/ls.txt": LS_INDEX,
    "https://docs.example/langsmith/datasets.md": PAGE,
    "https://docs.example/langsmith/self-hosted-changelog.md": "Use `langsmith datasets share get`.",
    "https://docs.example/langsmith/evaluation.md": "# Evaluation\n\n## Experiments\n\nAn experiment runs your agent over a dataset.\n",
}


def fake_fetch(url):
    if "rest-api" in url:
        raise AssertionError("REST reference index must not be fetched")
    return PAGES[url]


def test_snapshot_follows_indexes_and_filters(tmp_path):
    data = snapshot(tmp_path, "https://docs.example/llms.txt", fetch=fake_fetch, workers=2)
    urls = [p["url"] for p in data["pages"]]
    assert urls == [
        "https://docs.example/langsmith/datasets.md",
        "https://docs.example/langsmith/evaluation.md",
        "https://docs.example/langsmith/self-hosted-changelog.md",
    ]
    assert data["errors"] == []
    saved = json.loads((tmp_path / "index.json").read_text())
    assert (tmp_path / "pages" / saved["pages"][0]["file"]).read_text() == PAGE


def test_check_docs_skips_changelogs_and_reports_coverage(tmp_path, inventory):
    snapshot(tmp_path, "https://docs.example/llms.txt", fetch=fake_fetch, workers=2)
    r = check_docs(tmp_path, inventory)
    assert not any("changelog" in res["url"] for res in r["results"])
    assert r["commands_with_issues"] == 0
    assert "datasets list" not in r["undocumented"]
    assert "datasets delete" in r["undocumented"]


def test_search_ranks_the_relevant_section(tmp_path):
    snapshot(tmp_path, "https://docs.example/llms.txt", fetch=fake_fetch, workers=2)
    hits = DocsIndex(tmp_path).search("run an experiment over a dataset", k=2)
    assert hits[0]["url"].endswith("evaluation.md")
    assert hits[0]["heading"] == "Experiments"
