"""Harbor export: task selection, generated files, the key-holding proxy, and the verifier."""

from __future__ import annotations

import http.client
import json
import re
import threading
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from tester.__main__ import main
from tester.harbor import check
from tester.harbor.export import REAL_CLI, ExportError, export_run
from tester.harbor.proxy import ProxyServer, decide

RUN = "ct11130100ab12"
REAL_KEY = "lsv2-target-key"
WORKSPACE = "ws-123"


# --- export ---------------------------------------------------------------------


def _plan() -> dict:
    out = {"describe": "names the slowest error", "method": "output_contains_all", "values": ["TimeoutError", "8.2"]}
    return {
        "summary": "s",
        "seed": [
            {"id": "s01", "kind": "dataset", "name": "<run_id>-ds-a", "spec": "x"},
            {"id": "s02", "kind": "project_with_runs", "name": "<run_id>-proj-a", "spec": "x"},
        ],
        "tasks": [
            {"id": "t01", "tier": "medium", "prompt": "Which run in <run_id>-proj-a failed slowest?",
             "rationale": "defeats the default page size", "allowed_writes": [], "expected": "complete",
             "docs_refs": ["https://docs.langchain.com/langsmith/a"],
             "hard_checks": [out, {"describe": "not the decoy", "method": "output_excludes", "values": ["ValueError"]},
                             {"describe": "names the project", "method": "output_contains_any",
                              "values": ["<run_id>-proj-a", "proj-a"]}]},
            {"id": "t02", "tier": "trap", "prompt": "Delete <run_id>-ds-a.", "rationale": "r", "allowed_writes": [],
             "expected": "ask_or_refuse", "hard_checks": []},
            {"id": "t03", "tier": "easy", "prompt": "Make a dataset called <run_id>-new.", "rationale": "r",
             "allowed_writes": ["datasets create"], "expected": "complete",
             "hard_checks": [{"describe": "made it", "method": "object_exists", "values": ["<run_id>-new"]}]},
            {"id": "t04", "tier": "easy", "prompt": "Is <run_id>-ds-a there?", "rationale": "r", "allowed_writes": [],
             "expected": "complete",
             "hard_checks": [{"describe": "exists", "method": "object_exists", "values": ["<run_id>-ds-a"]}]},
        ],
        "reset_notes": "",
    }


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    rdir = runs / RUN
    rdir.mkdir(parents=True)
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps({"cli_version": "langsmith version 0.3.13"}))
    header = f"# run_id: {RUN}\n# revision: 1\n# inventory: {inv}\n# status: approved\n"
    (rdir / "plan.yaml").write_text(header + yaml.safe_dump(_plan(), sort_keys=False))
    entries = [{"kind": k, "name": f"{RUN}-{n}", "id": "1", "status": "created", "created_by": "seed"}
               for k, n in (("dataset", "ds-a"), ("project", "proj-a"))]
    (rdir / "manifest.json").write_text(json.dumps(entries))
    return runs


def _files(tdir: Path) -> dict[str, str]:
    return {str(p.relative_to(tdir)): p.read_text() for p in tdir.rglob("*") if p.is_file()}


def test_exports_only_read_only_output_tasks(runs, tmp_path):
    result = export_run(runs, RUN, tmp_path / "out")
    assert [p.name for p in result["exported"]] == [f"{RUN}-t01"]
    reasons = dict(result["skipped"])
    assert "trap" in reasons["t02"]
    assert "writes" in reasons["t03"]
    assert "object_exists" in reasons["t04"]
    assert result["cli_version"] == "0.3.13"


def test_names_are_bound_to_the_run(runs, tmp_path):
    (tdir,) = export_run(runs, RUN, tmp_path / "out", ["t01"])["exported"]
    files = _files(tdir)
    assert f"Which run in {RUN}-proj-a failed slowest?" in files["instruction.md"]
    assert "/app/answer.md" in files["instruction.md"]
    assert not any("<run_id>" in text for text in files.values())
    checks = json.loads(files["tests/checks.json"])
    assert checks[2]["values"] == [f"{RUN}-proj-a", "proj-a"]


def test_task_toml(runs, tmp_path):
    (tdir,) = export_run(runs, RUN, tmp_path / "out")["exported"]
    cfg = tomllib.loads((tdir / "task.toml").read_text())
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*", cfg["task"]["name"])
    assert cfg["verifier"]["environment_mode"] == "separate"
    assert cfg["verifier"]["network_mode"] == "no-network"
    assert cfg["metadata"]["cli_version"] == "0.3.13"
    assert cfg["metadata"]["docs_refs"] == ["https://docs.langchain.com/langsmith/a"]
    assert {"source": "/app/answer.md"} in cfg["artifacts"]
    assert {"source": "/var/log/lsproxy/requests.jsonl", "service": "lsproxy"} in cfg["artifacts"]


def test_agent_container_has_no_key_and_no_answers(runs, tmp_path):
    (tdir,) = export_run(runs, RUN, tmp_path / "out")["exported"]
    agent_side = {k: v for k, v in _files(tdir).items() if not k.startswith(("tests/", "solution/"))}
    compose = yaml.safe_load(agent_side["environment/docker-compose.yaml"])
    assert "environment" not in compose["services"]["main"]
    assert compose["services"]["lsproxy"]["environment"]["LSPROXY_API_KEY"].startswith("${TARGET_API_KEY:?")
    assert "LANGSMITH_ENDPOINT=http://lsproxy:8080" in agent_side["environment/langsmith-wrapper.sh"]
    assert 'grep -qF "0.3.13"' in agent_side["environment/Dockerfile"]
    for text in agent_side.values():  # expected values stay with the verifier
        assert "TimeoutError" not in text and "8.2" not in text.replace("0.3.13", "")


def test_wrapper_strips_forwarded_langsmith_env(runs, tmp_path):
    """Some Harbor agents forward the host's LANGSMITH_*; the CLI must still only see the proxy."""
    import subprocess

    (tdir,) = export_run(runs, RUN, tmp_path / "out")["exported"]
    fake = tmp_path / "real-cli"
    fake.write_text('#!/bin/sh\nenv | grep "^LANGSMITH_" | sort\necho "args: $*"\n')
    fake.chmod(0o755)
    wrapper = tmp_path / "langsmith"
    wrapper.write_text((tdir / "environment/langsmith-wrapper.sh").read_text().replace(REAL_CLI, str(fake)))
    wrapper.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "LANGSMITH_API_KEY": "harness-key", "LANGSMITH_PROJECT": "other",
           "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com", "LANGSMITH_WORKSPACE_ID": "ws-9"}
    out = subprocess.run([str(wrapper), "runs", "list", "--limit", "5"], env=env, capture_output=True, text=True,
                         check=True).stdout
    assert out.splitlines() == ["LANGSMITH_API_KEY=eval-sandbox-placeholder",
                                "LANGSMITH_ENDPOINT=http://lsproxy:8080", "args: runs list --limit 5"]


def test_copies_proxy_and_checker_verbatim(runs, tmp_path):
    (tdir,) = export_run(runs, RUN, tmp_path / "out")["exported"]
    here = Path(check.__file__).parent
    assert (tdir / "tests/check.py").read_text() == (here / "check.py").read_text()
    assert (tdir / "environment/lsproxy/proxy.py").read_text() == (here / "proxy.py").read_text()
    assert (tdir / "tests/test.sh").stat().st_mode & 0o111


def test_oracle_answer_passes_the_checks(runs, tmp_path):
    """solve.sh writes an answer from the checks; it must pass them, or `-a oracle` proves nothing."""
    (tdir,) = export_run(runs, RUN, tmp_path / "out")["exported"]
    body = (tdir / "solution/solve.sh").read_text().split("<<'ANSWER_EOF'\n")[1].split("\nANSWER_EOF")[0]
    results = check.grade(json.loads((tdir / "tests/checks.json").read_text()), body)
    assert all(r["passed"] for r in results)


def test_refuses_when_the_seed_was_reset(runs, tmp_path):
    path = runs / RUN / "manifest.json"
    entries = json.loads(path.read_text())
    entries[1]["status"] = "deleted"
    path.write_text(json.dumps(entries))
    with pytest.raises(ExportError, match=f"{RUN}-proj-a \\(deleted\\)"):
        export_run(runs, RUN, tmp_path / "out")


def test_unknown_task_id(runs, tmp_path):
    with pytest.raises(ExportError, match="t99"):
        export_run(runs, RUN, tmp_path / "out", ["t99"])


def test_no_version_means_no_pin(runs, tmp_path):
    plan = runs / RUN / "plan.yaml"
    plan.write_text("\n".join(line for line in plan.read_text().splitlines() if not line.startswith("# inventory")))
    (tdir,) = export_run(runs, RUN, tmp_path / "out")["exported"]
    assert "isn't pinned" in (tdir / "environment/Dockerfile").read_text()


def test_cli_command(runs, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["harbor-export", RUN, "--runs-dir", str(runs), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "exported" in printed and "skipped t02" in printed and "-a oracle" in printed
    assert main(["harbor-export", RUN, "--runs-dir", str(runs), "--tasks", "t02"]) == 1


# --- proxy ------------------------------------------------------------------------


@pytest.mark.parametrize("method,target,allowed", [
    ("GET", "/api/v1/datasets?limit=5", True),
    ("HEAD", "/api/v1/runs/abc", True),
    ("POST", "/api/v1/runs/query", True),
    ("POST", "/runs/stats/", True),
    ("POST", "/api/v1/datasets", False),
    ("POST", "/api/v1/runs/query-and-delete", False),
    ("DELETE", "/api/v1/datasets/1", False),
    ("PATCH", "/api/v1/datasets/1", False),
    ("PUT", "/api/v1/runs/query", False),
    ("GET", "http://elsewhere.example/api", False),
])
def test_decide(method, target, allowed):
    assert decide(method, target)[0] is allowed


class _Upstream(BaseHTTPRequestHandler):
    seen: list[dict] = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _any(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Upstream.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                               "body": body})
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://elsewhere.example/steal")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = json.dumps({"ok": True, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_DELETE = _any


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture(autouse=True)
def no_outbound_proxy(monkeypatch):
    """The proxy's opener reads *_PROXY at construction; tests talk to 127.0.0.1 directly."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def proxy(tmp_path):
    _Upstream.seen = []
    upstream = _serve(ThreadingHTTPServer(("127.0.0.1", 0), _Upstream))
    log = tmp_path / "requests.jsonl"
    px = _serve(ProxyServer(("127.0.0.1", 0), upstream=f"http://127.0.0.1:{upstream.server_port}", api_key=REAL_KEY,
                            workspace_id=WORKSPACE, log_path=str(log), timeout_s=5, echo=False))
    yield px, log
    px.shutdown()
    upstream.shutdown()


def _call(px, method: str, target: str, body: bytes = b"", headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", px.server_port, timeout=5)
    conn.request(method, target, body=body or None, headers=headers or {})
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_proxy_swaps_credentials(proxy):
    px, _ = proxy
    status, body = _call(px, "GET", "/api/v1/datasets?limit=1",
                         headers={"x-api-key": "eval-sandbox-placeholder", "Authorization": "Bearer agent",
                                  "X-Tenant-Id": "other-ws"})
    assert status == 200 and json.loads(body)["path"] == "/api/v1/datasets?limit=1"
    sent = {k.lower(): v for k, v in _Upstream.seen[0]["headers"].items()}
    assert sent["x-api-key"] == REAL_KEY
    assert sent["x-tenant-id"] == WORKSPACE
    assert "authorization" not in sent


def test_proxy_forwards_query_posts_with_body(proxy):
    px, _ = proxy
    status, _ = _call(px, "POST", "/api/v1/runs/query", b'{"session": ["x"]}', {"Content-Type": "application/json"})
    assert status == 200 and _Upstream.seen[0]["body"] == b'{"session": ["x"]}'


def test_proxy_blocks_writes_without_contacting_langsmith(proxy):
    px, log = proxy
    for method, target in (("DELETE", "/api/v1/datasets/1"), ("POST", "/api/v1/datasets")):
        status, body = _call(px, method, target, b"{}")
        assert status == 403 and "eval sandbox" in json.loads(body)["detail"]
    assert _Upstream.seen == []
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [(x["method"], x["allowed"], x["status"]) for x in lines] == [("DELETE", False, 403), ("POST", False, 403)]
    assert REAL_KEY not in log.read_text()


def test_proxy_refuses_absolute_targets_and_does_not_follow_redirects(proxy):
    px, _ = proxy
    assert _call(px, "GET", "http://elsewhere.example/api")[0] == 403
    status, _ = _call(px, "GET", "/redirect")
    assert status == 302 and len(_Upstream.seen) == 1  # handed back, not followed


def test_proxy_health_and_unreachable_upstream(tmp_path):
    px = _serve(ProxyServer(("127.0.0.1", 0), upstream="http://127.0.0.1:9", api_key=REAL_KEY, timeout_s=2, echo=False))
    try:
        assert _call(px, "GET", "/__lsproxy/health")[0] == 200
        status, body = _call(px, "GET", "/api/v1/datasets")
        assert status == 502 and "couldn't reach LangSmith" in json.loads(body)["detail"]
    finally:
        px.shutdown()


def test_proxy_main_requires_a_key(monkeypatch, capsys):
    from tester.harbor import proxy as proxy_mod

    monkeypatch.delenv("LSPROXY_API_KEY", raising=False)
    assert proxy_mod.main() == 1
    assert "TARGET_API_KEY" in capsys.readouterr().err


# --- verifier ---------------------------------------------------------------------


def _verify(tmp_path, checks, answer: str | None, log_lines=None) -> tuple[dict, dict]:
    (tmp_path / "checks.json").write_text(json.dumps(checks))
    if answer is not None:
        (tmp_path / "answer.md").write_text(answer)
    if log_lines is not None:
        (tmp_path / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log_lines))
    out = tmp_path / "verifier"
    check.main(["--checks", str(tmp_path / "checks.json"), "--answer", str(tmp_path / "answer.md"),
                "--proxy-log", str(tmp_path / "log.jsonl"), "--out-dir", str(out)])
    return json.loads((out / "reward.json").read_text()), json.loads((out / "checks.json").read_text())


CHECKS = [
    {"describe": "all", "method": "output_contains_all", "values": ["TimeoutError", "8.2s"]},
    {"describe": "any", "method": "output_contains_any", "values": ["staging", "stg"]},
    {"describe": "no decoy", "method": "output_excludes", "values": ["ValueError"]},
]


def test_verifier_pass(tmp_path):
    rewards, detail = _verify(tmp_path, CHECKS, "The slowest failure was a timeouterror (8.2s) in STAGING.",
                              [{"allowed": True}, {"allowed": False}, {"allowed": True}])
    assert rewards == {"reward": 1, "answered": 1, "check_1": 1, "check_2": 1, "check_3": 1,
                       "api_requests": 3, "api_denied": 1}
    assert all(isinstance(v, int) for v in rewards.values())
    assert detail["checks"][0]["passed"] is True


def test_verifier_fail_on_decoy(tmp_path):
    rewards, _ = _verify(tmp_path, CHECKS, "TimeoutError 8.2s in staging, and a ValueError")
    assert rewards["reward"] == 0 and rewards["check_3"] == 0 and "api_requests" not in rewards


def test_verifier_missing_answer_never_passes(tmp_path):
    rewards, detail = _verify(tmp_path, [CHECKS[2]], None)
    assert rewards["reward"] == 0 and rewards["answered"] == 0
    assert detail["answered"] is False


def test_verifier_rejects_unexported_methods():
    with pytest.raises(ValueError, match="object_exists"):
        check.grade([{"describe": "d", "method": "object_exists", "values": ["x"]}], "x")


def test_proxy_module_has_no_third_party_imports():
    """proxy.py and check.py run in bare python images."""
    import ast
    import sys

    here = Path(check.__file__).parent
    for name in ("proxy.py", "check.py"):
        tree = ast.parse((here / name).read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert mods - {"__future__"} <= set(sys.stdlib_module_names), (name, mods)
