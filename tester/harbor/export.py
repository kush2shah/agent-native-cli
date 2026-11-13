"""Export a run's approved plan as Harbor tasks, so off-the-shelf agents attempt the same tasks.

Our worker is a hand-built ReAct loop with one tool, `run_cli`, and no shell. Real
coding agents (Claude Code, Codex, ...) have a shell: they pipe into `jq`, read
`--help` through `head`, and retry differently. Harbor runs those agents against a
task directory in a container, so exporting lets us check whether our friction
findings hold for the agents people actually use. See docs/harbor.md.

Spike scope, kept narrow because the agent gets a shell in the container:
- Read-only `complete` tasks with output checks only. Traps, writes, and `object_*`
  checks are skipped with a reason.
- The seed must still be live: run with `keep` at confirm reset, then export.
- The target key never enters the agent's container. A sidecar (`proxy.py`) holds it
  and forwards reads only.

One task directory per plan task:

    <run_id>-<task_id>/
      task.toml                  # verifier in its own container, no network
      instruction.md             # the bound prompt, plus where to put the answer
      environment/
        Dockerfile               # the agent's image: the LangSmith CLI behind a wrapper
        langsmith-wrapper.sh     # forces the proxy endpoint and a placeholder key
        docker-compose.yaml      # adds the lsproxy sidecar (gets the key from the host's env)
        lsproxy/                 # the sidecar's image
      tests/                     # the verifier's image: check.py + checks.json
      solution/solve.sh          # plumbing check for `-a oracle`, not a reference solution
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from tester.manifest import CREATED, Manifest
from tester.plan import RUN_PREFIX, bind, load_plan

HERE = Path(__file__).parent
ORG = "agent-native-cli"
OUTPUT_CHECKS = ("output_contains_all", "output_contains_any", "output_excludes")
ANSWER_PATH = "/app/answer.md"
PROXY_LOG = "/var/log/lsproxy/requests.jsonl"
_VERSION = re.compile(r"\d+\.\d+\.\d+[\w.+-]*")


class ExportError(Exception):
    pass


def skip_reason(task: dict) -> str | None:
    """Why a task can't be exported yet, or None if it can."""
    if task["expected"] != "complete":
        return f"expected={task['expected']}: traps need the judge and a write-aware proxy"
    if task.get("allowed_writes"):
        return "it writes; the proxy only forwards reads so far"
    if not task.get("hard_checks"):
        return "no hard checks"
    other = sorted({c["method"] for c in task["hard_checks"] if c["method"] not in OUTPUT_CHECKS})
    if other:
        return f"{', '.join(other)} checks need workspace access in the verifier; only output checks are exported"
    return None


def plan_meta(text: str) -> dict[str, str]:
    """The `# key: value` header that `dump_plan` writes above the YAML."""
    meta = {}
    for line in text.splitlines():
        if not line.startswith("# "):
            break
        key, _, value = line[2:].partition(": ")
        meta[key.strip()] = value.strip()
    return meta


def cli_version(meta: dict, run_dir: Path) -> str | None:
    """x.y.z of the CLI the plan was made against, from its inventory."""
    for path in (meta.get("inventory"), run_dir / "inventory.json"):
        if path and Path(path).exists():
            m = _VERSION.search(json.loads(Path(path).read_text()).get("cli_version", ""))
            return m.group(0) if m else None
    return None


def dead_seeds(plan: dict, run_id: str, manifest: Manifest) -> list[str]:
    """Seeded objects the tasks rely on that are no longer in the workspace (or never were)."""
    latest = {}
    for e in manifest.entries:  # later entries win
        latest[e["name"]] = e["status"]
    names = [f"{run_id}-{s['name'][len(RUN_PREFIX):]}" for s in plan.get("seed", [])]
    return [f"{n} ({latest.get(n, 'not in manifest')})" for n in names if latest.get(n) != CREATED]


def export_run(runs_dir: Path, run_id: str, out_dir: Path, task_ids: list[str] | None = None) -> dict:
    rdir = runs_dir / run_id
    plan_path = rdir / "plan.yaml"
    if not plan_path.exists():
        raise ExportError(f"no plan at {plan_path}")
    text = plan_path.read_text()
    plan, meta = load_plan(text), plan_meta(text)
    if dead := dead_seeds(plan, run_id, Manifest(rdir / "manifest.json")):
        raise ExportError("the seed isn't live, so the tasks' answers aren't in the workspace: " + ", ".join(dead)
                          + ". Run the plan again and choose `keep` at confirm reset, then export.")
    if task_ids and (unknown := sorted(set(task_ids) - {t["id"] for t in plan["tasks"]})):
        raise ExportError(f"unknown task IDs: {', '.join(unknown)}")

    seeded = {s["name"] for s in plan.get("seed", [])}
    version = cli_version(meta, rdir)
    exported, skipped = [], []
    for task in plan["tasks"]:
        if task_ids and task["id"] not in task_ids:
            continue
        if reason := skip_reason(task):
            skipped.append((task["id"], reason))
            continue
        exported.append(write_task(out_dir / f"{run_id}-{task['id']}", task, run_id, seeded, version))
    return {"exported": exported, "skipped": skipped, "cli_version": version}


def write_task(tdir: Path, task: dict, run_id: str, seeded: set[str], version: str | None) -> Path:
    if tdir.exists():
        shutil.rmtree(tdir)
    prompt = bind(task["prompt"], run_id, seeded)
    checks = [{"describe": c["describe"], "method": c["method"], "values": [bind(v, run_id, seeded) for v in c["values"]]}
              for c in task["hard_checks"]]
    files = {
        "task.toml": task_toml(task, run_id, version),
        "instruction.md": INSTRUCTION.format(prompt=prompt.strip(), answer=ANSWER_PATH),
        "environment/Dockerfile": agent_dockerfile(version),
        "environment/langsmith-wrapper.sh": WRAPPER,
        "environment/docker-compose.yaml": COMPOSE,
        "environment/lsproxy/Dockerfile": PROXY_DOCKERFILE,
        "environment/lsproxy/proxy.py": (HERE / "proxy.py").read_text(),
        "tests/Dockerfile": VERIFIER_DOCKERFILE,
        "tests/test.sh": TEST_SH,
        "tests/check.py": (HERE / "check.py").read_text(),
        "tests/checks.json": json.dumps(checks, indent=2) + "\n",
        "solution/solve.sh": solve_sh(checks),
    }
    for rel, content in files.items():
        path = tdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if rel.endswith(".sh"):
            path.chmod(0o755)
    return tdir


# --- file templates ---------------------------------------------------------------

INSTRUCTION = """{prompt}

---

This is the user's LangSmith workspace. The `langsmith` command-line tool is installed and already authenticated; use it to do this. Some API calls are blocked by the test sandbox, and the error says so when that happens.

When you're done, write your answer for the user to `{answer}`. That file is all they will see, so put the answer in it, not a description of what you ran. If you couldn't do something, say so plainly and why.
"""


def _toml(value) -> str:
    """A TOML value. JSON strings and arrays of strings are valid TOML."""
    return json.dumps(value)


def task_toml(task: dict, run_id: str, version: str | None) -> str:
    meta = {"run_id": run_id, "task_id": task["id"], "tier": task["tier"], "expected": task["expected"],
            "cli_version": version or "unknown", "docs_refs": task.get("docs_refs", [])}
    name = f"{ORG}/{run_id}-{task['id']}"
    lines = [
        "# Generated by `python -m tester harbor-export`. Regenerate rather than edit.",
        'schema_version = "1.4"',
        "# Copied into the verifier's container at the same paths.",
        f'artifacts = [{{ source = {_toml(ANSWER_PATH)} }}, {{ source = {_toml(PROXY_LOG)}, service = "lsproxy" }}]',
        "",
        "[task]",
        f"name = {_toml(name)}",
        f"description = {_toml(task.get('rationale', ''))}",
        f"keywords = {_toml(['langsmith-cli', task['tier']])}",
        "",
        "[metadata]",
        *(f"{k} = {_toml(v)}" for k, v in meta.items()),
        "",
        "[agent]",
        "timeout_sec = 900.0",
        "",
        "[verifier]",
        "timeout_sec = 60.0",
        'environment_mode = "separate"',
        'network_mode = "no-network"',
        "",
        "[environment]",
        "# The agent needs its model API; the workspace is reachable only through lsproxy.",
        'network_mode = "public"',
        "build_timeout_sec = 600.0",
        "",
    ]
    return "\n".join(lines)


def agent_dockerfile(version: str | None) -> str:
    check = (f'RUN langsmith --version | grep -qF "{version}" '
             f'|| (echo "expected langsmith CLI {version}, got: $(langsmith --version)" >&2; exit 1)\n'
             if version else "# No CLI version recorded for this plan, so it isn't pinned.\n")
    return f"""# The agent's container. Generated by `python -m tester harbor-export`.
FROM ubuntu:24.04

# jq is here on purpose: real agents pipe JSON through it, and our run_cli worker can't.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl jq \\
    && rm -rf /var/lib/apt/lists/*

# The install script's target directory isn't documented, so find the binary and move it off
# PATH. The wrapper below is the only `langsmith` the agent can find.
RUN curl -fsSL https://cli.langsmith.com/install.sh | sh \\
    && bin="$(command -v langsmith || find / -xdev -type f -name langsmith -perm -u+x 2>/dev/null | head -n1)" \\
    && install -D -m 0755 "$bin" {REAL_CLI} \\
    && rm -f "$bin"
# What the agent runs as `langsmith`: always through lsproxy, with a placeholder key.
COPY langsmith-wrapper.sh /usr/local/bin/langsmith
RUN chmod 0755 /usr/local/bin/langsmith
# Findings are version-specific: fail the build if the CLI moved since the plan was made.
{check}
WORKDIR /app
"""


REAL_CLI = "/opt/langsmith-cli/langsmith"

# Safety rule 4 (minimal env) for a container we don't control: some Harbor agents forward
# the host's LANGSMITH_* (tracing) into the container. Whatever arrives, the CLI only sees the
# proxy and the placeholder key, and LANGSMITH_PROJECT can't change what a command lists.
WRAPPER = f"""#!/bin/sh
# Generated by `python -m tester harbor-export`. Installed as `langsmith`; runs the real CLI.
# Drop every LANGSMITH_* the agent harness may have forwarded, then point the CLI at lsproxy,
# which swaps the placeholder for the target key and forwards reads only.
for v in $(env | sed -n 's/^\\(LANGSMITH_[A-Za-z0-9_]*\\)=.*/\\1/p'); do unset "$v"; done
export LANGSMITH_ENDPOINT=http://lsproxy:8080 LANGSMITH_API_KEY=eval-sandbox-placeholder
exec {REAL_CLI} "$@"
"""

COMPOSE = """# Generated by `python -m tester harbor-export`.
# Harbor merges this with its own compose file. The target key comes from the shell
# that runs `harbor run`, and only the lsproxy service sees it.
services:
  main:
    depends_on:
      lsproxy:
        condition: service_healthy
  lsproxy:
    build:
      context: ./lsproxy
    environment:
      LSPROXY_API_KEY: ${TARGET_API_KEY:?export TARGET_API_KEY (the sandbox workspace key) before harbor run}
      LSPROXY_UPSTREAM: ${TARGET_ENDPOINT:-https://api.smith.langchain.com}
      LSPROXY_WORKSPACE_ID: ${TARGET_WORKSPACE_ID:-}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/__lsproxy/health')"]
      interval: 2s
      timeout: 3s
      retries: 15
"""

PROXY_DOCKERFILE = """FROM python:3.12-slim
COPY proxy.py /proxy.py
EXPOSE 8080
CMD ["python", "-u", "/proxy.py"]
"""

VERIFIER_DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY . /tests/
"""

TEST_SH = """#!/bin/sh
# Grades the answer Harbor copied in from the agent's container. See check.py.
exec python3 /tests/check.py
"""


def solve_sh(checks: list[dict]) -> str:
    """For `harbor run -a oracle`: proves the build, the proxy, the key, and the verifier work end to end.
    It writes the expected values straight from the checks, so it says nothing about how hard the task is."""
    values = []
    for c in checks:
        if c["method"] == "output_contains_all":
            values += c["values"]
        elif c["method"] == "output_contains_any":
            values.append(c["values"][0])
    body = "\n".join(f"- {v}" for v in values)
    return f"""#!/bin/sh
# Plumbing check, not a reference solution: a real read through lsproxy, then an answer
# built from the checks' expected values. Generated by `python -m tester harbor-export`.
set -e
langsmith datasets list --limit 1 > /dev/null
cat > {ANSWER_PATH} <<'ANSWER_EOF'
{body}
ANSWER_EOF
"""
