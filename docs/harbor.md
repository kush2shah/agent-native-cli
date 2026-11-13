# Harbor: running our tasks with real coding agents

Status: experiment. Nothing here has run against a real workspace yet (see [Not verified](#not-verified)).

## Why

[Harbor](https://github.com/harbor-framework/harbor) is the eval framework from the Terminal-Bench team. A task is a directory: an instruction, a container to work in, and a verifier that writes a reward. Harbor runs any of dozens of built-in agents (Claude Code, Codex CLI, OpenHands, Gemini CLI, ...) against it with any model, in parallel, locally or in cloud sandboxes, and keeps each trial's trajectory.

Our tester already does the hard part: it learns the CLI, plans tasks that defeat defaults and plant decoys, seeds the data, and reports findings. What it doesn't do is use a real agent. The worker is a hand-built ReAct loop with one tool, `run_cli`, no shell, and the root `--help` pasted into its prompt. Some findings may come from that setup and not from the CLI:

- **Oversized output** (`workspaces list` returned 184 KB): an agent with a shell pipes through `jq` or `head` and may never feel it.
- **The docs arm never used the docs:** maybe `--help` really is enough, or maybe that's how our worker behaves.
- **Acting on destructive requests without asking:** that's our worker's judgment, not necessarily Claude Code's or Codex's.

Exporting the same tasks to Harbor answers that directly. If Claude Code and Codex hit the same friction, the finding is about the CLI. If they don't, it was our harness. It also brings agent and model comparisons nearly for free (a non-goal in SPEC.md today, because building them ourselves would be expensive).

## How the pieces map

| Tester | Harbor |
|---|---|
| plan task (prompt, checks) | task dir: `instruction.md`, `tests/` |
| worker run (task × arm × repeat) | trial (task × agent × model × attempt, `-k`) |
| `help` arm | default: the agent finds `--help` on its own |
| `docs` arm | `[[environment.mcp_servers]]`, but see Safety |
| `skill` arm (later) | `[environment] skills_dir` |
| hard checks → `passed` | verifier → `reward.json` (any number of numeric keys) |
| transcript, `commands.jsonl` | Harbor trajectory + the proxy's request log |
| `MAX_CONCURRENCY`, repeats | `-n`, `-k` |
| intake, discover, propose, seed, reset, report | stay ours |

Harbor replaces the runner, not the tester. Planning, seeding and reporting are the experiment; Harbor only runs agents.

## What this adds

`python -m tester harbor-export RUN_ID [--tasks t01,t03] [--out DIR]` writes one Harbor task per eligible plan task, by default to `runs/<run_id>/harbor/`.

```
<run_id>-<task_id>/
  task.toml               # verifier in its own container with no network; artifacts to carry over
  instruction.md          # the bound prompt + "write your answer to /app/answer.md"
  environment/
    Dockerfile            # ubuntu + jq + the LangSmith CLI (build fails if the version drifted)
    langsmith-wrapper.sh  # the only `langsmith` on PATH: forces the proxy and a placeholder key
    docker-compose.yaml   # adds the lsproxy sidecar
    lsproxy/              # tester/harbor/proxy.py
  tests/                  # verifier image: tester/harbor/check.py + checks.json
  solution/solve.sh       # plumbing check for `-a oracle`
```

Then:

```bash
export TARGET_API_KEY=...        # plus TARGET_ENDPOINT / TARGET_WORKSPACE_ID if you use them
H="env -u LANGSMITH_API_KEY -u LANGSMITH_ENDPOINT -u LANGSMITH_WORKSPACE_ID harbor run -p runs/<run_id>/harbor"
$H -a oracle
$H -a claude-code -m anthropic/claude-sonnet-5-5 -k 3 -n 3
```

Why the `env -u`: some Harbor agents (`langgraph`, `deerflow`) forward the host's `LANGSMITH_*` variables into the agent's container for tracing. That would hand the agent a real key (the harness key, if it's in your shell) and an endpoint that bypasses the proxy. For the same reason, don't use `harbor run --env-file .env`.

**Which tasks export.** Only read-only `complete` tasks whose checks are all `output_*`. Traps, writes, and `object_*` checks are skipped with a reason. The seed has to be live: run the plan, choose `keep` at confirm reset, export, run Harbor, then `python -m tester reset <run_id>`. The exporter refuses if any seeded object isn't `created` in the manifest.

**The answer.** Our worker's final message is the answer. Harbor agents don't have one we can read reliably, so the instruction asks for `/app/answer.md`. Harbor copies it into the verifier's container, and `check.py` grades it with the same case-insensitive matching as `run_checks`.

**The reward.** `reward.json` holds `reward` (1 only if answered and every check passed), `answered`, `check_<n>` per check, and `api_requests` / `api_denied` from the proxy log. `checks.json` next to it has the readable version.

**The oracle.** `solve.sh` makes one real read through the proxy, then writes an answer built from the checks' expected values. If it passes, the image builds, the CLI installs, the key works through the proxy, artifacts reach the verifier, and the verifier grades. It says nothing about task difficulty.

## Safety

The tester's rules assume the agent can only call `run_cli`. In Harbor the agent has a root shell in its container. So the rules move from the CLI layer to the API layer:

- **The key never enters the agent's container.** The CLI points at `http://lsproxy:8080` with a placeholder key. The sidecar gets `TARGET_API_KEY` from the host's environment through compose interpolation. Only that service sees it, and the agent has no Docker socket to reach into it.
- **The CLI can't be pointed elsewhere by the environment.** `langsmith` on PATH is a wrapper that drops every `LANGSMITH_*` variable, sets the proxy endpoint and placeholder key, and runs the real binary from `/opt`. So a forwarded `LANGSMITH_ENDPOINT` or `LANGSMITH_PROJECT` can't bypass the proxy or change what a command lists. The `env -u` above still matters: the wrapper protects the CLI, not a key sitting in the container's environment.
- **Reads only, enforced by the proxy.** GET/HEAD pass. POST passes only to `…/query`, `…/stats`, `…/search`, `…/count`, which LangSmith uses for filtered listings. Everything else gets a 403 whose detail says the eval sandbox blocked it, so it can't be confused with a LangSmith permission error. Unknown read endpoints fail closed and show up as `denied` lines in the log.
- **No key leaks:** the proxy drops the client's `x-api-key`, `Authorization` and `X-Tenant-Id`, sets its own, refuses absolute-URL request targets, and never follows redirects.
- **One workspace:** the proxy sets `X-Tenant-Id` when `TARGET_WORKSPACE_ID` is set, overriding anything the agent sends.

How SPEC.md's hard rules carry over:

| Rule | In Harbor |
|---|---|
| 1. One workspace | Proxy pins the key and workspace |
| 2. Manifest-only deletes | Nothing can be created or deleted; only reads are exported |
| 3. Allowlist execution | Replaced by the proxy's API allowlist. The CLI-level denylist no longer applies: `api` (raw HTTP) is just more reads, and `share` is a POST, so it's denied. `self-update`, `completion`, and `traces setup` only change a throwaway container |
| 4. Minimal env | The wrapper gives the CLI exactly the proxy endpoint and the placeholder key |
| 5. Separate harness tracing | Run Harbor with `LANGSMITH_*` unset (see above), so nothing traces anywhere |
| 6. Budget stops | **Gap.** Harbor has agent timeouts (900 s here) but no CLI-call cap or cost ceiling. Start with one task and `-k 1` |
| 7. Separate harness executor | Seeding and reset stay in the tester |

**Docs arm, not exported.** Harbor can attach the docs MCP, but it exposes every tool on the server, including `submit_feedback`, which writes to the docs team. Our `docs` arm deliberately exposes only the two read tools. Exporting it needs a filtering MCP proxy, the same pattern as lsproxy.

## Not verified

This was written in a sandbox with no access to Docker Hub or GitHub releases, so no image was built and nothing ran against LangSmith. What was checked:

- unit tests for selection, binding, file generation, the proxy (policy, credential swap, write blocking, redirects, absolute targets, unreachable upstream), and the verifier
- `harbor lint` (0.25.0) passes the default rules on an exported task (with `TARGET_API_KEY` set to anything; compose refuses to load without it, on purpose)
- `docker compose config` on Harbor's base file plus ours resolves the sidecar's build context and the `service_healthy` dependency

Open questions the first real run will answer:

- where `install.sh` puts the binary (the Dockerfile searches for it) and whether it can pin a version (today the build fails if the version drifted)
- whether the CLI accepts an `http://` endpoint and still sends the key on every request
- which POST endpoints the CLI uses for reads (the proxy log lists every denial)
- whether `X-Tenant-Id` is the header the API uses for workspace selection (it's what the Python SDK sends)
- whether sidecars work on cloud environments (`--env daytona`/`modal`). Start with local Docker.

`harbor lint --select ALL` also flags unpinned base images and a root agent user. Both are fine for a first run.

## Next steps

1. On a kept run, export, run the oracle, then `claude-code` and `codex` with `-k 3` on the exported tasks.
2. Compare pass rates with our `help` arm on the same tasks. Where they disagree, that's the finding.
3. If it's worth keeping: read Harbor job dirs into `report.compute()` (reward keys plus CLI calls parsed from the trajectories), then add writes (the proxy would need the task's contract and the manifest) and traps (the judge reads the trajectory).
