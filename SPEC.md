# CLI tester prototype — spec

An experiment: an agent that learns a CLI on its own, proposes realistic flows to test, runs them against a real workspace, and reports where the CLI is hard for agents to use. First target: the LangSmith CLI (`langsmith`, v0.3.x).

This is a prototype, not a product. Favor simple and readable over general. Build one milestone at a time and stop after each for review.

## Goals

- Find out where the LangSmith CLI is hard for an agent to use: invented flags and subcommands, excessive `--help` lookups, unparseable or oversized output, unclear errors.
- Make blind spots (commands the key can't run) loud, never silent.
- Learn LangGraph by building it: hand-built ReAct loop (cycles), Send API (map-reduce), parallel branches, interrupts with human edits.

## Non-goals (for now)

- Supporting any CLI other than the one configured
- A UI (terminal only)
- LangSmith Engine integration
- Variant comparisons (models, docs on/off)
- Handling CLI version changes

## Hard safety rules

These are enforced in code, never left to the model's judgment.

1. **One target workspace.** The prototype operates only in a dedicated, initially empty workspace, using a key scoped to that workspace.
2. **Manifest-only deletes.** Every object the tester creates gets the run ID as a name prefix plus a metadata tag where supported. It is recorded in `runs/<run_id>/manifest.json` before creation and confirmed after. Reset deletes only manifest entries. Never delete by search, name pattern, or "delete all".
3. **Allowlist execution.** The agent's tool, `run_cli`, enforces all of the following:
   - It executes only the configured CLI binary (resolved to an absolute path), via `subprocess` with an argument list. No shell, pipes, redirects, or other programs.
   - Each command is resolved to a command path and checked against the inventory classification: `read` runs, `write` runs only if the task's approved contract lists that path, and `destructive` or `unknown` is denied and logged. Unclassified means denied.
   - **Argument checks, independent of classification:**
     - Target-switching global flags are denied: `--workspace`, `--profile`, `--api-url`, `--api-key` (hidden; works but isn't in any help), and `--base-url` (generated commands). `--debug` is denied too, since it can log request headers.
     - File references are denied: any argument starting with `@` or containing `=@` (`--body @file`, `-F key=@path`).
     - Paths that escape the sandbox are denied: absolute paths, `~`, and `..` segments.
   - **Hard denylist** (rules in `tester/classify.py`; nothing overrides it, including human overrides):
     - Raw HTTP: `api`, an authenticated escape hatch that would bypass classification.
     - Credentials and saved defaults: `auth`, `profiles`, and any `select` / `login` / `logout` / `set-workspace` subcommand.
     - Long-running or interactive: `connector`, `relay`, `tunnel`, `sandboxes`.
     - Changes the local environment: `self-update`, `completion`, `traces setup`.
     - Outward-facing: `report`, `product-feedback`, and any `share` subcommand (makes data public).
   - `--help` is always allowed (it is the agent's main discovery tool and has no side effects). On a hard-denied command only the bare `<path> --help` form is allowed.
   - An unknown subcommand under a group is denied, not passed through. The inventory may be incomplete, and "anything under this group runs" is not a safe fallback. The real CLI would print help and exit 0; graders count these denials as invented subcommands.
   - Every call runs with `cwd` set to a temp directory, `stdin` closed, and a timeout (default 60s; the whole process group is killed on expiry).
4. **Minimal subprocess env.** The CLI subprocess gets an explicitly built env: `PATH` (CLI dir + `/usr/bin:/bin`), a temp `HOME`, and the target credentials only. It never inherits the harness environment. This matters doubly here because the CLI reads the same variable names the harness uses for tracing (`LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_WORKSPACE_ID`). Target values are never written to the harness's `os.environ`.
5. **Separate harness tracing.** Harness tracing credentials should be distinct from target credentials, with harness traces in a separate project and workspace. Relaxed for now: the same key is allowed with a loud warning. Revisit before M3, since harness traces in the target workspace break "initially empty" and are visible to the agent.
6. **Budget stops.** Max CLI calls per task (default 15, enforced in `run_cli`), a recursion limit as a backstop, and a cost ceiling (default $5) enforced by a thread-safe process-level counter, not graph state, so parallel workers can't all overshoot before a state merge.
7. **Separate harness executor.** Seeding and reset need commands the agent must never get (`projects delete`, `runs create`). They run through a harness-side executor that only acts on the plan's seed steps and manifest entries. The worker's `run_cli` never has those permissions.

## Configuration

`.env` (gitignored). See `env.example`.

```
# Target: what the agent operates on
TARGET_CLI_BIN=langsmith
TARGET_API_KEY=...                 # workspace-scoped key for the sandbox workspace
TARGET_ENDPOINT=                   # optional; self-hosted only
TARGET_WORKSPACE_ID=               # optional; needed if the key spans workspaces
TARGET_DOCS_PATH=./target_docs/README.md   # README / AGENTS file the agent may read

# Harness: tracing for this project (separate workspace/project)
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=cli-tester-harness
LANGSMITH_TRACING=true

# Model
MODEL=anthropic:claude-sonnet-5-5
ANTHROPIC_API_KEY=...

# Budgets
MAX_CLI_CALLS_PER_TASK=15
COST_CEILING_USD=5
REPEATS=3
MAX_CONCURRENCY=3
CLI_TIMEOUT_S=60
```

The harness maps `TARGET_*` to the CLI's names only inside the subprocess env:

| `.env` | CLI reads |
|---|---|
| `TARGET_API_KEY` | `LANGSMITH_API_KEY` |
| `TARGET_ENDPOINT` | `LANGSMITH_ENDPOINT` |
| `TARGET_WORKSPACE_ID` | `LANGSMITH_WORKSPACE_ID` |

Claude Code must not read `.env`. `.claude/settings.json` denies `Read(./.env)`, `Read(./.env.*)`, and Bash commands mentioning `.env`. A Read deny alone doesn't stop `cat .env`, and Bash pattern denies are best-effort, so keep real secrets out of anything Claude is asked to look at.

## Architecture

A LangGraph `StateGraph` with a SQLite checkpointer, so paused runs survive restarts. Human checkpoints use `interrupt()`. The terminal driver resumes them with `Command(resume=...)`.

```
intake (interrupt) → discover → propose → review (interrupt) ─edits─┐
                                   ▲                              │
                                   └──────────────────────────────┘
review ─approve→ seed → [Send × tasks×repeats] → worker → grade (parallel) → report (interrupt)
report ─dispute→ regrade → report        (cut first if M4 runs long)
report ─approve reset→ reset → END
```

Workers are a subgraph: `agent ⇄ run_cli` (hand-built ReAct; no prebuilt agent or `ToolNode`). There are no interrupts inside workers. Writes outside the approved contract are auto-denied and logged.

**What the worker sees:** the task prompt, the docs file, and the CLI's root `--help`. Never the inventory. That gap is the experiment.

### Phases

| Phase | Does | Output |
|---|---|---|
| intake | Asks 3–4 terminal questions: flows that matter, what "good" means, risk tolerance | `runs/<id>/intake.json` |
| discover | Crawls `--help` recursively, classifies commands, probes permissions with cheap read calls | `runs/<id>/inventory.json` |
| propose | Generator prompt drafts a seed plan, 5 tasks with contracts, and a reset plan | `plan.yaml` |
| review | Shows the plan; user approves, edits, or asks for changes (loops to propose) | approved `plan.yaml` |
| seed | Harness executor creates seed objects, writing the manifest ahead of each create; waits until async-ingested objects (runs) are visible | `manifest.json` |
| run | One Send per task × repeat; each runs the worker subgraph | per-run results in state |
| grade | Four graders as parallel branches | scores in state |
| report | Writes markdown, blind spots first; user can dispute a verdict | `report.md` |
| reset | Harness executor deletes manifest entries in reverse order; verifies they're gone | updated manifest |

### Classification

Precedence, highest first:

1. **Hard denylist** (code): always `destructive`.
2. **Human overrides** in `classifications.yaml` (committed, reviewed).
3. **Verb rules** on the last path segment (`list`/`get` → read, `create`/`update` → write, `delete`/`archive` → destructive, ...). Hyphenated verbs match on any part and the most dangerous part wins (`bulk-delete` → destructive).
4. Otherwise `unknown` (denied).

Non-runnable group commands (they can only print help) are `read`.

An optional LLM pass suggests classes for `unknown` commands. Suggestions are written to `runs/<id>/suggested_classifications.yaml` and **never applied automatically**. A human merges what they agree with into `classifications.yaml`. That keeps rule 3 out of the model's judgment.

### Writes by workers

The model picks names, so `run_cli` can only record a create in the manifest beforehand if it knows which argument is the name. For the prototype:

- Only create commands with a hand-written name-argument mapping (in code) may appear in `allowed_writes`, starting with `datasets create`.
- `run_cli` enforces the `<run_id>-<task>-r<n>-` prefix on that argument, writes the `intended` manifest entry, runs, and parses the ID from JSON output.
- Every worker run takes a listing snapshot before and after, so `no_objects_outside_manifest` has something to compare against.

### Concurrency

Repeats of the same task run in parallel against one workspace. Every object name includes `<run_id>-<task>-r<n>`, and hard checks are scoped to their own prefix so one worker's writes can't pass or fail another's check.

### State (sketch)

```python
class RunState(TypedDict):
    run_id: str
    intake: dict
    inventory: dict
    plan: dict                     # seed plan, tasks, contracts, reset plan
    approved: bool
    results: Annotated[list[dict], operator.add]   # one per worker run
    grades: Annotated[list[dict], operator.add]    # one per grader per run
    blind_spots: Annotated[list[dict], operator.add]
    disputes: list[dict]
    cost_usd: Annotated[float, operator.add]       # reporting only; enforcement is process-level
```

Worker input (via Send): `{task, contract, repeat, run_id}`.

## File formats

**inventory.json**

```json
{
  "cli_version": "x.y.z",
  "global_flags": [{"name": "format", "short": null, "type": "string"}],
  "commands": [
    {"path": "datasets list", "summary": "...", "help": "...",
     "runnable": true, "children": [], "required_args": [],
     "flags": [{"name": "limit", "short": "n", "type": "int"}],
     "class": "read", "class_source": "rule:verb",
     "suggested_class": null,
     "probe": "ok | permission_denied | error | not_probed", "probe_detail": ""}
  ]
}
```

Flags are recorded so invented flags can be detected statically from the command log, not only from stderr.

**plan.yaml: one task contract**

```yaml
- id: t03
  tier: medium            # easy | medium | hard | trap
  prompt: "Find the runs that errored in project <run_id>-proj-a and summarize the error types."
  allowed_writes: []      # command paths this task may run (must have a name-arg mapping)
  expected: complete      # complete | ask_or_refuse | blind_spot
  hard_checks:            # programmatic; query state after the run
    - describe: "summary mentions each seeded error type"
      method: output_contains_all
      values: ["TimeoutError", "ValueError"]
  behavior_checks: [no_writes_outside_contract, no_objects_outside_manifest]
  soft_metrics: [cli_calls, help_calls, invented_flags, invented_subcommands, used_json_format, output_tokens, error_recoveries]
```

**manifest.json:** a list of `{kind, name, id, status: intended|created|deleted, created_by_task}`. Deleting a tracing project removes its traces, so seeded runs are covered by their project's entry.

**commands.jsonl:** one line per `run_cli` call, allowed or denied: argv, resolved path, class, decision and reason, exit code, duration, output sizes and heads.

## Observed CLI behavior (from M1, v0.3.13)

These shape the parser and graders, and several are report findings already:

- **Two help formats.** Hand-written commands use cobra (`Usage:` / `Flags:`). Generated API-resource commands (e.g. `runs share create`) use urfave/cli (`USAGE:` / `OPTIONS:`), with different flag conventions (`--format auto|explore|json|jsonl|pretty|raw|yaml` vs `pretty|json`) and different unknown-flag errors (`flag provided but not defined:` vs `unknown flag:`).
- **Generated help shows a usage line you can't run:** `langsmith runs:share create`. The real invocation is `langsmith runs share create`, and `runs:share` errors with `unknown command`.
- **An unknown subcommand under a group prints help and exits 0**, so an agent can't tell from the exit code that it invented a command.
- **A wrong key or wrong region gets 403, not 401.** Discover warns when every probe is refused instead of reporting everything as a blind spot.
- **`--api-key` is a hidden flag.** The "not authenticated" error tells you to pass it, but no help output lists it.
- **`datasets list` help says "default 100"**, but the request it sends uses `limit=20`.
- 275 commands (221 runnable, 54 groups). Cobra pads command names to the longest plus one space, so a two-space parser silently misses `annotation-queues` and `online-evaluators`.

## Graders (run as parallel branches)

- **State check:** runs the contract's `hard_checks` against workspace state or the final output.
- **Behavior:** checks the worker's command log against `allowed_writes` and the manifest.
- **Efficiency:** computes soft metrics.
  - **Invented flag:** a flag not in the inventory for that path, confirmed by the CLI's `unknown flag` / `unknown shorthand flag`.
  - **Invented subcommand:** a positional at a group node that isn't a child. `run_cli` denies these (see rule 3) and the log records them.
  - **`used_json_format`:** whether the agent found `--format json`. The default is `pretty`.
- **Blind spots:** classifies each failed command. A permission error (401/403 or the CLI's equivalent) is a blind spot, not an agent failure.

## Report (`report.md`)

1. **Not tested:** blind spots, plus inventory commands never exercised, with reasons
2. **Scorecard:** per task, passes out of `REPEATS`, plus soft metrics
3. **Findings:** each failure pattern mapped to a concrete CLI change it suggests
4. **Caveat:** generator and tester share a model; flag if results look suspiciously clean

## Milestones

Each milestone ends with something runnable. Stop after each one for review.

**M1: Discover**
- `run_cli` with allowlist + argument checks + minimal env, help crawler, rule classifier with overrides, optional LLM suggestions, permission probe.
- Done when: `python -m tester discover` writes `inventory.json`, and every command is classified with an unknown → denied default.

**M2: Propose + review**
- intake, propose, and review nodes; the terminal interrupt loop; SQLite checkpointer.
- Done when: you can generate a plan, request edits, approve, kill the process mid-review, and resume.

**M3: Seed, run, reset (sequential)**
- Manifest writer, harness executor, seed node (with wait-for-visible), worker subgraph (hand ReAct), name-arg mapping for allowed creates, reset.
- Tasks run one at a time.
- Done when: a full run leaves the workspace exactly as it started, verified by listing.

**M4: Fan out + grade + report**
- Replace the sequential loop with Send; add parallel graders, the report, and dispute → regrade.
- Done when: `report.md` shows blind spots first and repeats ran concurrently (visible as shared step numbers in traces).

## Testing

- **Fake model:** a scripted model, like the `03_react_by_hand.py` pattern, so graph logic is tested offline.
- **Fake CLI:** a stub binary (`tests/fake_cli/`) that mimics cobra `--help` trees, permission errors, unknown-flag errors, and the exit-0 unknown-subcommand behavior. All safety rules must be tested against the stub before anything touches the real workspace.
- **Required safety tests:**
  - an unclassified command is denied
  - a disallowed write is denied and logged
  - hard-denied commands stay denied even with a human override
  - target-switching flags, `@file` references, and escaping paths are denied
  - the subprocess env is exactly `PATH`, `HOME`, and the target credentials; the CLI's `LANGSMITH_API_KEY` is the target key, never the harness key
  - a hanging command is killed at the timeout
  - reset ignores objects not in the manifest (M3)

## Repo layout

```
tester/
  __main__.py        # CLI entry: discover | plan | run | reset
  config.py          # .env loading, env mapping
  run_cli.py         # allowlisted subprocess tool + policy
  classify.py        # hard denylist, verb rules, overrides, LLM suggestions
  inventory.py       # help crawl, parse, probe
  graph.py           # top-level graph (M2)
  worker.py          # ReAct subgraph (M3)
  graders.py         # (M4)
  manifest.py        # (M3)
  report.py          # (M4)
  prompts/           # classifier.md, generator.md, tester.md
classifications.yaml # human-reviewed overrides
tests/
  fake_cli/          # stub binary + help tree
runs/                # gitignored; one folder per run
```

## Open questions

- Which commands support metadata tags (for the manifest's second marker)
- Which docs file to give the agent. Default: the CLI's root `--help` (already written for agents) plus the README.
- ~~Exact CLI binary name and auth env var names~~ `langsmith`; see the config table.
- ~~LLM vs rules classifier~~ Rules + human overrides decide; the LLM only suggests.
