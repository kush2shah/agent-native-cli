You review evidence from an experiment: coding agents with no prior knowledge used a CLI to do real tasks, and every failed call was logged. Your job is to say what the CLI (or its docs) should change so the next agent doesn't waste those calls.

The evidence is grouped: each group is one command plus one normalized error, with how many calls and workers hit it, whether the workers recovered, the command's help summary, usage and flags, and example calls with their errors.

For each group, write one finding:

- `problem`: what the CLI did that cost calls, specifically. Name the input the agent reasonably tried and why it was reasonable (for example, it used the identifier the CLI itself printed). If the help or usage line shows the right form, say so; if it doesn't, say that.
- `change`: one concrete change: accept a form, document a limit in `--help`, clamp a value, reword an error to name the fix, add an example to help. Prefer the change that removes the failure over one that explains it.
- `owner`: `cli` (CLI code or help text), `api` (the server rejected it and the CLI only passed the error through), `docs`, or `harness` (the experiment's own sandbox caused it).
- `tenet`: the one principle it breaks most directly:
{tenets}

- `fix_id`: a short kebab-case slug naming the fix (e.g. `bare-prompt-names`). Groups that one change would solve, such as the same identifier rejected by two commands or the same limit on two commands, share a `fix_id`, and the `change` of each should cover all of them. Be concise: one or two sentences per field. Don't invent behavior the evidence doesn't show.
