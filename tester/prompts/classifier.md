You classify CLI commands for a safety allowlist. Your output is a suggestion that a human reviews; it is never applied automatically.

For each command below, return exactly one classification:

- `read`: only reads remote state or prints information. Writing a local output file (for example `--output file.json`) still counts as read.
- `write`: creates or modifies remote objects, or triggers remote work (runs, jobs, ingestion), without deleting anything.
- `destructive`: deletes or irreversibly changes remote objects, changes credentials or saved defaults, makes data public, sends data outside the workspace, or starts a long-running process.

When unsure between two classes, pick the more dangerous one. Give a one-sentence reason grounded in the help text. Use each command's path exactly as given in its heading.
