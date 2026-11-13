"""Harbor verifier for exported tasks: grades the agent's answer with the plan's output checks.

Runs in its own container after the agent's is stopped. Harbor copies the agent's
answer and the proxy's request log in as artifacts, at their original paths.

Writes `/logs/verifier/reward.json`. Every value must be a number:
- `reward`: 1 only if there is an answer and every check passed (same rule as the native runner)
- `answered`: whether `/app/answer.md` exists and isn't empty
- `check_<n>`: 1/0 per check, in plan order
- `api_requests`, `api_denied`: from the proxy log, when it was collected

`/logs/verifier/checks.json` keeps the readable version (describe, values, passed).

Check semantics mirror `run_checks` in tester/graph.py: case-insensitive substring
matches on the final answer. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def grade(checks: list[dict], answer: str) -> list[dict]:
    low = answer.lower()
    out = []
    for c in checks:
        values = [v.lower() for v in c["values"]]
        if c["method"] == "output_contains_all":
            ok = all(v in low for v in values)
        elif c["method"] == "output_contains_any":
            ok = any(v in low for v in values)
        elif c["method"] == "output_excludes":
            ok = not any(v in low for v in values)
        else:
            raise ValueError(f"unsupported check method in an exported task: {c['method']}")
        out.append({**c, "passed": ok})
    return out


def api_counts(log: Path) -> dict[str, int]:
    if not log.exists():
        return {}
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return {"api_requests": len(lines), "api_denied": sum(1 for r in lines if not r.get("allowed"))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checks", default="/tests/checks.json")
    p.add_argument("--answer", default="/app/answer.md")
    p.add_argument("--proxy-log", default="/var/log/lsproxy/requests.jsonl")
    p.add_argument("--out-dir", default="/logs/verifier")
    args = p.parse_args(argv)

    checks = json.loads(Path(args.checks).read_text())
    answer_path = Path(args.answer)
    answer = answer_path.read_text() if answer_path.exists() else ""
    answered = bool(answer.strip())
    results = grade(checks, answer)

    rewards: dict[str, int] = {"reward": int(answered and all(r["passed"] for r in results)), "answered": int(answered)}
    rewards.update({f"check_{i}": int(r["passed"]) for i, r in enumerate(results, 1)})
    rewards.update(api_counts(Path(args.proxy_log)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps({"answered": answered, "checks": results}, indent=2))
    (out / "reward.json").write_text(json.dumps(rewards))

    print("answer: " + ("present" if answered else f"MISSING ({answer_path})"))
    for i, r in enumerate(results, 1):
        print(f"  check_{i} {'PASS' if r['passed'] else 'FAIL'}  {r['describe']}  [{r['method']}: {r['values']}]")
    print(json.dumps(rewards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
