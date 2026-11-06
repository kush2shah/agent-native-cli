"""Configuration: load the dotenv file, keep target credentials out of the harness env.

Target values (TARGET_*) are read into `Config.target_env`, already mapped to the
names the CLI expects, and are never written to `os.environ`. Everything else in
the file (harness tracing, model keys) is exported to `os.environ` so LangChain
picks it up.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# TARGET_* name in the dotenv file -> env var name the LangSmith CLI reads.
TARGET_ENV_MAP = {
    "TARGET_API_KEY": "LANGSMITH_API_KEY",
    "TARGET_ENDPOINT": "LANGSMITH_ENDPOINT",
    "TARGET_WORKSPACE_ID": "LANGSMITH_WORKSPACE_ID",
}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    cli_bin: Path  # absolute path to the target CLI
    target_env: dict[str, str]  # credentials, already mapped to the CLI's names
    docs_path: Path | None
    model: str | None
    max_cli_calls_per_task: int
    cost_ceiling_usd: float
    repeats: int
    max_concurrency: int
    cli_timeout_s: float
    runs_dir: Path
    warnings: tuple[str, ...] = ()

    @property
    def has_target_key(self) -> bool:
        return bool(self.target_env.get("LANGSMITH_API_KEY"))


def load_config(env_file: Path | None = DEFAULT_ENV_FILE, runs_dir: Path | None = None) -> Config:
    values = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file and env_file.exists() else {}

    for key, value in values.items():
        if not key.startswith("TARGET_"):
            os.environ.setdefault(key, value)

    def get(key: str, default: str | None = None) -> str | None:
        # The dotenv file wins; the shell env is a fallback (handy for tests/CI).
        return values.get(key) or os.environ.get(key) or default

    bin_name = get("TARGET_CLI_BIN", "langsmith")
    resolved = shutil.which(bin_name)
    if not resolved:
        raise ConfigError(f"TARGET_CLI_BIN={bin_name!r} not found on PATH")

    target_env = {cli_name: v for src, cli_name in TARGET_ENV_MAP.items() if (v := get(src))}

    warnings: list[str] = []
    harness_key = os.environ.get("LANGSMITH_API_KEY")
    if harness_key and harness_key == target_env.get("LANGSMITH_API_KEY"):
        warnings.append(
            "TARGET_API_KEY equals the harness LANGSMITH_API_KEY (safety rule 5 relaxed). "
            "If tracing is on, harness traces land in the workspace the agent tests, so it isn't empty."
        )

    docs = get("TARGET_DOCS_PATH")
    return Config(
        cli_bin=Path(resolved).resolve(),
        target_env=target_env,
        docs_path=(REPO_ROOT / docs).resolve() if docs else None,
        model=get("MODEL"),
        max_cli_calls_per_task=int(get("MAX_CLI_CALLS_PER_TASK", "15")),
        cost_ceiling_usd=float(get("COST_CEILING_USD", "5")),
        repeats=int(get("REPEATS", "3")),
        max_concurrency=int(get("MAX_CONCURRENCY", "3")),
        cli_timeout_s=float(get("CLI_TIMEOUT_S", "60")),
        runs_dir=runs_dir or REPO_ROOT / "runs",
        warnings=tuple(warnings),
    )


def build_subprocess_env(cfg: Config, home: Path) -> dict[str, str]:
    """The complete env for a CLI subprocess. Nothing is inherited from the harness."""
    path_dirs = dict.fromkeys([str(cfg.cli_bin.parent), "/usr/bin", "/bin"])
    return {"PATH": os.pathsep.join(path_dirs), "HOME": str(home), **cfg.target_env}
