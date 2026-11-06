import stat
import sys
from pathlib import Path

import pytest

from tester.config import load_config
from tester.inventory import Inventory, discover

FAKE = Path(__file__).parent / "fake_cli" / "fake_langsmith.py"
TARGET_KEY = "target-key-123"
HARNESS_KEY = "harness-key-456"


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """An executable named `langsmith` that runs the stub with this interpreter."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "langsmith"
    exe.write_text(
        f"#!{sys.executable}\nimport runpy, sys\nsys.argv[0] = 'langsmith'\n"
        f"runpy.run_path({str(FAKE)!r}, run_name='__main__')\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


@pytest.fixture
def harness_env(monkeypatch):
    """Harness variables that must never reach the CLI subprocess."""
    monkeypatch.setenv("LANGSMITH_API_KEY", HARNESS_KEY)
    monkeypatch.setenv("LANGSMITH_PROJECT", "cli-tester-harness")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("SOME_HARNESS_SECRET", "nope")


@pytest.fixture
def cfg(fake_bin, tmp_path, monkeypatch, harness_env):
    for var in ("TARGET_ENDPOINT", "TARGET_WORKSPACE_ID", "MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TARGET_CLI_BIN", str(fake_bin))
    monkeypatch.setenv("TARGET_API_KEY", TARGET_KEY)
    monkeypatch.setenv("CLI_TIMEOUT_S", "3")
    return load_config(env_file=None, runs_dir=tmp_path / "runs")


@pytest.fixture
def no_overrides(tmp_path) -> Path:
    return tmp_path / "none.yaml"


@pytest.fixture
def inventory_data(cfg, no_overrides) -> dict:
    return discover(cfg, "test", use_llm=False, overrides_path=no_overrides)


@pytest.fixture
def inventory(inventory_data) -> Inventory:
    return Inventory(inventory_data)
