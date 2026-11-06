import os

import pytest

from tester.config import ConfigError, load_config

from .conftest import HARNESS_KEY, TARGET_KEY


def test_target_key_mapped_but_not_exported(cfg):
    assert cfg.target_env == {"LANGSMITH_API_KEY": TARGET_KEY}
    assert os.environ["LANGSMITH_API_KEY"] == HARNESS_KEY  # harness key untouched


def test_dotenv_target_values_stay_out_of_os_environ(fake_bin, tmp_path, monkeypatch):
    monkeypatch.delenv("TARGET_API_KEY", raising=False)
    monkeypatch.delenv("TARGET_CLI_BIN", raising=False)
    monkeypatch.delenv("FROM_DOTENV", raising=False)
    env_file = tmp_path / "dotenv"
    env_file.write_text(f"TARGET_CLI_BIN={fake_bin}\nTARGET_API_KEY=from-file\nFROM_DOTENV=yes\n")
    cfg = load_config(env_file=env_file)
    assert cfg.target_env["LANGSMITH_API_KEY"] == "from-file"
    assert "TARGET_API_KEY" not in os.environ
    assert os.environ["FROM_DOTENV"] == "yes"
    monkeypatch.delenv("FROM_DOTENV")


def test_shared_target_and_harness_key_warns(fake_bin, monkeypatch):
    monkeypatch.setenv("TARGET_CLI_BIN", str(fake_bin))
    monkeypatch.setenv("LANGSMITH_API_KEY", "same")
    monkeypatch.setenv("TARGET_API_KEY", "same")
    cfg = load_config(env_file=None)
    assert any("rule 5" in w for w in cfg.warnings)


def test_missing_binary(monkeypatch):
    monkeypatch.setenv("TARGET_CLI_BIN", "definitely-not-a-real-binary-xyz")
    with pytest.raises(ConfigError):
        load_config(env_file=None)
