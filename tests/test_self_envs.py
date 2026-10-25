"""Tests for ``llmexer self envs``."""

from typer.testing import CliRunner

from llmexer.base.llm_provider import URL_MAP
from llmexer.cli import app

runner = CliRunner()


def _invoke_envs(monkeypatch, **env):
    monkeypatch.setenv("COLUMNS", "200")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = runner.invoke(app, ["self", "envs"])
    assert result.exit_code == 0, result.output
    return result.output


def test_envs_lists_every_provider(monkeypatch):
    output = _invoke_envs(monkeypatch)
    for provider in URL_MAP:
        assert f"PROVIDER_{provider.upper()}_URL" in output
        assert f"PROVIDER_{provider.upper()}_KEY" in output
    assert "PROVIDER_OPENROUTER_MAX_SPEND" in output
    assert "MAX_OPEN_ALEX_RESPONSES" in output
    assert "APP_LOG_LEVEL" in output


def test_envs_masks_secrets(monkeypatch):
    output = _invoke_envs(monkeypatch, PROVIDER_OPENAI_KEY="sk-secret-value")
    assert "sk-secret-value" not in output
    assert "********" in output


def test_envs_shows_custom_provider(monkeypatch):
    output = _invoke_envs(monkeypatch, PROVIDER_MYLAB_URL="http://mylab.example/v1")
    assert "PROVIDER_MYLAB_URL" in output
    assert "PROVIDER_MYLAB_KEY" in output
    assert "http://mylab.example/v1" in output
