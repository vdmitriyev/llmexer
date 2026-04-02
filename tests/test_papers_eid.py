"""Tests for the `papers rename` command with --eid parameter."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def test_papers_rename_uses_current_experiment_as_default(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is not provided, should use EXPERIMENT_ID from environment."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["papers", "rename"])
    assert result.exit_code == 0


def test_papers_rename_without_eid_and_no_env_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is not provided and EXPERIMENT_ID is not set, should raise error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["papers", "rename"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)
    assert "No experiment ID provided" in str(result.exception)


def test_papers_rename_eid_overrides_env(experiments_dir, mock_no_dotenv, monkeypatch):
    """When --eid is provided, it should override EXPERIMENT_ID from environment."""
    os.makedirs(experiments_dir / "env-exp")
    os.makedirs(experiments_dir / "cli-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "env-exp")

    result = runner.invoke(app, ["papers", "rename", "--eid", "cli-exp"])
    assert result.exit_code == 0


def test_papers_rename_nonexistent_experiment_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When experiment does not exist, should raise ExperimentNotExistsException."""
    monkeypatch.setenv("EXPERIMENT_ID", "nonexistent-exp")

    result = runner.invoke(app, ["papers", "rename"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)
