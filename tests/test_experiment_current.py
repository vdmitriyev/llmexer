"""Tests for the `experiment current` command."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.commands.experiment as exp_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(exp_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def test_current_no_experiment_set(experiments_dir, mock_no_dotenv, monkeypatch):
    """When EXPERIMENT_ID is not set, should display a message."""
    # Reset settings from previous tests
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "current"])
    assert result.exit_code == 0
    assert "No current experiment set" in result.output
    assert "EXPERIMENT_ID" in result.output


def test_current_experiment_exists(experiments_dir, mock_no_dotenv, monkeypatch):
    """When EXPERIMENT_ID is set and the experiment exists, should display it."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")
    result = runner.invoke(app, ["experiment", "current"])
    assert result.exit_code == 0
    assert "test-exp" in result.output
    assert "not found" not in result.output.lower()


def test_current_experiment_not_found(experiments_dir, mock_no_dotenv, monkeypatch):
    """When EXPERIMENT_ID is set but experiment doesn't exist, should show warning."""
    monkeypatch.setenv("EXPERIMENT_ID", "nonexistent-exp")
    result = runner.invoke(app, ["experiment", "current"])
    assert result.exit_code == 0
    assert "nonexistent-exp" in result.output
    assert "not found" in result.output.lower()


def test_current_with_verbose_flag(experiments_dir, mock_no_dotenv, monkeypatch):
    """The current command should work with --verbose flag."""
    os.makedirs(experiments_dir / "verbose-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "verbose-exp")
    result = runner.invoke(app, ["--verbose", "experiment", "current"])
    assert result.exit_code == 0
    assert "verbose-exp" in result.output
