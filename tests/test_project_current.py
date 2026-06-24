"""Tests for the `experiment current` command."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def test_current_no_experiment_set(projects_dir, mock_no_dotenv, monkeypatch):
    """When PROJECT_ID is not set, should display a message."""
    # Reset settings from previous tests
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["project", "current"])
    assert result.exit_code == 0
    assert "No current project set" in result.output
    assert "PROJECT_ID" in result.output


def test_current_experiment_exists(projects_dir, mock_no_dotenv, monkeypatch):
    """When PROJECT_ID is set and the experiment exists, should display it."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")
    result = runner.invoke(app, ["project", "current"])
    assert result.exit_code == 0
    assert "test-exp" in result.output
    assert "not found" not in result.output.lower()


def test_current_experiment_not_found(projects_dir, mock_no_dotenv, monkeypatch):
    """When PROJECT_ID is set but experiment doesn't exist, should show warning."""
    monkeypatch.setenv("PROJECT_ID", "nonexistent-exp")
    result = runner.invoke(app, ["project", "current"])
    assert result.exit_code == 0
    assert "nonexistent-exp" in result.output
    assert "not found" in result.output.lower()


def test_current_with_verbose_flag(projects_dir, mock_no_dotenv, monkeypatch):
    """The current command should work with --verbose flag."""
    os.makedirs(projects_dir / "verbose-exp")
    monkeypatch.setenv("PROJECT_ID", "verbose-exp")
    result = runner.invoke(app, ["--verbose", "project", "current"])
    assert result.exit_code == 0
    assert "verbose-exp" in result.output
