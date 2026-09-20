"""Tests for bare `llmexer experiment` — which database am I working on?"""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from tests.db_helpers import OLLAMA_ROW, seed_db

runner = CliRunner()

PID = "bare-experiment-exp"


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.experiment as experiment_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(experiment_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def test_without_a_project_it_says_so(projects_dir, mock_no_dotenv, monkeypatch):
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["experiment"])

    assert result.exit_code == 0
    assert "No default project has been set." in result.output
    assert "Usage:" not in result.output


def test_a_missing_project_folder_is_reported(projects_dir, mock_no_dotenv, monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "ghost-exp")

    result = runner.invoke(app, ["experiment"])

    assert result.exit_code == 0
    assert "ghost-exp" in result.output
    assert "not found" in result.output.lower()


def test_an_uninitialised_project_points_at_init(projects_dir, mock_no_dotenv, monkeypatch):
    os.makedirs(projects_dir / PID)
    monkeypatch.setenv("PROJECT_ID", PID)

    result = runner.invoke(app, ["experiment"])

    assert result.exit_code == 0
    assert "experiment init" in result.output


def test_a_project_without_a_database_points_at_generate(projects_dir, mock_no_dotenv, monkeypatch):
    os.makedirs(projects_dir / PID / "experiment")
    monkeypatch.setenv("PROJECT_ID", PID)

    result = runner.invoke(app, ["experiment"])

    assert result.exit_code == 0
    assert "experiment generate" in result.output


def test_the_newest_database_is_reported(projects_dir, mock_no_dotenv, monkeypatch):
    exp_subdir = projects_dir / PID / "experiment"
    os.makedirs(exp_subdir)
    for name in ("experiment_20240101_01.db", "experiment_20240101_02.db"):
        seed_db(exp_subdir / name, {"ollama": [dict(OLLAMA_ROW)]})
    monkeypatch.setenv("PROJECT_ID", PID)

    result = runner.invoke(app, ["experiment"])

    assert result.exit_code == 0
    assert "experiment_20240101_02.db" in result.output
    assert "experiment_20240101_01.db" not in result.output


def test_the_exp_alias_behaves_the_same(projects_dir, mock_no_dotenv, monkeypatch):
    exp_subdir = projects_dir / PID / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / "experiment_20240101_01.db", {"ollama": [dict(OLLAMA_ROW)]})
    monkeypatch.setenv("PROJECT_ID", PID)

    assert runner.invoke(app, ["exp"]).output == runner.invoke(app, ["experiment"]).output


def test_help_still_lists_the_commands(projects_dir, mock_no_dotenv):
    result = runner.invoke(app, ["experiment", "--help"])

    assert result.exit_code == 0
    for command in ("init", "generate", "run", "list", "export", "fix"):
        assert command in result.output
