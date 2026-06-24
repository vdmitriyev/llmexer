"""Tests for the `experiment create` command."""

import os

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import ProjectAlreadyExistsException

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


def test_create_auto_generated_id(projects_dir):
    """Running `experiment create` without --id should create a YYYYMMDD-XXXX folder."""
    result = runner.invoke(app, ["project", "create"])
    assert result.exit_code == 0
    folders = list(projects_dir.iterdir())
    assert len(folders) == 1
    name = folders[0].name
    assert len(name) == 17  # YYYYMMDD-8chars
    assert name[8] == "-"


def test_create_with_custom_id(projects_dir):
    """Running `experiment create --id my-exp` should create a folder named my-exp."""
    result = runner.invoke(app, ["project", "create", "--id", "my-exp"])
    assert result.exit_code == 0
    assert (projects_dir / "my-exp").is_dir()
    assert "my-exp" in result.output


def test_create_duplicate_id_raises(projects_dir):
    """Running `experiment create --id` twice with the same ID should raise ProjectAlreadyExistsException."""
    runner.invoke(app, ["project", "create", "--id", "duplicate-exp"])
    result = runner.invoke(app, ["project", "create", "--id", "duplicate-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectAlreadyExistsException)


def test_create_duplicate_id_error_message(projects_dir):
    """The exception message should mention the duplicate experiment ID."""
    exp_id = "dup-exp"
    runner.invoke(app, ["project", "create", "--id", exp_id])
    result = runner.invoke(app, ["project", "create", "--id", exp_id])
    assert exp_id in str(result.exception)
