"""Tests for the `experiment create` command."""

import os

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import ExperimentAlreadyExistsException

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.commands.experiment as exp_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(exp_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


def test_create_auto_generated_id(experiments_dir):
    """Running `experiment create` without --id should create a YYYYMMDD-XXXX folder."""
    result = runner.invoke(app, ["experiment", "create"])
    assert result.exit_code == 0
    folders = list(experiments_dir.iterdir())
    assert len(folders) == 1
    name = folders[0].name
    assert len(name) == 17  # YYYYMMDD-8chars
    assert name[8] == "-"


def test_create_with_custom_id(experiments_dir):
    """Running `experiment create --id my-exp` should create a folder named my-exp."""
    result = runner.invoke(app, ["experiment", "create", "--id", "my-exp"])
    assert result.exit_code == 0
    assert (experiments_dir / "my-exp").is_dir()
    assert "my-exp" in result.output


def test_create_duplicate_id_raises(experiments_dir):
    """Running `experiment create --id` twice with the same ID should raise ExperimentAlreadyExistsException."""
    runner.invoke(app, ["experiment", "create", "--id", "duplicate-exp"])
    result = runner.invoke(app, ["experiment", "create", "--id", "duplicate-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentAlreadyExistsException)


def test_create_duplicate_id_error_message(experiments_dir):
    """The exception message should mention the duplicate experiment ID."""
    exp_id = "dup-exp"
    runner.invoke(app, ["experiment", "create", "--id", exp_id])
    result = runner.invoke(app, ["experiment", "create", "--id", exp_id])
    assert exp_id in str(result.exception)
