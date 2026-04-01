"""Tests for the `experiment rename` command."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import ExperimentAlreadyExistsException, LLMExerException

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


def test_rename_success(experiments_dir):
    """Renaming an existing experiment should succeed and update the folder name."""
    os.makedirs(experiments_dir / "old-exp")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "old-exp", "--new-id", "new-exp"]
    )
    assert result.exit_code == 0
    assert not (experiments_dir / "old-exp").exists()
    assert (experiments_dir / "new-exp").is_dir()
    assert "old-exp" in result.output
    assert "new-exp" in result.output


def test_rename_nonexistent_experiment_raises(experiments_dir):
    """Renaming a non-existent experiment should raise LLMExerException."""
    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "nonexistent", "--new-id", "new-exp"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_rename_nonexistent_experiment_error_message(experiments_dir):
    """The exception message should mention the non-existent experiment ID."""
    old_id = "nonexistent-exp"
    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", old_id, "--new-id", "new-exp"]
    )
    assert old_id in str(result.exception)
    assert "does not exist" in str(result.exception)


def test_rename_to_existing_experiment_raises(experiments_dir):
    """Renaming to an existing experiment ID should raise ExperimentAlreadyExistsException."""
    os.makedirs(experiments_dir / "old-exp")
    os.makedirs(experiments_dir / "existing-exp")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "old-exp", "--new-id", "existing-exp"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentAlreadyExistsException)


def test_rename_to_existing_experiment_error_message(experiments_dir):
    """The exception message should mention the already-existing experiment ID."""
    os.makedirs(experiments_dir / "old-exp")
    os.makedirs(experiments_dir / "existing-exp")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "old-exp", "--new-id", "existing-exp"]
    )
    assert "existing-exp" in str(result.exception)
    assert "already exists" in str(result.exception)


def test_rename_preserves_experiment_contents(experiments_dir):
    """Renaming should preserve all files and subdirectories in the experiment folder."""
    old_exp_path = experiments_dir / "old-exp"
    os.makedirs(old_exp_path)
    (old_exp_path / "file1.txt").write_text("content1")
    (old_exp_path / "file2.txt").write_text("content2")
    os.makedirs(old_exp_path / "subdir")
    (old_exp_path / "subdir" / "file3.txt").write_text("content3")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "old-exp", "--new-id", "new-exp"]
    )
    assert result.exit_code == 0

    new_exp_path = experiments_dir / "new-exp"
    assert (new_exp_path / "file1.txt").read_text() == "content1"
    assert (new_exp_path / "file2.txt").read_text() == "content2"
    assert (new_exp_path / "subdir" / "file3.txt").read_text() == "content3"


def test_rename_with_special_characters(experiments_dir):
    """Renaming should work with experiment IDs containing special characters."""
    os.makedirs(experiments_dir / "exp_123")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "exp_123", "--new-id", "exp-456_test"]
    )
    assert result.exit_code == 0
    assert not (experiments_dir / "exp_123").exists()
    assert (experiments_dir / "exp-456_test").is_dir()


def test_rename_old_experiment_leaves_other_experiments_intact(experiments_dir):
    """Renaming one experiment should not affect other experiments."""
    os.makedirs(experiments_dir / "exp1")
    os.makedirs(experiments_dir / "exp2")
    os.makedirs(experiments_dir / "exp3")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "exp2", "--new-id", "exp2-renamed"]
    )
    assert result.exit_code == 0
    assert (experiments_dir / "exp1").is_dir()
    assert (experiments_dir / "exp2-renamed").is_dir()
    assert (experiments_dir / "exp3").is_dir()
    assert not (experiments_dir / "exp2").exists()


def test_rename_requires_new_id(experiments_dir):
    """Calling rename without --new-id should fail."""
    os.makedirs(experiments_dir / "old-exp")

    # Missing --new-id
    result = runner.invoke(app, ["experiment", "rename", "--old-id", "old-exp"])
    assert result.exit_code != 0

    # Missing both
    result = runner.invoke(app, ["experiment", "rename"])
    assert result.exit_code != 0


def test_rename_uses_current_experiment_as_default(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --old-id is not provided, should use EXPERIMENT_ID from environment."""
    os.makedirs(experiments_dir / "current-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "current-exp")

    result = runner.invoke(app, ["experiment", "rename", "--new-id", "renamed-exp"])
    assert result.exit_code == 0
    assert not (experiments_dir / "current-exp").exists()
    assert (experiments_dir / "renamed-exp").is_dir()
    assert "current-exp" in result.output
    assert "renamed-exp" in result.output


def test_rename_without_old_id_and_no_env_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --old-id is not provided and EXPERIMENT_ID is not set, should raise error."""
    # Reset settings from previous tests
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "rename", "--new-id", "new-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "No experiment ID provided" in str(result.exception)


def test_rename_old_id_overrides_env(experiments_dir, mock_no_dotenv, monkeypatch):
    """When --old-id is provided, it should override EXPERIMENT_ID from environment."""
    os.makedirs(experiments_dir / "env-exp")
    os.makedirs(experiments_dir / "cli-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "env-exp")

    result = runner.invoke(
        app, ["experiment", "rename", "--old-id", "cli-exp", "--new-id", "renamed-exp"]
    )
    assert result.exit_code == 0
    assert (experiments_dir / "env-exp").is_dir()  # env-exp should still exist
    assert not (experiments_dir / "cli-exp").exists()  # cli-exp should be renamed
    assert (experiments_dir / "renamed-exp").is_dir()
    assert "cli-exp" in result.output
    assert "renamed-exp" in result.output
