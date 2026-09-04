"""Tests for the `experiment rename` command."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException, ProjectAlreadyExistsException

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


def test_rename_success(projects_dir):
    """Renaming an existing experiment should succeed and update the folder name."""
    os.makedirs(projects_dir / "old-exp")

    result = runner.invoke(app, ["project", "rename", "--old-id", "old-exp", "--new-id", "new-exp"])
    assert result.exit_code == 0
    assert not (projects_dir / "old-exp").exists()
    assert (projects_dir / "new-exp").is_dir()
    assert "old-exp" in result.output
    assert "new-exp" in result.output


def test_rename_nonexistent_experiment_raises(projects_dir):
    """Renaming a non-existent experiment should raise LLMExerException."""
    result = runner.invoke(app, ["project", "rename", "--old-id", "nonexistent", "--new-id", "new-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_rename_nonexistent_experiment_error_message(projects_dir):
    """The exception message should mention the non-existent experiment ID."""
    old_id = "nonexistent-exp"
    result = runner.invoke(app, ["project", "rename", "--old-id", old_id, "--new-id", "new-exp"])
    assert old_id in str(result.exception)
    assert "does not exist" in str(result.exception)


def test_rename_to_existing_experiment_raises(projects_dir):
    """Renaming to an existing experiment ID should raise ProjectAlreadyExistsException."""
    os.makedirs(projects_dir / "old-exp")
    os.makedirs(projects_dir / "existing-exp")

    result = runner.invoke(app, ["project", "rename", "--old-id", "old-exp", "--new-id", "existing-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectAlreadyExistsException)


def test_rename_to_existing_experiment_error_message(projects_dir):
    """The exception message should mention the already-existing experiment ID."""
    os.makedirs(projects_dir / "old-exp")
    os.makedirs(projects_dir / "existing-exp")

    result = runner.invoke(app, ["project", "rename", "--old-id", "old-exp", "--new-id", "existing-exp"])
    assert "existing-exp" in str(result.exception)
    assert "already exists" in str(result.exception)


def test_rename_preserves_experiment_contents(projects_dir):
    """Renaming should preserve all files and subdirectories in the experiment folder."""
    old_exp_path = projects_dir / "old-exp"
    os.makedirs(old_exp_path)
    (old_exp_path / "file1.txt").write_text("content1")
    (old_exp_path / "file2.txt").write_text("content2")
    os.makedirs(old_exp_path / "subdir")
    (old_exp_path / "subdir" / "file3.txt").write_text("content3")

    result = runner.invoke(app, ["project", "rename", "--old-id", "old-exp", "--new-id", "new-exp"])
    assert result.exit_code == 0

    new_exp_path = projects_dir / "new-exp"
    assert (new_exp_path / "file1.txt").read_text() == "content1"
    assert (new_exp_path / "file2.txt").read_text() == "content2"
    assert (new_exp_path / "subdir" / "file3.txt").read_text() == "content3"


def test_rename_with_special_characters(projects_dir):
    """Renaming should work with experiment IDs containing special characters."""
    os.makedirs(projects_dir / "exp_123")

    result = runner.invoke(app, ["project", "rename", "--old-id", "exp_123", "--new-id", "exp-456_test"])
    assert result.exit_code == 0
    assert not (projects_dir / "exp_123").exists()
    assert (projects_dir / "exp-456_test").is_dir()


def test_rename_old_experiment_leaves_other_experiments_intact(projects_dir):
    """Renaming one experiment should not affect other experiments."""
    os.makedirs(projects_dir / "exp1")
    os.makedirs(projects_dir / "exp2")
    os.makedirs(projects_dir / "exp3")

    result = runner.invoke(app, ["project", "rename", "--old-id", "exp2", "--new-id", "exp2-renamed"])
    assert result.exit_code == 0
    assert (projects_dir / "exp1").is_dir()
    assert (projects_dir / "exp2-renamed").is_dir()
    assert (projects_dir / "exp3").is_dir()
    assert not (projects_dir / "exp2").exists()


def test_rename_requires_new_id(projects_dir):
    """Calling rename without --new-id should fail."""
    os.makedirs(projects_dir / "old-exp")

    # Missing --new-id
    result = runner.invoke(app, ["project", "rename", "--old-id", "old-exp"])
    assert result.exit_code != 0

    # Missing both
    result = runner.invoke(app, ["project", "rename"])
    assert result.exit_code != 0


def test_rename_uses_current_experiment_as_default(projects_dir, mock_no_dotenv, monkeypatch):
    """When --old-id is not provided, should use PROJECT_ID from environment."""
    os.makedirs(projects_dir / "current-exp")
    monkeypatch.setenv("PROJECT_ID", "current-exp")

    result = runner.invoke(app, ["project", "rename", "--new-id", "renamed-exp"])
    assert result.exit_code == 0
    assert not (projects_dir / "current-exp").exists()
    assert (projects_dir / "renamed-exp").is_dir()
    assert "current-exp" in result.output
    assert "renamed-exp" in result.output


def test_rename_without_old_id_and_no_env_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """When --old-id is not provided and PROJECT_ID is not set, should raise error."""
    # Reset settings from previous tests
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["project", "rename", "--new-id", "new-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "No project ID provided" in str(result.exception)


def test_rename_old_id_overrides_env(projects_dir, mock_no_dotenv, monkeypatch):
    """When --old-id is provided, it should override PROJECT_ID from environment."""
    os.makedirs(projects_dir / "env-exp")
    os.makedirs(projects_dir / "cli-exp")
    monkeypatch.setenv("PROJECT_ID", "env-exp")

    result = runner.invoke(app, ["project", "rename", "--old-id", "cli-exp", "--new-id", "renamed-exp"])
    assert result.exit_code == 0
    assert (projects_dir / "env-exp").is_dir()  # env-exp should still exist
    assert not (projects_dir / "cli-exp").exists()  # cli-exp should be renamed
    assert (projects_dir / "renamed-exp").is_dir()
    assert "cli-exp" in result.output
    assert "renamed-exp" in result.output
