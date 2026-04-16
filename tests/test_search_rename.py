"""Tests for the `search rename` command."""

import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
    LLMExerException,
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


@pytest.fixture()
def searches_dir(experiments_dir, monkeypatch):
    """Create experiment + searches/ directory and set EXPERIMENT_ID."""
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")
    s = experiments_dir / "test-exp" / "searches"
    s.mkdir(parents=True)
    return s


def _make_yaml(searches_dir, search_id):
    p = searches_dir / f"{search_id}.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump({"query": "test", "year": "2020-2025", "onlyOpenAccess": False}, f)
    return p


def test_search_rename_renames_yaml(searches_dir, mock_no_dotenv):
    """Renaming should move the YAML file from old-id to new-id."""
    _make_yaml(searches_dir, "old-search")

    result = runner.invoke(
        app, ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"]
    )
    assert result.exit_code == 0
    assert not (searches_dir / "old-search.yaml").exists()
    assert (searches_dir / "new-search.yaml").exists()


def test_search_rename_renames_all_associated_files(searches_dir, mock_no_dotenv):
    """All associated files should be renamed when they exist."""
    suffixes = [
        ".yaml",
        "_results.csv",
        "_results_raw.json",
        "_filtered.csv",
        "_results_download_failed.csv",
    ]
    for suffix in suffixes:
        (searches_dir / f"old-search{suffix}").write_text("data")

    result = runner.invoke(
        app, ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"]
    )
    assert result.exit_code == 0
    for suffix in suffixes:
        assert not (searches_dir / f"old-search{suffix}").exists()
        assert (searches_dir / f"new-search{suffix}").exists()


def test_search_rename_skips_missing_optional_files(searches_dir, mock_no_dotenv):
    """Rename should succeed even when only the YAML is present."""
    _make_yaml(searches_dir, "old-search")

    result = runner.invoke(
        app, ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"]
    )
    assert result.exit_code == 0
    assert (searches_dir / "new-search.yaml").exists()


def test_search_rename_accepts_yaml_extension(searches_dir, mock_no_dotenv):
    """Passing old-id with .yaml extension should work the same as bare stem."""
    _make_yaml(searches_dir, "old-search")

    result = runner.invoke(
        app,
        [
            "search",
            "rename",
            "--old-id",
            "old-search.yaml",
            "--new-id",
            "new-search",
        ],
    )
    assert result.exit_code == 0
    assert (searches_dir / "new-search.yaml").exists()


def test_search_rename_nonexistent_old_id_raises(searches_dir, mock_no_dotenv):
    """Renaming a non-existent search ID should raise LLMExerException."""
    result = runner.invoke(
        app,
        ["search", "rename", "--old-id", "ghost", "--new-id", "new-search"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_search_rename_existing_new_id_raises(searches_dir, mock_no_dotenv):
    """Renaming to an already-existing search ID should raise LLMExerException."""
    _make_yaml(searches_dir, "old-search")
    _make_yaml(searches_dir, "new-search")

    result = runner.invoke(
        app,
        ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_search_rename_without_eid_raises(experiments_dir, mock_no_dotenv, monkeypatch):
    """Omitting --eid with no EXPERIMENT_ID env var should raise ExperimentIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(
        app,
        ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)


def test_search_rename_nonexistent_experiment_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Providing a non-existent experiment ID should raise ExperimentNotExistsException."""
    monkeypatch.setenv("EXPERIMENT_ID", "nonexistent")

    result = runner.invoke(
        app,
        ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)
