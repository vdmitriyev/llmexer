"""Tests for the `search rename` command."""

import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    LLMExerException,
    ProjectIDRequiredException,
    ProjectNotExistsException,
)

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def searches_dir(projects_dir, monkeypatch):
    """Create experiment + searches/ directory and set PROJECT_ID."""
    monkeypatch.setenv("PROJECT_ID", "test-exp")
    s = projects_dir / "test-exp" / "searches"
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
        "__results.csv",
        "__filtered.csv",
        "__results_download_failed.csv",
    ]
    for suffix in suffixes:
        (searches_dir / f"old-search{suffix}").write_text("data")

    # The raw JSON responses live in the `jsons/` subdirectory.
    jsons_dir = searches_dir / "jsons"
    jsons_dir.mkdir()
    (jsons_dir / "old-search__results_raw.json").write_text("data")

    result = runner.invoke(
        app, ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"]
    )
    assert result.exit_code == 0
    for suffix in suffixes:
        assert not (searches_dir / f"old-search{suffix}").exists()
        assert (searches_dir / f"new-search{suffix}").exists()
    assert not (jsons_dir / "old-search__results_raw.json").exists()
    assert (jsons_dir / "new-search__results_raw.json").exists()


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


def test_search_rename_without_eid_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Omitting --pid with no PROJECT_ID env var should raise ProjectIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(
        app,
        ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_search_rename_nonexistent_experiment_raises(
    projects_dir, mock_no_dotenv, monkeypatch
):
    """Providing a non-existent experiment ID should raise ProjectNotExistsException."""
    monkeypatch.setenv("PROJECT_ID", "nonexistent")

    result = runner.invoke(
        app,
        ["search", "rename", "--old-id", "old-search", "--new-id", "new-search"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)
