"""Tests for the `search list` command."""

import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
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


def _make_yaml(searches_dir, search_id, query="test query", year="2020-2025"):
    yaml_path = searches_dir / f"{search_id}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"query": query, "year": year, "onlyOpenAccess": False}, f)
    return yaml_path


def test_search_list_no_searches_dir(projects_dir, mock_no_dotenv, monkeypatch):
    """Experiment exists but has no searches/ directory -> prints no searches found."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code == 0
    assert "No searches found." in result.output


def test_search_list_empty_dir(projects_dir, mock_no_dotenv, monkeypatch):
    """searches/ directory exists but contains no YAML files -> prints no searches found."""
    searches_dir = projects_dir / "test-exp" / "searches"
    searches_dir.mkdir(parents=True)
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code == 0
    assert "No searches found." in result.output


def test_search_list_shows_yaml_files(projects_dir, mock_no_dotenv, monkeypatch):
    """Two YAML files in searches/ should both appear in the table output."""
    searches_dir = projects_dir / "test-exp" / "searches"
    searches_dir.mkdir(parents=True)
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    _make_yaml(searches_dir, "20260101-aaaaaaaa", query="first query")
    _make_yaml(searches_dir, "20260102-bbbbbbbb", query="second query")

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code == 0
    assert "20260101-aaaaaaaa.yaml" in result.output
    assert "20260102-bbbbbbbb.yaml" in result.output
    # Wide columns that are never truncated at default terminal width
    assert "Search file" in result.output
    assert "Results" in result.output


def test_search_list_shows_results_column(projects_dir, mock_no_dotenv, monkeypatch):
    """Row with a matching __results.csv shows Yes; row without shows No."""
    searches_dir = projects_dir / "test-exp" / "searches"
    searches_dir.mkdir(parents=True)
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    _make_yaml(searches_dir, "20260101-aaaaaaaa")
    _make_yaml(searches_dir, "20260102-bbbbbbbb")

    # Only the first search has a results CSV
    (searches_dir / "20260101-aaaaaaaa__results.csv").write_text("col\nval\n")

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code == 0
    assert "Yes" in result.output
    assert "No" in result.output


def test_search_list_without_eid_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Omitting --pid with no PROJECT_ID env var should raise ProjectIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_search_list_nonexistent_experiment_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Providing a non-existent experiment ID should raise ProjectNotExistsException."""
    monkeypatch.setenv("PROJECT_ID", "nonexistent")

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


def test_search_list_shows_stats_hint(projects_dir, mock_no_dotenv, monkeypatch):
    """The hint after the table should reference the latest (last alphabetically) YAML file."""
    searches_dir = projects_dir / "test-exp" / "searches"
    searches_dir.mkdir(parents=True)
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    _make_yaml(searches_dir, "20260101-aaaaaaaa")
    _make_yaml(searches_dir, "20260102-bbbbbbbb")

    result = runner.invoke(app, ["search", "list"])
    assert result.exit_code == 0
    assert "Example to view search stats:" in result.output
    assert "llmexer search stats --file 20260102-bbbbbbbb.yaml" in result.output
