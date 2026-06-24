"""Tests for the `search create` command."""

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


def test_search_create_creates_yaml_file(projects_dir, mock_no_dotenv, monkeypatch):
    """Creating a new search config should create a YAML file in the searches directory."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "create", "--query", "test query"])
    assert result.exit_code == 0
    assert "Created search config:" in result.output
    assert ".yaml" in result.output
    assert ".yaml" in result.output

    # Check that searches directory was created
    searches_dir = projects_dir / "test-exp" / "searches"
    assert searches_dir.exists()

    # Check that YAML file was created
    yaml_files = list(searches_dir.glob("*.yaml"))
    assert len(yaml_files) == 1


def test_search_create_default_values(projects_dir, mock_no_dotenv, monkeypatch):
    """Default values should be used when not provided."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "create"])
    assert result.exit_code == 0

    # Read the YAML file
    searches_dir = projects_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("*.yaml"))
    yaml_file = yaml_files[0]

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    assert config["query"] == "influence of machine learning on computer science"
    assert config["year"] == "2020-2025"
    assert config["onlyOpenAccess"] is False


def test_search_create_without_eid_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Creating search config without experiment ID should raise error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["search", "create", "--query", "test"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_search_create_nonexistent_experiment_raises(
    projects_dir, mock_no_dotenv, monkeypatch
):
    """Creating search config for nonexistent experiment should raise error."""
    monkeypatch.setenv("PROJECT_ID", "nonexistent")

    result = runner.invoke(app, ["search", "create", "--query", "test"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


def test_search_create_filename_format(projects_dir, mock_no_dotenv, monkeypatch):
    """The YAML filename should follow the YYYYMMDD-GUID.yaml format."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "create", "--query", "test"])
    assert result.exit_code == 0

    searches_dir = projects_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("*.yaml"))
    filename = yaml_files[0].name

    # Check format: YYYYMMDD-XXXXXXXX.yaml
    assert filename.endswith(".yaml")
    parts = filename[:-5].split("-")  # Remove ".yaml"
    assert len(parts) == 2
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 8  # GUID
