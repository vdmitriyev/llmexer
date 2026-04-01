"""Tests for the `search new` command."""

import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.commands.search as search_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(search_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def test_search_new_creates_yaml_file(experiments_dir, mock_no_dotenv, monkeypatch):
    """Creating a new search config should create a YAML file in the searches directory."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "new", "--query", "test query"])
    assert result.exit_code == 0
    assert "Created search config:" in result.output
    assert "search_" in result.output
    assert ".yaml" in result.output

    # Check that searches directory was created
    searches_dir = experiments_dir / "test-exp" / "searches"
    assert searches_dir.exists()

    # Check that YAML file was created
    yaml_files = list(searches_dir.glob("search_*.yaml"))
    assert len(yaml_files) == 1


def test_search_new_yaml_content(experiments_dir, mock_no_dotenv, monkeypatch):
    """The created YAML file should contain correct parameters."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(
        app,
        ["search", "new", "--query", "machine learning", "--year", "2022-2024"],
    )
    assert result.exit_code == 0

    # Read the YAML file
    searches_dir = experiments_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("search_*.yaml"))
    yaml_file = yaml_files[0]

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    assert config["query"] == "machine learning"
    assert config["year"] == "2022-2024"
    assert config["onlyOpenAccess"] is False


def test_search_new_with_only_open_access(experiments_dir, mock_no_dotenv, monkeypatch):
    """The onlyOpenAccess parameter should be set correctly."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(
        app,
        ["search", "new", "--query", "test", "--only-open-access"],
    )
    assert result.exit_code == 0

    # Read the YAML file
    searches_dir = experiments_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("search_*.yaml"))
    yaml_file = yaml_files[0]

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    assert config["onlyOpenAccess"] is True


def test_search_new_default_values(experiments_dir, mock_no_dotenv, monkeypatch):
    """Default values should be used when not provided."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "new"])
    assert result.exit_code == 0

    # Read the YAML file
    searches_dir = experiments_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("search_*.yaml"))
    yaml_file = yaml_files[0]

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    assert config["query"] == "sample request"
    assert config["year"] == "2020-2025"
    assert config["onlyOpenAccess"] is False


def test_search_new_without_eid_raises(experiments_dir, mock_no_dotenv, monkeypatch):
    """Creating search config without experiment ID should raise error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["search", "new", "--query", "test"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)


def test_search_new_nonexistent_experiment_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Creating search config for nonexistent experiment should raise error."""
    monkeypatch.setenv("EXPERIMENT_ID", "nonexistent")

    result = runner.invoke(app, ["search", "new", "--query", "test"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)


def test_search_new_filename_format(experiments_dir, mock_no_dotenv, monkeypatch):
    """The YAML filename should follow the search_YYYYMMDD-GUID.yaml format."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "new", "--query", "test"])
    assert result.exit_code == 0

    searches_dir = experiments_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("search_*.yaml"))
    filename = yaml_files[0].name

    # Check format: search_YYYYMMDD-XXXXXXXX.yaml
    assert filename.startswith("search_")
    assert filename.endswith(".yaml")
    parts = filename[7:-5].split("-")  # Remove "search_" and ".yaml"
    assert len(parts) == 2
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 8  # GUID
