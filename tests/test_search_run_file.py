"""Tests for the `search run` command with search file support."""

import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException

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


def test_search_run_with_query(experiments_dir, mock_no_dotenv, monkeypatch):
    """Running search with --query should work."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result.exit_code == 0
    assert "Query: test query" in result.output
    assert "Year: 2020-2025" in result.output
    assert "Only Open Access: False" in result.output


def test_search_run_with_file_file(experiments_dir, mock_no_dotenv, monkeypatch):
    """Running search with --file should load parameters from YAML."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    # Create a search file
    search_file_path = experiments_dir / "test-exp" / "test_search_file.yaml"
    search_file_data = {
        "query": "neural networks",
        "year": "2022-2024",
        "onlyOpenAccess": True,
    }
    with open(search_file_path, "w") as f:
        yaml.dump(search_file_data, f)

    result = runner.invoke(app, ["search", "run", "--file", str(search_file_path)])
    assert result.exit_code == 0
    assert "Loaded config from:" in result.output
    assert "Query: neural networks" in result.output
    assert "Year: 2022-2024" in result.output
    assert "Only Open Access: True" in result.output


def test_search_run_nonexistent_file_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Using a nonexistent search file should raise error."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(
        app, ["search", "run", "--file", "/nonexistent/search_file.yaml"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "does not exist" in str(result.exception)


def test_search_run_without_query_or_file_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Running search without query or config should raise error."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "No query provided" in str(result.exception)


def test_search_run_file_with_missing_fields(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """search file with missing fields should use defaults."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    # Create a search file with only query
    search_file_path = experiments_dir / "test-exp" / "minimal_search_file.yaml"
    search_file_data = {"query": "minimal query"}
    with open(search_file_path, "w") as f:
        yaml.dump(search_file_data, f)

    result = runner.invoke(app, ["search", "run", "--file", str(search_file_path)])
    assert result.exit_code == 0
    assert "Query: minimal query" in result.output
    assert "Year: 2020-2025" in result.output  # Default
    assert "Only Open Access: False" in result.output  # Default
