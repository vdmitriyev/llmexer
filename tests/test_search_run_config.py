"""Tests for the `search run` command with config file support."""

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


def test_search_run_with_config_file(experiments_dir, mock_no_dotenv, monkeypatch):
    """Running search with --config should load parameters from YAML."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    # Create a config file
    config_path = experiments_dir / "test-exp" / "test_config.yaml"
    config_data = {
        "query": "neural networks",
        "year": "2022-2024",
        "onlyOpenAccess": True,
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["search", "run", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "Loaded config from:" in result.output
    assert "Query: neural networks" in result.output
    assert "Year: 2022-2024" in result.output
    assert "Only Open Access: True" in result.output


def test_search_run_config_overrides_query(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Config file parameters should override command line query."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    # Create a config file
    config_path = experiments_dir / "test-exp" / "test_config.yaml"
    config_data = {"query": "from config", "year": "2023", "onlyOpenAccess": False}
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(
        app,
        ["search", "run", "--query", "from cli", "--config", str(config_path)],
    )
    assert result.exit_code == 0
    assert "Query: from config" in result.output


def test_search_run_nonexistent_config_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Using a nonexistent config file should raise error."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(
        app, ["search", "run", "--config", "/nonexistent/config.yaml"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "does not exist" in str(result.exception)


def test_search_run_without_query_or_config_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Running search without query or config should raise error."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "No query provided" in str(result.exception)


def test_search_run_config_with_missing_fields(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """Config file with missing fields should use defaults."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    # Create a config file with only query
    config_path = experiments_dir / "test-exp" / "minimal_config.yaml"
    config_data = {"query": "minimal query"}
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["search", "run", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "Query: minimal query" in result.output
    assert "Year: 2020-2025" in result.output  # Default
    assert "Only Open Access: False" in result.output  # Default
