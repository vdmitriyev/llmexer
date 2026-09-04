"""Tests for the `search run` command with --pid parameter."""

import os
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ProjectIDRequiredException,
    ProjectNotExistsException,
)

runner = CliRunner()


def _make_mock_s2_response() -> Mock:
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = {"data": [], "token": None}
    return mock_resp


def _make_mock_openalex_response() -> Mock:
    """A single empty OpenAlex page: no results and no next cursor -> one call, then stop."""
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = {
        "results": [],
        "meta": {"count": 0, "next_cursor": None},
    }
    return mock_resp


@pytest.fixture()
def mock_openalex_session():
    """Mock the OpenAlex HTTP session so tests never hit the real API when the key is set."""
    mock_session = Mock()
    mock_session.get.return_value = _make_mock_openalex_response()
    with patch(
        "llmexer.base.search_openalex.make_http_session",
        return_value=mock_session,
    ):
        yield mock_session


@pytest.fixture()
def mock_s2_session(mock_openalex_session):
    mock_session = Mock()
    mock_session.get.return_value = _make_mock_s2_response()
    with patch(
        "llmexer.base.search_semantic_scholar.make_http_session",
        return_value=mock_session,
    ):
        yield mock_session


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


def test_search_uses_current_experiment_as_default(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """When --pid is not provided, should use PROJECT_ID from environment."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result.exit_code == 0
    assert "test-exp" in result.output


def test_search_without_eid_and_no_env_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """When --pid is not provided and PROJECT_ID is not set, should raise error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)
    assert "No project ID provided" in str(result.exception)


def test_search_eid_overrides_env(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """When --pid is provided, it should override PROJECT_ID from environment."""
    os.makedirs(projects_dir / "env-exp")
    os.makedirs(projects_dir / "cli-exp")
    monkeypatch.setenv("PROJECT_ID", "env-exp")

    result = runner.invoke(app, ["search", "run", "--pid", "cli-exp", "--query", "test query"])
    assert result.exit_code == 0
    assert "cli-exp" in result.output


def test_search_nonexistent_experiment_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """When experiment does not exist, should raise ProjectNotExistsException."""
    monkeypatch.setenv("PROJECT_ID", "nonexistent-exp")

    result = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)
