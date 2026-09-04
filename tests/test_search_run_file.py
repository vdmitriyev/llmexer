"""Tests for the `search run` command with search file support."""

import json
import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException, SearchResultsAlreadyExistException

runner = CliRunner()


def _make_mock_s2_response(papers: list[dict]) -> Mock:
    """Build a mock requests.Response for one page with no pagination token."""
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = {"data": papers, "token": None}
    return mock_resp


_SAMPLE_PAPER = {
    "paperId": "abc123",
    "title": "Test Paper",
    "authors": [{"name": "Alice"}, {"name": "Bob"}],
    "abstract": "An abstract.",
    "isOpenAccess": True,
    "externalIds": {"DOI": "10.1234/test"},
    "year": 2023,
}


@pytest.fixture()
def mock_detect_language():
    """Patch _detect_language to avoid shadowed builtin filter() inside the command module."""
    with patch(
        "llmexer.base.search_semantic_scholar.detect_publication_lang",
        return_value="en",
    ):
        yield


@pytest.fixture()
def mock_openalex_session():
    """Mock the OpenAlex HTTP session so tests never hit the real API when the key is set.

    Returns a single empty page (no results, no next cursor) so run_openalex_search
    finishes in one iteration.
    """
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = {
        "results": [],
        "meta": {"count": 0, "next_cursor": None},
    }
    mock_session = Mock()
    mock_session.get.return_value = mock_resp
    with patch(
        "llmexer.base.search_openalex.make_http_session",
        return_value=mock_session,
    ):
        yield mock_session


@pytest.fixture()
def mock_s2_session(mock_detect_language, mock_openalex_session):
    """Patch requests.Session to return a single-page response with _SAMPLE_PAPER."""
    mock_session = Mock()
    mock_session.get.return_value = _make_mock_s2_response([_SAMPLE_PAPER])
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


def test_search_run_with_query(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """Running search with --query should work."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result.exit_code == 0
    assert "Query:" in result.output
    assert "test query" in result.output
    assert "Year:" in result.output
    assert "2020-2025" in result.output
    assert "Only Open Access:" in result.output
    assert "False" in result.output


def test_search_run_with_file_file(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """Running search with --file should load parameters from YAML."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    # Create a search file
    search_file_path = projects_dir / "test-exp" / "test_search_file.yaml"
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
    assert "Query:" in result.output
    assert "neural networks" in result.output
    assert "Year:" in result.output
    assert "2022-2024" in result.output
    assert "Only Open Access:" in result.output
    assert "True" in result.output


def test_search_run_nonexistent_file_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Using a nonexistent search file should raise error."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run", "--file", "/nonexistent/search_file.yaml"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "does not exist" in str(result.exception)


def test_search_run_without_query_or_file_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Running search without query or config should raise error."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "No query provided" in str(result.exception)


def test_search_run_file_with_missing_fields(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """search file with missing fields should use defaults."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    # Create a search file with only query
    search_file_path = projects_dir / "test-exp" / "minimal_search_file.yaml"
    search_file_data = {"query": "minimal query"}
    with open(search_file_path, "w") as f:
        yaml.dump(search_file_data, f)

    result = runner.invoke(app, ["search", "run", "--file", str(search_file_path)])
    assert result.exit_code == 0
    assert "Query:" in result.output
    assert "minimal query" in result.output
    assert "Year:" in result.output
    assert "2020-2025" in result.output  # Default
    assert "Only Open Access:" in result.output
    assert "False" in result.output  # Default


def test_search_run_creates_output_files(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """Happy path: running search creates JSON and CSV result files."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "run", "--query", "test query", "--limit", "10"])

    assert result.exit_code == 0
    assert "File with results" in result.output

    searches_dir = projects_dir / "test-exp" / "searches"
    json_files = list((searches_dir / "jsons").glob("*__results_raw.json"))
    csv_files = list(searches_dir.glob("*__results.csv"))
    assert len(json_files) == 1
    assert len(csv_files) == 1

    raw = json.loads(json_files[0].read_text())
    assert isinstance(raw, list)
    assert raw[0]["data"][0]["paperId"] == "abc123"

    df = pd.read_csv(csv_files[0], sep=";")
    assert len(df) == 1
    assert df.iloc[0]["year"] == 2023
    assert df.iloc[0]["search_engine_internal_id"] == "abc123"
    assert df.iloc[0]["doi"] == "10.1234/test"
    assert df.iloc[0]["authors"] == "Alice; Bob"
    assert "language" in df.columns
    assert df.iloc[0]["language"] == "en"


def test_search_run_fails_if_files_exist(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """If result files already exist, raises SearchResultsAlreadyExistException."""
    os.makedirs(projects_dir / "test-exp" / "searches", exist_ok=True)
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    # First run to create the files
    result1 = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result1.exit_code == 0

    searches_dir = projects_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("*.yaml"))
    assert yaml_files, "Expected a search YAML file to have been created"

    # Second run with same YAML file → should fail because output files exist
    result2 = runner.invoke(app, ["search", "run", "--file", yaml_files[0].name])

    assert result2.exit_code != 0
    assert isinstance(result2.exception, SearchResultsAlreadyExistException)


def test_search_run_rewrite_overwrites(projects_dir, mock_no_dotenv, mock_s2_session, monkeypatch):
    """--rewrite allows overwriting existing result files."""
    os.makedirs(projects_dir / "test-exp" / "searches", exist_ok=True)
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result1 = runner.invoke(app, ["search", "run", "--query", "test query"])
    assert result1.exit_code == 0

    searches_dir = projects_dir / "test-exp" / "searches"
    yaml_files = list(searches_dir.glob("*.yaml"))

    result2 = runner.invoke(
        app,
        ["search", "run", "--file", yaml_files[0].name, "--rewrite"],
    )

    assert result2.exit_code == 0
    assert "File with results" in result2.output


def test_search_run_language_unknown_on_empty(projects_dir, mock_no_dotenv, mock_openalex_session, monkeypatch):
    """Paper with no title and no abstract gets language='unknown'."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    empty_paper = {
        "paperId": "empty1",
        "title": None,
        "authors": [],
        "abstract": None,
        "isOpenAccess": False,
        "externalIds": {},
        "year": 2024,
    }
    mock_session = Mock()
    mock_session.get.return_value = _make_mock_s2_response([empty_paper])
    with patch(
        "llmexer.base.search_semantic_scholar.make_http_session",
        return_value=mock_session,
    ):
        with patch(
            "llmexer.base.search_semantic_scholar.detect_publication_lang",
            return_value="unknown",
        ):
            result = runner.invoke(app, ["search", "run", "--query", "test"])

    assert result.exit_code == 0
    searches_dir = projects_dir / "test-exp" / "searches"
    csv_files = list(searches_dir.glob("*__results.csv"))
    assert len(csv_files) == 1
    df = pd.read_csv(csv_files[0], sep=";")
    assert df.iloc[0]["language"] == "unknown"


def test_search_run_dry_run_no_files(projects_dir, mock_no_dotenv, monkeypatch):
    """With --dry-run, no output files are written."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["--dry-run", "search", "run", "--query", "test query"])

    assert result.exit_code == 0
    assert "Dry run" in result.output

    searches_dir = projects_dir / "test-exp" / "searches"
    if searches_dir.exists():
        assert not list((searches_dir / "jsons").glob("*__results_raw.json"))
        assert not list(searches_dir.glob("*__results.csv"))
