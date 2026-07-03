"""Tests for the `search filter` command."""

import os
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import UnexpectedCLIParamsException

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
def experiment_with_search(projects_dir, monkeypatch):
    """Create an experiment with a search YAML and a __results.csv."""
    pid = "20260409-test-filter"
    exp_path = projects_dir / pid
    searches_path = exp_path / "searches"
    os.makedirs(searches_path)
    monkeypatch.setenv("PROJECT_ID", pid)

    search_id = "20260409-aabbccdd"
    yaml_file = searches_path / f"{search_id}.yaml"
    yaml_file.write_text(
        yaml.dump(
            {"query": "test query", "year": "2020-2025", "onlyOpenAccess": False}
        ),
        encoding="utf-8",
    )

    df = pd.DataFrame(
        [
            {
                "sem_scholar_paper_id": "p1",
                "year": 2023,
                "title": "English Paper",
                "authors": "Alice",
                "abstract": "Abstract in English.",
                "isOpenAccess": True,
                "doi": "10.1/en1",
                "language": "en",
                "entry_source": "Semantic Scholar",
                "pdf_filename": "2023_English_Paper_10.1_en1.pdf",
                "pdf_downloaded": True,
            },
            {
                "sem_scholar_paper_id": "p2",
                "year": 2022,
                "title": "German Paper",
                "authors": "Bob",
                "abstract": "Zusammenfassung auf Deutsch.",
                "isOpenAccess": False,
                "doi": "10.1/de1",
                "language": "de",
                "entry_source": "Semantic Scholar",
                "pdf_filename": "2022_German_Paper_10.1_de1.pdf",
                "pdf_downloaded": False,
            },
            {
                "sem_scholar_paper_id": "p3",
                "year": 2021,
                "title": "Another English Paper",
                "authors": "Carol",
                "abstract": "More English content.",
                "isOpenAccess": True,
                "doi": "10.1/en2",
                "language": "en",
                "entry_source": "manually added",
                "pdf_filename": "2021_Another_English_Paper_10.1_en2.pdf",
                "pdf_downloaded": False,
            },
        ]
    )
    csv_path = searches_path / f"{search_id}__results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

    return pid, search_id, searches_path


def _invoke(pid, search_id, *extra):
    return runner.invoke(
        app,
        ["search", "filter", "--pid", pid, "--file", f"{search_id}.yaml", *extra],
    )


def test_filter_excludes_language(projects_dir, mock_no_dotenv, experiment_with_search):
    """--language de drops German rows (exclusion, not keep-matching)."""
    pid, search_id, searches_path = experiment_with_search

    result = _invoke(pid, search_id, "--language", "de")

    assert result.exit_code == 0, result.output
    df = pd.read_csv(searches_path / f"{search_id}__filtered.csv", sep=";")
    assert len(df) == 2
    assert "de" not in set(df["language"])


def test_filter_excludes_source(projects_dir, mock_no_dotenv, experiment_with_search):
    """--source drops rows whose entry_source equals the value."""
    pid, search_id, searches_path = experiment_with_search

    result = _invoke(pid, search_id, "--source", "manually added")

    assert result.exit_code == 0, result.output
    df = pd.read_csv(searches_path / f"{search_id}__filtered.csv", sep=";")
    assert len(df) == 2
    assert "manually added" not in set(df["entry_source"])


def test_filter_excludes_doi(projects_dir, mock_no_dotenv, experiment_with_search):
    """--doi drops the row with the exact DOI value."""
    pid, search_id, searches_path = experiment_with_search

    result = _invoke(pid, search_id, "--doi", "10.1/en1")

    assert result.exit_code == 0, result.output
    df = pd.read_csv(searches_path / f"{search_id}__filtered.csv", sep=";")
    assert len(df) == 2
    assert "10.1/en1" not in set(df["doi"])


def test_filter_excludes_not_downloaded(
    projects_dir, mock_no_dotenv, experiment_with_search
):
    """--downloaded keeps only rows that are downloaded."""
    pid, search_id, searches_path = experiment_with_search

    result = _invoke(pid, search_id, "--downloaded")

    assert result.exit_code == 0, result.output
    df = pd.read_csv(searches_path / f"{search_id}__filtered.csv", sep=";")
    assert len(df) == 1
    assert df.iloc[0]["sem_scholar_paper_id"] == "p1"


def test_filter_chains_on_existing_filtered(
    projects_dir, mock_no_dotenv, experiment_with_search
):
    """A second run reads the existing __filtered.csv (not the results) and rewrites it."""
    pid, search_id, searches_path = experiment_with_search

    # First filter: drop German → filtered has p1, p3.
    assert _invoke(pid, search_id, "--language", "de").exit_code == 0
    # Second filter: drop manually-added → must read the filtered file, leaving only p1.
    result = _invoke(pid, search_id, "--source", "manually added")

    assert result.exit_code == 0, result.output
    df = pd.read_csv(searches_path / f"{search_id}__filtered.csv", sep=";")
    assert len(df) == 1
    assert df.iloc[0]["sem_scholar_paper_id"] == "p1"


def test_filter_logs_applied(projects_dir, mock_no_dotenv, experiment_with_search):
    """Each applied filter appends a line to searches/logs/filters-applied.log."""
    pid, search_id, searches_path = experiment_with_search

    assert _invoke(pid, search_id, "--language", "de").exit_code == 0

    log_path = searches_path / "logs" / "filters-applied.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    assert (
        f"search file: {search_id}__filtered.csv; "
        "filter applied: language=de ; input rows: 3; output rows: 2"
    ) in content


def test_filter_no_criteria_raises(
    projects_dir, mock_no_dotenv, experiment_with_search
):
    """Supplying no filter criterion raises UnexpectedCLIParamsException."""
    pid, search_id, _ = experiment_with_search

    result = _invoke(pid, search_id)
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_filter_no_file_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """Omitting --file raises UnexpectedCLIParamsException."""
    os.makedirs(projects_dir / "test-exp")
    monkeypatch.setenv("PROJECT_ID", "test-exp")

    result = runner.invoke(app, ["search", "filter"])
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_filter_missing_source_warns(
    projects_dir, mock_no_dotenv, experiment_with_search
):
    """When neither filtered nor results CSV exists, exits 0 with a warning."""
    pid, search_id, searches_path = experiment_with_search
    (searches_path / f"{search_id}__results.csv").unlink()

    result = _invoke(pid, search_id, "--language", "de")

    assert result.exit_code == 0
    assert "Warning" in result.output
    assert not (searches_path / f"{search_id}__filtered.csv").exists()


def test_filter_dry_run(projects_dir, mock_no_dotenv, experiment_with_search):
    """With --dry-run, no __filtered.csv and no log are written."""
    pid, search_id, searches_path = experiment_with_search

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "search",
            "filter",
            "--pid",
            pid,
            "--file",
            f"{search_id}.yaml",
            "--language",
            "de",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert not (searches_path / f"{search_id}__filtered.csv").exists()
    assert not (searches_path / "logs" / "filters-applied.log").exists()
