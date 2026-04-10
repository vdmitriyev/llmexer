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
def experiment_with_search(experiments_dir, monkeypatch):
    """Create an experiment with a search YAML and a _results.csv."""
    eid = "20260409-test-filter"
    exp_path = experiments_dir / eid
    searches_path = exp_path / "searches"
    os.makedirs(searches_path)
    monkeypatch.setenv("EXPERIMENT_ID", eid)

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
                "pdf_filename": "2023_English_Paper_10.1_en1.pdf",
                "pdf_downloaded": False,
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
                "pdf_filename": "2021_Another_English_Paper_10.1_en2.pdf",
                "pdf_downloaded": False,
            },
        ]
    )
    csv_path = searches_path / f"{search_id}_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

    return eid, search_id, searches_path


def test_search_filter_happy_path(
    experiments_dir, mock_no_dotenv, experiment_with_search
):
    """Default --language en filters to only English papers."""
    eid, search_id, searches_path = experiment_with_search

    result = runner.invoke(
        app, ["search", "filter", "--eid", eid, "--file", f"{search_id}.yaml"]
    )

    assert result.exit_code == 0
    assert "en" in result.output
    assert "Total:" in result.output
    assert "Filtered out:" in result.output
    assert "Remaining:" in result.output

    filtered_path = searches_path / f"{search_id}_filtered.csv"
    assert filtered_path.exists()

    df = pd.read_csv(filtered_path, sep=";")
    assert len(df) == 2
    assert all(df["language"] == "en")


def test_search_filter_custom_language(
    experiments_dir, mock_no_dotenv, experiment_with_search
):
    """--language de filters to only German papers."""
    eid, search_id, searches_path = experiment_with_search

    result = runner.invoke(
        app,
        [
            "search",
            "filter",
            "--eid",
            eid,
            "--file",
            f"{search_id}.yaml",
            "--language",
            "de",
        ],
    )

    assert result.exit_code == 0
    filtered_path = searches_path / f"{search_id}_filtered.csv"
    assert filtered_path.exists()

    df = pd.read_csv(filtered_path, sep=";")
    assert len(df) == 1
    assert df.iloc[0]["language"] == "de"


def test_search_filter_no_file_raises(experiments_dir, mock_no_dotenv, monkeypatch):
    """Omitting --file raises UnexpectedCLIParamsException."""
    os.makedirs(experiments_dir / "test-exp")
    monkeypatch.setenv("EXPERIMENT_ID", "test-exp")

    result = runner.invoke(app, ["search", "filter"])
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_search_filter_missing_csv_warns(
    experiments_dir, mock_no_dotenv, experiment_with_search
):
    """When _results.csv is missing, exits 0 with a warning."""
    eid, search_id, searches_path = experiment_with_search
    (searches_path / f"{search_id}_results.csv").unlink()

    result = runner.invoke(
        app, ["search", "filter", "--eid", eid, "--file", f"{search_id}.yaml"]
    )

    assert result.exit_code == 0
    assert "Warning" in result.output
    assert not (searches_path / f"{search_id}_filtered.csv").exists()


def test_search_filter_dry_run(experiments_dir, mock_no_dotenv, experiment_with_search):
    """With --dry-run, no _filtered.csv is written."""
    eid, search_id, searches_path = experiment_with_search

    result = runner.invoke(
        app,
        ["--dry-run", "search", "filter", "--eid", eid, "--file", f"{search_id}.yaml"],
    )

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert not (searches_path / f"{search_id}_filtered.csv").exists()


def test_search_filter_no_matches(
    experiments_dir, mock_no_dotenv, experiment_with_search
):
    """When no rows match, _filtered.csv is written with header only (0 data rows)."""
    eid, search_id, searches_path = experiment_with_search

    result = runner.invoke(
        app,
        [
            "search",
            "filter",
            "--eid",
            eid,
            "--file",
            f"{search_id}.yaml",
            "--language",
            "zh",
        ],
    )

    assert result.exit_code == 0
    filtered_path = searches_path / f"{search_id}_filtered.csv"
    assert filtered_path.exists()

    df = pd.read_csv(filtered_path, sep=";")
    assert len(df) == 0
