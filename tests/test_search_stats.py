"""Tests for the `search stats` merged-file fallback."""

import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.commands.search import (
    _PAPER_CSV_COLUMNS,
    MERGED_FILTERED_SUFFIX,
    MERGED_RESULTS_SUFFIX,
)
from llmexer.exceptions import UnexpectedCLIParamsException

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    from unittest.mock import Mock

    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def experiment(projects_dir, monkeypatch):
    """Create a minimal experiment directory tree and set PROJECT_ID."""
    pid = "test-exp"
    exp_path = projects_dir / pid
    os.makedirs(exp_path / "searches", exist_ok=True)
    os.makedirs(exp_path / "papers", exist_ok=True)
    monkeypatch.setenv("PROJECT_ID", pid)
    return pid, exp_path


def _row(**overrides):
    row = {col: "" for col in _PAPER_CSV_COLUMNS}
    row["pdf_downloaded"] = False
    row.update(overrides)
    return row


def _write_merged(exp_path, pid, rows, suffix):
    df = pd.DataFrame(rows)
    df.to_csv(
        exp_path / "searches" / f"{pid}{suffix}",
        index=False,
        encoding="utf-8",
        sep=";",
    )


def _sample_rows():
    return [
        {
            **_row(doi="10.1/a", title="A", year=2023, isOpenAccess=True),
            "duplicates_counter": 2,
        },
        {
            **_row(doi="10.1/b", title="B", year=2024, isOpenAccess=False),
            "duplicates_counter": 1,
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stats_uses_merged_results_when_no_file(
    projects_dir, mock_no_dotenv, experiment
):
    """With no --file and a merged results file present, stats reads it."""
    pid, exp_path = experiment
    _write_merged(exp_path, pid, _sample_rows(), MERGED_RESULTS_SUFFIX)

    result = runner.invoke(app, ["search", "stats"])
    assert result.exit_code == 0, result.output
    assert f"{pid}{MERGED_RESULTS_SUFFIX}" in result.output


def test_stats_shows_both_merged_files(projects_dir, mock_no_dotenv, experiment):
    """When both merged files exist, stats displays both."""
    pid, exp_path = experiment
    _write_merged(exp_path, pid, _sample_rows(), MERGED_RESULTS_SUFFIX)
    _write_merged(exp_path, pid, _sample_rows()[:1], MERGED_FILTERED_SUFFIX)

    result = runner.invoke(app, ["search", "stats"])
    assert result.exit_code == 0, result.output
    assert f"{pid}{MERGED_RESULTS_SUFFIX}" in result.output
    assert f"{pid}{MERGED_FILTERED_SUFFIX}" in result.output


def test_stats_uses_merged_filtered_only(projects_dir, mock_no_dotenv, experiment):
    """With only a merged filtered file, stats still works off it."""
    pid, exp_path = experiment
    _write_merged(exp_path, pid, _sample_rows(), MERGED_FILTERED_SUFFIX)

    result = runner.invoke(app, ["search", "stats"])
    assert result.exit_code == 0, result.output
    assert f"{pid}{MERGED_FILTERED_SUFFIX}" in result.output


def test_stats_no_file_no_merged_raises(projects_dir, mock_no_dotenv, experiment):
    """With no --file and no merged file, stats raises UnexpectedCLIParamsException."""
    result = runner.invoke(app, ["search", "stats"])
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)
