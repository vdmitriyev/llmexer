"""Tests for the `search merge` command."""

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
from llmexer.exceptions import (
    LLMExerException,
    SearchResultsAlreadyExistException,
)

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


def _write_csv(exp_path, filename, rows):
    """Write a search CSV (results or filtered) with the given rows (list of dicts)."""
    csv_path = exp_path / "searches" / filename
    df = pd.DataFrame(rows, columns=_PAPER_CSV_COLUMNS)
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")
    return csv_path


def _row(**overrides):
    """Return a row dict with all columns blank, optionally overriding some."""
    row = {col: "" for col in _PAPER_CSV_COLUMNS}
    row["pdf_downloaded"] = False
    row.update(overrides)
    return row


def _read_results(exp_path, pid):
    return pd.read_csv(exp_path / "searches" / f"{pid}{MERGED_RESULTS_SUFFIX}", sep=";")


def _read_filtered(exp_path, pid):
    return pd.read_csv(
        exp_path / "searches" / f"{pid}{MERGED_FILTERED_SUFFIX}", sep=";"
    )


SEARCH_A = "20260101-aaaaaaaa"
SEARCH_B = "20260102-bbbbbbbb"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_merge_results_dedups_by_doi(projects_dir, mock_no_dotenv, experiment):
    """A shared DOI collapses to one row; run columns and duplicates_counter are correct."""
    pid, exp_path = experiment
    _write_csv(
        exp_path,
        f"{SEARCH_A}_results.csv",
        [
            _row(doi="10.1/shared", title="Shared Paper"),
            _row(doi="10.1/only-a", title="Only A"),
        ],
    )
    _write_csv(
        exp_path,
        f"{SEARCH_B}_results.csv",
        [
            _row(doi="10.1/shared", title="Shared Paper"),
            _row(doi="10.1/only-b", title="Only B"),
        ],
    )

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    df = _read_results(exp_path, pid)
    assert len(df) == 3
    # Columns are named after the YAML search id, not the CSV filename.
    assert SEARCH_A in df.columns and SEARCH_B in df.columns
    assert "duplicates_counter" in df.columns

    shared = df[df["doi"] == "10.1/shared"].iloc[0]
    assert shared["duplicates_counter"] == 2
    assert shared[SEARCH_A] == 1 and shared[SEARCH_B] == 1

    only_a = df[df["doi"] == "10.1/only-a"].iloc[0]
    assert only_a["duplicates_counter"] == 1
    assert only_a[SEARCH_A] == 1 and only_a[SEARCH_B] == 0


def test_merge_produces_two_files_with_yaml_columns(
    projects_dir, mock_no_dotenv, experiment
):
    """merge writes both merged files; the filtered file's column is the search id (YAML stem)."""
    pid, exp_path = experiment
    _write_csv(exp_path, f"{SEARCH_A}_results.csv", [_row(doi="10.1/x", title="X")])
    _write_csv(exp_path, f"{SEARCH_A}_filtered.csv", [_row(doi="10.1/x", title="X")])

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    assert os.path.exists(exp_path / "searches" / f"{pid}{MERGED_RESULTS_SUFFIX}")
    assert os.path.exists(exp_path / "searches" / f"{pid}{MERGED_FILTERED_SUFFIX}")

    results_df = _read_results(exp_path, pid)
    filtered_df = _read_filtered(exp_path, pid)
    # Both reference the search by its YAML id, with no `_results`/`_filtered` suffix.
    assert SEARCH_A in results_df.columns
    assert SEARCH_A in filtered_df.columns
    assert f"{SEARCH_A}_filtered" not in filtered_df.columns
    assert filtered_df.iloc[0][SEARCH_A] == 1
    assert filtered_df.iloc[0]["duplicates_counter"] == 1


def test_merge_only_filtered_files(projects_dir, mock_no_dotenv, experiment):
    """With only _filtered.csv present, only the filtered merged file is produced."""
    pid, exp_path = experiment
    _write_csv(exp_path, f"{SEARCH_A}_filtered.csv", [_row(doi="10.1/x", title="X")])

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    assert not os.path.exists(exp_path / "searches" / f"{pid}{MERGED_RESULTS_SUFFIX}")
    assert os.path.exists(exp_path / "searches" / f"{pid}{MERGED_FILTERED_SUFFIX}")


def test_merge_case_insensitive_doi(projects_dir, mock_no_dotenv, experiment):
    """DOIs differing only in case are treated as the same publication."""
    pid, exp_path = experiment
    _write_csv(
        exp_path, f"{SEARCH_A}_results.csv", [_row(doi="10.1/AbC", title="Paper")]
    )
    _write_csv(
        exp_path, f"{SEARCH_B}_results.csv", [_row(doi="10.1/abc", title="Paper")]
    )

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    df = _read_results(exp_path, pid)
    assert len(df) == 1
    assert df.iloc[0]["duplicates_counter"] == 2


def test_merge_title_fallback(projects_dir, mock_no_dotenv, experiment):
    """Rows without a DOI dedup by normalized title."""
    pid, exp_path = experiment
    _write_csv(
        exp_path, f"{SEARCH_A}_results.csv", [_row(doi="", title="A Shared  Title")]
    )
    _write_csv(
        exp_path, f"{SEARCH_B}_results.csv", [_row(doi="", title="a shared title")]
    )

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    df = _read_results(exp_path, pid)
    assert len(df) == 1
    assert df.iloc[0]["duplicates_counter"] == 2


def test_merge_metadata_completed_from_duplicates(
    projects_dir, mock_no_dotenv, experiment
):
    """Missing metadata is filled from the first duplicate that has it."""
    pid, exp_path = experiment
    _write_csv(
        exp_path,
        f"{SEARCH_A}_results.csv",
        [_row(doi="10.1/x", title="X", abstract="")],
    )
    _write_csv(
        exp_path,
        f"{SEARCH_B}_results.csv",
        [_row(doi="10.1/x", title="X", abstract="Full abstract")],
    )

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    df = _read_results(exp_path, pid)
    assert df.iloc[0]["abstract"] == "Full abstract"


def test_merge_no_files_raises(projects_dir, mock_no_dotenv, experiment):
    """No search result files present raises LLMExerException."""
    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_merge_existing_without_rewrite_raises(
    projects_dir, mock_no_dotenv, experiment
):
    """An existing merged file without --rewrite raises SearchResultsAlreadyExistException."""
    pid, exp_path = experiment
    _write_csv(exp_path, f"{SEARCH_A}_results.csv", [_row(doi="10.1/x", title="X")])
    (exp_path / "searches" / f"{pid}{MERGED_RESULTS_SUFFIX}").write_text("stale")

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code != 0
    assert isinstance(result.exception, SearchResultsAlreadyExistException)


def test_merge_rewrite_overwrites(projects_dir, mock_no_dotenv, experiment):
    """--rewrite overwrites an existing merged file."""
    pid, exp_path = experiment
    _write_csv(exp_path, f"{SEARCH_A}_results.csv", [_row(doi="10.1/x", title="X")])
    (exp_path / "searches" / f"{pid}{MERGED_RESULTS_SUFFIX}").write_text("stale")

    result = runner.invoke(app, ["search", "merge", "--rewrite"])
    assert result.exit_code == 0, result.output

    df = _read_results(exp_path, pid)
    assert len(df) == 1
    assert df.iloc[0]["doi"] == "10.1/x"


def test_merge_dry_run_writes_nothing(projects_dir, mock_no_dotenv, experiment):
    """--dry-run does not write the merged files."""
    pid, exp_path = experiment
    _write_csv(exp_path, f"{SEARCH_A}_results.csv", [_row(doi="10.1/x", title="X")])

    result = runner.invoke(app, ["--dry-run", "search", "merge"])
    assert result.exit_code == 0, result.output
    assert not os.path.exists(exp_path / "searches" / f"{pid}{MERGED_RESULTS_SUFFIX}")


def test_merge_excludes_prior_merged_files(projects_dir, mock_no_dotenv, experiment):
    """A previously produced merged file is not itself re-merged."""
    pid, exp_path = experiment
    _write_csv(exp_path, f"{SEARCH_A}_results.csv", [_row(doi="10.1/x", title="X")])

    result = runner.invoke(app, ["search", "merge"])
    assert result.exit_code == 0, result.output

    # Re-run with --rewrite; the merged file must not create a run column for itself.
    result = runner.invoke(app, ["search", "merge", "--rewrite"])
    assert result.exit_code == 0, result.output
    df = _read_results(exp_path, pid)
    assert list(c for c in df.columns if "__merged" in c) == []
    assert df.columns.tolist().count(SEARCH_A) == 1
    assert len(df) == 1
