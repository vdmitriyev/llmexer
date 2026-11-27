"""Tests for the `papers find --missing` command."""

import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.base.papers import find_missing_papers
from llmexer.cli import app
from llmexer.commands.search import _PAPER_CSV_COLUMNS
from llmexer.exceptions import LLMExerException, UnexpectedCLIParamsException

runner = CliRunner()

PID = "test-exp"
SEARCH_FILE = "20260410-aabbccdd__filtered.csv"
PDF_PRESENT = "2023_Smith_Present_10.1234_a.pdf"
PDF_ABSENT = "2024_Jones_Absent_10.1234_b.pdf"


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))


@pytest.fixture()
def experiment(projects_dir, mock_no_dotenv, monkeypatch):
    exp_path = projects_dir / PID
    os.makedirs(exp_path / "searches", exist_ok=True)
    os.makedirs(exp_path / "papers", exist_ok=True)
    monkeypatch.setenv("PROJECT_ID", PID)
    (exp_path / "papers" / PDF_PRESENT).write_bytes(b"%PDF-1.4")
    return exp_path


def _row(**overrides):
    row = {col: "" for col in _PAPER_CSV_COLUMNS}
    row["pdf_downloaded"] = False
    row.update(overrides)
    return row


def _write_search_file(exp_path, rows):
    path = exp_path / "searches" / SEARCH_FILE
    pd.DataFrame(rows, columns=_PAPER_CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8", sep=";")
    return path


def _three_rows():
    return [
        _row(year=2023, title="Present", doi="10.1234/a", pdf_filename=PDF_PRESENT),
        # pdf_downloaded says True, but the file is gone: still missing.
        _row(year=2024, title="Absent", doi="10.1234/b", pdf_filename=PDF_ABSENT, pdf_downloaded=True),
        _row(year=2025, title="No file name", doi="10.1234/c"),
    ]


def test_find_missing_lists_absent_papers(experiment):
    csv_path = _write_search_file(experiment, _three_rows())
    csv_before = csv_path.read_bytes()
    papers_before = sorted(os.listdir(experiment / "papers"))

    result = runner.invoke(app, ["papers", "find", "--missing", "--search-file", SEARCH_FILE])

    assert result.exit_code == 0, result.output
    assert "Absent" in result.output
    assert "No file name" in result.output
    assert "Present" not in result.output
    assert "2 of 3 paper(s) missing" in result.output
    assert f"llmexer papers download --pid {PID} --search-file {SEARCH_FILE}" in result.output
    assert csv_path.read_bytes() == csv_before
    assert sorted(os.listdir(experiment / "papers")) == papers_before


def test_find_missing_filename_only(experiment):
    _write_search_file(experiment, _three_rows())

    result = runner.invoke(app, ["papers", "find", "--missing", "--filename-only", "--search-file", SEARCH_FILE])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [PDF_ABSENT]


def test_find_missing_doi_only(experiment):
    _write_search_file(experiment, _three_rows())

    result = runner.invoke(app, ["papers", "find", "--missing", "--doi-only", "--search-file", SEARCH_FILE])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["10.1234/b", "10.1234/c"]


def test_find_missing_doi_only_prints_nothing_when_complete(experiment):
    _write_search_file(experiment, [_row(title="Present", doi="10.1234/a", pdf_filename=PDF_PRESENT)])

    result = runner.invoke(app, ["papers", "find", "--missing", "--doi-only", "--search-file", SEARCH_FILE])

    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_find_missing_filename_only_and_doi_only_conflict(experiment):
    _write_search_file(experiment, _three_rows())

    result = runner.invoke(
        app, ["papers", "find", "--missing", "--filename-only", "--doi-only", "--search-file", SEARCH_FILE]
    )

    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_find_missing_nothing_missing(experiment):
    _write_search_file(experiment, [_row(title="Present", pdf_filename=PDF_PRESENT)])

    result = runner.invoke(app, ["papers", "find", "--missing", "--search-file", SEARCH_FILE])

    assert result.exit_code == 0, result.output
    assert f"All 1 paper(s) of {SEARCH_FILE} are in papers/." in result.output


def test_find_requires_search_file(experiment):
    result = runner.invoke(app, ["papers", "find", "--missing"])

    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_find_requires_missing(experiment):
    _write_search_file(experiment, _three_rows())

    result = runner.invoke(app, ["papers", "find", "--search-file", SEARCH_FILE])

    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_find_unknown_search_file(experiment):
    result = runner.invoke(app, ["papers", "find", "--missing", "--search-file", "nope__filtered.csv"])

    assert isinstance(result.exception, LLMExerException)


def test_find_missing_papers_helper(tmp_path):
    (tmp_path / PDF_PRESENT).write_bytes(b"%PDF-1.4")
    df = pd.DataFrame(_three_rows(), columns=_PAPER_CSV_COLUMNS)

    missing_df = find_missing_papers(df, str(tmp_path))

    assert list(missing_df["title"]) == ["Absent", "No file name"]
