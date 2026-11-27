"""Tests for the `papers remove` and `search remove` commands."""

import os
import re

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.commands.search import _PAPER_CSV_COLUMNS
from llmexer.exceptions import PaperRemoveException, UnexpectedCLIParamsException

runner = CliRunner()

SEARCH_ID = "20260410-aabbccdd"
PID = "test-exp"
PDF_A = "2023_Smith_Paper_A_10.1234_a.pdf"
PDF_B = "2024_Jones_Paper_B_10.1234_b.pdf"
DOI_A = "10.1234/A"
DOI_B = "10.1234/b"


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
    """A project with one search (two rows) and paper A's PDF, TXT and MD in papers/."""
    exp_path = projects_dir / PID
    os.makedirs(exp_path / "searches", exist_ok=True)
    os.makedirs(exp_path / "papers", exist_ok=True)
    monkeypatch.setenv("PROJECT_ID", PID)

    rows = [
        _row(doi=DOI_A, title="Paper A", pdf_filename=PDF_A, pdf_downloaded=True),
        _row(doi=DOI_B, title="Paper B", pdf_filename=PDF_B),
    ]
    _write_csv(exp_path / "searches" / f"{SEARCH_ID}__results.csv", rows)
    (exp_path / "searches" / f"{SEARCH_ID}.yaml").write_text(
        "query: test query\nyear: 2020-2025\nonlyOpenAccess: false\n"
    )

    stem_a = PDF_A[:-4]
    (exp_path / "papers" / PDF_A).write_bytes(b"%PDF-1.4")
    (exp_path / "papers" / f"{stem_a}.txt").write_text("text")
    (exp_path / "papers" / f"{stem_a}.md").write_text("markdown")
    return exp_path


def _row(**overrides):
    row = {col: "" for col in _PAPER_CSV_COLUMNS}
    row["pdf_downloaded"] = False
    row.update(overrides)
    return row


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=_PAPER_CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8", sep=";")


def _results(exp_path):
    return exp_path / "searches" / f"{SEARCH_ID}__results.csv"


def _filtered(exp_path):
    return exp_path / "searches" / f"{SEARCH_ID}__filtered.csv"


def _titles(path):
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    return list(df["title"])


def _paper_files(exp_path):
    return sorted(os.listdir(exp_path / "papers"))


@pytest.mark.parametrize("group", ["papers", "search"])
@pytest.mark.parametrize("command", ["remove", "delete"])
def test_remove_by_name_deletes_files_and_filtered_row(experiment, group, command):
    results_before = _results(experiment).read_bytes()

    result = runner.invoke(app, [group, command, "--name", PDF_A, "--yes"])

    assert result.exit_code == 0, result.output
    assert _paper_files(experiment) == []
    assert _titles(_filtered(experiment)) == ["Paper B"]
    assert _results(experiment).read_bytes() == results_before


def test_remove_by_doi_is_case_insensitive(experiment):
    result = runner.invoke(app, ["papers", "remove", "--doi", "10.1234/a", "--yes"])

    assert result.exit_code == 0, result.output
    assert _paper_files(experiment) == []
    assert _titles(_filtered(experiment)) == ["Paper B"]


def test_remove_row_without_file(experiment):
    result = runner.invoke(app, ["search", "remove", "--doi", DOI_B, "--yes"])

    assert result.exit_code == 0, result.output
    assert len(_paper_files(experiment)) == 3
    assert _titles(_filtered(experiment)) == ["Paper A"]


def test_remove_uses_existing_filtered_csv(experiment):
    _write_csv(_filtered(experiment), [_row(doi=DOI_A, title="Paper A", pdf_filename=PDF_A)])

    result = runner.invoke(app, ["papers", "remove", "--doi", DOI_A, "--yes"])

    assert result.exit_code == 0, result.output
    assert _titles(_filtered(experiment)) == []
    assert _titles(_results(experiment)) == ["Paper A", "Paper B"]


def test_remove_deletes_files_of_already_filtered_paper(experiment):
    _write_csv(_filtered(experiment), [_row(doi=DOI_B, title="Paper B", pdf_filename=PDF_B)])
    filtered_before = _filtered(experiment).read_bytes()

    result = runner.invoke(app, ["papers", "remove", "--doi", DOI_A, "--yes"])

    assert result.exit_code == 0, result.output
    assert _paper_files(experiment) == []
    assert _filtered(experiment).read_bytes() == filtered_before


def test_remove_rebuilds_merged_filtered_only(experiment):
    merged_filtered = experiment / "searches" / f"{PID}__merged_filtered.csv"
    merged_results = experiment / "searches" / f"{PID}__merged_results.csv"
    rows = [_row(doi=DOI_A, title="Paper A"), _row(doi=DOI_B, title="Paper B")]
    _write_csv(merged_filtered, rows)
    _write_csv(merged_results, rows)
    merged_results_before = merged_results.read_bytes()

    result = runner.invoke(app, ["papers", "remove", "--doi", DOI_A, "--yes"])

    assert result.exit_code == 0, result.output
    merged_df = pd.read_csv(merged_filtered, sep=";")
    assert list(merged_df["doi"]) == [DOI_B]
    assert merged_results.read_bytes() == merged_results_before


def test_remove_logs_like_search_filter(experiment):
    result = runner.invoke(app, ["papers", "remove", "--name", PDF_A, "--yes"])

    assert result.exit_code == 0, result.output
    assert f"Excluded name={PDF_A}" in result.output

    # Same pattern `analysis draw-prisma` reads the applied filters with.
    filters_log = (experiment / "searches" / "logs" / "filters-applied.log").read_text(encoding="utf-8")
    match = re.search(r"search file: (\S+); filter applied: (.*?) ; input rows", filters_log)
    assert match.group(1) == f"{SEARCH_ID}__filtered.csv"
    assert match.group(2) == f"name={PDF_A}"
    assert "input rows: 2; output rows: 1" in filters_log


def test_remove_leaves_unmatched_search_alone(experiment):
    other_id = "20260411-other"
    (experiment / "searches" / f"{other_id}.yaml").write_text("query: other\nyear: 2020-2025\nonlyOpenAccess: false\n")
    other_results = experiment / "searches" / f"{other_id}__results.csv"
    _write_csv(other_results, [_row(doi="10.5555/other", title="Other")])

    result = runner.invoke(app, ["papers", "remove", "--doi", DOI_A, "--yes"])

    assert result.exit_code == 0, result.output
    assert not (experiment / "searches" / f"{other_id}__filtered.csv").exists()
    filters_log = (experiment / "searches" / "logs" / "filters-applied.log").read_text(encoding="utf-8")
    assert other_id not in filters_log


def test_remove_unknown_identifier_changes_nothing(experiment):
    result = runner.invoke(app, ["papers", "remove", "--name", PDF_A, "--doi", "10.9999/none", "--yes"])

    assert isinstance(result.exception, PaperRemoveException)
    assert "10.9999/none" in str(result.exception)
    assert len(_paper_files(experiment)) == 3
    assert not _filtered(experiment).exists()


def test_remove_requires_an_identifier(experiment):
    result = runner.invoke(app, ["papers", "remove", "--yes"])

    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_remove_rejects_path_as_name(experiment):
    result = runner.invoke(app, ["papers", "remove", "--name", f"../{PDF_A}", "--yes"])

    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_remove_dry_run_changes_nothing(experiment):
    result = runner.invoke(app, ["--dry-run", "papers", "remove", "--name", PDF_A])

    assert result.exit_code == 0, result.output
    assert len(_paper_files(experiment)) == 3
    assert not _filtered(experiment).exists()
    assert not (experiment / "searches" / "logs").exists()


def test_remove_declined_confirmation_changes_nothing(experiment):
    result = runner.invoke(app, ["papers", "remove", "--name", PDF_A], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Nothing removed" in result.output
    assert len(_paper_files(experiment)) == 3
    assert not _filtered(experiment).exists()


def test_remove_confirmed_prompt_removes(experiment):
    result = runner.invoke(app, ["papers", "remove", "--name", PDF_A], input="y\n")

    assert result.exit_code == 0, result.output
    assert _paper_files(experiment) == []


@pytest.mark.parametrize("group", ["papers", "search"])
def test_delete_alias_is_hidden_from_help(experiment, group):
    result = runner.invoke(app, [group, "--help"])

    assert result.exit_code == 0, result.output
    assert "remove" in result.output
    assert "delete" not in result.output
