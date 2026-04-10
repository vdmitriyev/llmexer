"""Tests for the `search sync` command."""

import os

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.commands.search import _PAPER_CSV_COLUMNS
from llmexer.exceptions import UnexpectedCLIParamsException

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    from unittest.mock import Mock

    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def experiment(experiments_dir, monkeypatch):
    """Create a minimal experiment directory tree and set EXPERIMENT_ID."""
    eid = "test-exp"
    exp_path = experiments_dir / eid
    os.makedirs(exp_path / "searches", exist_ok=True)
    os.makedirs(exp_path / "papers", exist_ok=True)
    monkeypatch.setenv("EXPERIMENT_ID", eid)
    return eid, exp_path


def _write_search_yaml(exp_path, search_id):
    """Write a minimal search YAML and return its filename."""
    yaml_filename = f"{search_id}.yaml"
    yaml_path = exp_path / "searches" / yaml_filename
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"query": "test query", "year": "2020-2025", "onlyOpenAccess": False}, f
        )
    return yaml_filename


def _write_results_csv(exp_path, search_id, rows):
    """Write a results CSV with the given rows (list of dicts)."""
    csv_path = exp_path / "searches" / f"{search_id}_results.csv"
    df = pd.DataFrame(rows, columns=_PAPER_CSV_COLUMNS)
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")
    return csv_path


def _blank_row(**overrides):
    """Return a row dict with all columns blank, optionally overriding some."""
    row = {col: "" for col in _PAPER_CSV_COLUMNS}
    row["pdf_downloaded"] = False
    row.update(overrides)
    return row


SEARCH_ID = "20260410-aabbccdd"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_updates_pdf_downloaded(experiments_dir, mock_no_dotenv, experiment):
    """If pdf_filename exists in papers/, pdf_downloaded must be set to True."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    pdf_name = "2023_Smith_Test_Paper_10.1234_test.pdf"
    _write_results_csv(
        exp_path, SEARCH_ID, [_blank_row(pdf_filename=pdf_name, pdf_downloaded=False)]
    )

    # Create the PDF in papers/
    (exp_path / "papers" / pdf_name).write_bytes(b"%PDF-1.4")

    result = runner.invoke(app, ["search", "sync", "--file", yaml_filename])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(exp_path / "searches" / f"{SEARCH_ID}_results.csv", sep=";")
    assert df.iloc[0]["pdf_downloaded"] == True


def test_sync_updates_txt_filename(experiments_dir, mock_no_dotenv, experiment):
    """If a .txt file matching the pdf stem exists in papers/, txt_filename is updated."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    pdf_name = "2023_Smith_Test_Paper_10.1234_test.pdf"
    stem = "2023_Smith_Test_Paper_10.1234_test"
    _write_results_csv(exp_path, SEARCH_ID, [_blank_row(pdf_filename=pdf_name)])

    (exp_path / "papers" / pdf_name).write_bytes(b"%PDF-1.4")
    (exp_path / "papers" / f"{stem}.txt").write_text("extracted text")

    result = runner.invoke(app, ["search", "sync", "--file", yaml_filename])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(exp_path / "searches" / f"{SEARCH_ID}_results.csv", sep=";")
    assert df.iloc[0]["txt_filename"] == f"{stem}.txt"


def test_sync_updates_markdown_filename(experiments_dir, mock_no_dotenv, experiment):
    """If a .md file matching the pdf stem exists in papers/, markdown_filename is updated."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    pdf_name = "2023_Smith_Test_Paper_10.1234_test.pdf"
    stem = "2023_Smith_Test_Paper_10.1234_test"
    _write_results_csv(exp_path, SEARCH_ID, [_blank_row(pdf_filename=pdf_name)])

    (exp_path / "papers" / pdf_name).write_bytes(b"%PDF-1.4")
    (exp_path / "papers" / f"{stem}.md").write_text("# extracted markdown")

    result = runner.invoke(app, ["search", "sync", "--file", yaml_filename])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(exp_path / "searches" / f"{SEARCH_ID}_results.csv", sep=";")
    assert df.iloc[0]["markdown_filename"] == f"{stem}.md"


def test_sync_adds_new_pdf(experiments_dir, mock_no_dotenv, experiment):
    """A PDF in papers/ that is not in the CSV must be added as a new row."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    _write_results_csv(
        exp_path, SEARCH_ID, [_blank_row(pdf_filename="existing_paper.pdf")]
    )
    (exp_path / "papers" / "existing_paper.pdf").write_bytes(b"%PDF-1.4")

    new_pdf = "2024_Jones_New_Paper_NO_doi.pdf"
    (exp_path / "papers" / new_pdf).write_bytes(b"%PDF-1.4")

    result = runner.invoke(app, ["search", "sync", "--file", yaml_filename])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(exp_path / "searches" / f"{SEARCH_ID}_results.csv", sep=";")
    new_rows = df[df["pdf_filename"] == new_pdf]
    assert len(new_rows) == 1
    assert new_rows.iloc[0]["entry_source"] == "manually added"
    assert new_rows.iloc[0]["pdf_downloaded"] == True
    assert "New rows added: 1" in result.output


def test_sync_new_pdf_with_txt_and_md(experiments_dir, mock_no_dotenv, experiment):
    """A newly discovered PDF whose .txt and .md companions exist should have those fields set."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    _write_results_csv(exp_path, SEARCH_ID, [])

    pdf_name = "2024_Jones_New_Paper_NO_doi.pdf"
    stem = "2024_Jones_New_Paper_NO_doi"
    (exp_path / "papers" / pdf_name).write_bytes(b"%PDF-1.4")
    (exp_path / "papers" / f"{stem}.txt").write_text("text content")
    (exp_path / "papers" / f"{stem}.md").write_text("# md content")

    result = runner.invoke(app, ["search", "sync", "--file", yaml_filename])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(exp_path / "searches" / f"{SEARCH_ID}_results.csv", sep=";")
    row = df[df["pdf_filename"] == pdf_name].iloc[0]
    assert row["txt_filename"] == f"{stem}.txt"
    assert row["markdown_filename"] == f"{stem}.md"


def test_sync_updates_filtered_csv(experiments_dir, mock_no_dotenv, experiment):
    """When a filtered CSV also exists, it must be synced too."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    pdf_name = "2023_Smith_Test_Paper_10.1234_test.pdf"
    row = _blank_row(pdf_filename=pdf_name, pdf_downloaded=False)
    _write_results_csv(exp_path, SEARCH_ID, [row])

    # Also write a filtered CSV with the same row
    filtered_path = exp_path / "searches" / f"{SEARCH_ID}_filtered.csv"
    df_filtered = pd.DataFrame([row], columns=_PAPER_CSV_COLUMNS)
    df_filtered.to_csv(filtered_path, index=False, encoding="utf-8", sep=";")

    (exp_path / "papers" / pdf_name).write_bytes(b"%PDF-1.4")

    result = runner.invoke(app, ["search", "sync", "--file", yaml_filename])

    assert result.exit_code == 0, result.output

    df_results = pd.read_csv(
        exp_path / "searches" / f"{SEARCH_ID}_results.csv", sep=";"
    )
    assert df_results.iloc[0]["pdf_downloaded"] == True

    df_filtered_after = pd.read_csv(filtered_path, sep=";")
    assert df_filtered_after.iloc[0]["pdf_downloaded"] == True


def test_sync_missing_file_raises(experiments_dir, mock_no_dotenv, experiment):
    """Calling sync without --file must raise UnexpectedCLIParamsException."""
    result = runner.invoke(app, ["search", "sync"])
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_sync_dry_run_no_writes(experiments_dir, mock_no_dotenv, experiment):
    """With --dry-run the CSV files must not be modified."""
    eid, exp_path = experiment
    yaml_filename = _write_search_yaml(exp_path, SEARCH_ID)

    pdf_name = "2023_Smith_Test_Paper_10.1234_test.pdf"
    _write_results_csv(
        exp_path, SEARCH_ID, [_blank_row(pdf_filename=pdf_name, pdf_downloaded=False)]
    )
    (exp_path / "papers" / pdf_name).write_bytes(b"%PDF-1.4")

    results_path = exp_path / "searches" / f"{SEARCH_ID}_results.csv"
    mtime_before = results_path.stat().st_mtime

    result = runner.invoke(
        app, ["--dry-run", "search", "sync", "--file", yaml_filename]
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert results_path.stat().st_mtime == mtime_before
