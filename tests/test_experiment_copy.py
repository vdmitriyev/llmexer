"""Tests for the `experiment copy-papers` and `experiment copy-search` commands."""

import os
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def _project(projects_dir, pid):
    """Create ``<pid>/experiment`` and ``<pid>/papers`` and return their paths."""
    exp_subdir = projects_dir / pid / "experiment"
    papers_dir = projects_dir / pid / "papers"
    searches_dir = projects_dir / pid / "searches"
    os.makedirs(exp_subdir)
    os.makedirs(papers_dir)
    os.makedirs(searches_dir)
    return exp_subdir, papers_dir, searches_dir


# ---------------------------------------------------------------------------
# copy-papers
# ---------------------------------------------------------------------------


def test_copy_papers_writes_rows_ordered_md_preferred(projects_dir):
    pid = "papers-exp"
    exp_subdir, papers_dir, _ = _project(projects_dir, pid)
    (papers_dir / "b.md").write_text("Body of B", encoding="utf-8")
    (papers_dir / "a.txt").write_text("Body of A", encoding="utf-8")
    # 'c' has both — .md must win.
    (papers_dir / "c.md").write_text("Body of C (md)", encoding="utf-8")
    (papers_dir / "c.txt").write_text("Body of C (txt)", encoding="utf-8")
    (papers_dir / "ignored.pdf").write_text("binary", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "copy-papers", "--pid", pid])

    assert result.exit_code == 0, result.exception
    df = pd.read_csv(exp_subdir / "data.csv", sep=";", encoding="utf-8")
    assert list(df.columns) == ["ID", "filename", "content"]
    assert list(df["ID"]) == ["P01", "P02", "P03"]
    assert list(df["filename"]) == ["a.txt", "b.md", "c.md"]
    assert list(df["content"]) == ["Body of A", "Body of B", "Body of C (md)"]


def test_copy_papers_content_with_special_chars_roundtrips(projects_dir):
    pid = "papers-special"
    exp_subdir, papers_dir, _ = _project(projects_dir, pid)
    body = 'Title; with semicolon\nand a newline and "quotes".'
    (papers_dir / "paper.md").write_text(body, encoding="utf-8")

    result = runner.invoke(app, ["experiment", "copy-papers", "--pid", pid])

    assert result.exit_code == 0, result.exception
    df = pd.read_csv(exp_subdir / "data.csv", sep=";", encoding="utf-8")
    assert df.iloc[0]["content"] == body


def test_copy_papers_backs_up_existing_data_csv(projects_dir):
    pid = "papers-backup"
    exp_subdir, papers_dir, _ = _project(projects_dir, pid)
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;old;old\n", encoding="utf-8"
    )
    (papers_dir / "a.md").write_text("Body of A", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "copy-papers", "--pid", pid])

    assert result.exit_code == 0, result.exception
    backups = list(exp_subdir.glob("data_backup_*.csv"))
    assert len(backups) == 1
    assert "old" in backups[0].read_text(encoding="utf-8")
    # New data.csv holds the copied paper.
    df = pd.read_csv(exp_subdir / "data.csv", sep=";", encoding="utf-8")
    assert list(df["filename"]) == ["a.md"]


def test_copy_papers_no_parsed_files_warns_and_skips(projects_dir):
    pid = "papers-empty"
    exp_subdir, papers_dir, _ = _project(projects_dir, pid)
    (papers_dir / "only.pdf").write_text("binary", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "copy-papers", "--pid", pid])

    assert result.exit_code == 0, result.exception
    assert "nothing to copy" in result.output.lower()
    assert not (exp_subdir / "data.csv").exists()


def test_copy_papers_missing_papers_folder_raises(projects_dir):
    pid = "papers-nofolder"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)  # no papers/ folder

    result = runner.invoke(app, ["experiment", "copy-papers", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


# ---------------------------------------------------------------------------
# copy-search
# ---------------------------------------------------------------------------


def _write_search_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, sep=";", encoding="utf-8")


def test_copy_search_writes_rows_in_order(projects_dir):
    pid = "search-exp"
    exp_subdir, _, searches_dir = _project(projects_dir, pid)
    _write_search_csv(
        searches_dir / "s1_results.csv",
        [
            {
                "year": 2023,
                "title": "First Paper",
                "authors": "Alice; Bob",
                "abstract": "Abstract one; with semicolon.",
                "doi": "10.1/one",
            },
            {
                "year": 2022,
                "title": "Second Paper",
                "authors": "Carol",
                "abstract": "Abstract two.",
                "doi": "",  # missing DOI
            },
        ],
    )

    result = runner.invoke(
        app, ["experiment", "copy-search", "--pid", pid, "--file", "s1_results.csv"]
    )

    assert result.exit_code == 0, result.exception
    df = pd.read_csv(exp_subdir / "data.csv", sep=";", encoding="utf-8").fillna("")
    assert list(df.columns) == ["ID", "Title", "Abstract", "doi", "authors"]
    assert list(df["ID"]) == ["S01", "S02"]
    assert list(df["Title"]) == ["First Paper", "Second Paper"]
    assert df.iloc[0]["Abstract"] == "Abstract one; with semicolon."
    assert df.iloc[0]["doi"] == "10.1/one"
    assert df.iloc[1]["doi"] == ""  # empty, not "nan"
    assert df.iloc[0]["authors"] == "Alice; Bob"


def test_copy_search_backs_up_existing_data_csv(projects_dir):
    pid = "search-backup"
    exp_subdir, _, searches_dir = _project(projects_dir, pid)
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;old;old\n", encoding="utf-8"
    )
    _write_search_csv(
        searches_dir / "s1_results.csv",
        [{"title": "T", "authors": "A", "abstract": "Ab", "doi": "10.1/x"}],
    )

    result = runner.invoke(
        app, ["experiment", "copy-search", "--pid", pid, "--file", "s1_results.csv"]
    )

    assert result.exit_code == 0, result.exception
    backups = list(exp_subdir.glob("data_backup_*.csv"))
    assert len(backups) == 1
    assert "old" in backups[0].read_text(encoding="utf-8")


def test_copy_search_missing_file_raises(projects_dir):
    pid = "search-missing"
    _project(projects_dir, pid)

    result = runner.invoke(
        app, ["experiment", "copy-search", "--pid", pid, "--file", "nope.csv"]
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_copy_search_missing_column_raises(projects_dir):
    pid = "search-badcols"
    _, _, searches_dir = _project(projects_dir, pid)
    # Missing 'authors' column.
    _write_search_csv(
        searches_dir / "s1_results.csv",
        [{"title": "T", "abstract": "Ab", "doi": "10.1/x"}],
    )

    result = runner.invoke(
        app, ["experiment", "copy-search", "--pid", pid, "--file", "s1_results.csv"]
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "authors" in str(result.exception)
