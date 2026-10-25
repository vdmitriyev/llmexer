"""Tests for the `papers list`, `papers view` and `papers stats` commands."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.base.papers import PaperSortBy, scan_papers
from llmexer.cli import app
from llmexer.exceptions import ProjectNotExistsException

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))


@pytest.fixture()
def project(projects_dir):
    """Create a test project with an empty papers folder."""
    pid = "test-exp"
    papers_path = projects_dir / pid / "papers"
    os.makedirs(papers_path)
    return pid, papers_path


@pytest.fixture()
def fake_ctime(monkeypatch):
    """Let tests choose each file's creation time: tests cannot set st_ctime directly.

    The ctime is taken from the file's mtime, which ``os.utime`` can set.
    """

    real_stat = Path.stat

    def _stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        values = list(result)
        values[9] = result.st_mtime  # st_ctime
        return os.stat_result(values)

    monkeypatch.setattr(Path, "stat", _stat)


def _write(folder, name, size=10, created=None):
    path = folder / name
    path.write_bytes(b"x" * size)
    if created is not None:
        os.utime(path, (created, created))
    return path


def _order(output, names):
    return sorted(names, key=output.index)


# --- scan_papers ---


def test_scan_groups_files_by_stem(project):
    _, papers_path = project
    _write(papers_path, "a.pdf", 100)
    _write(papers_path, "a.md", 20)
    _write(papers_path, "b.pdf", 50)
    _write(papers_path, "b.json", 5)

    entries = scan_papers(str(papers_path))

    assert [e.stem for e in entries] == ["a", "b"]
    assert entries[0].has_pdf and entries[0].has_md and not entries[0].has_txt
    assert entries[0].size("pdf") == 100
    assert entries[0].size("txt") is None
    assert entries[0].total_size == 120
    assert entries[1].other_types == ["json"]


def test_scan_missing_folder_is_empty(tmp_path):
    assert scan_papers(str(tmp_path / "nope")) == []


def test_scan_year_from_structured_name(project):
    _, papers_path = project
    _write(papers_path, "2024_Smith_Title_10.1000_x.pdf")
    _write(papers_path, "plain.pdf")

    years = {e.stem: e.year for e in scan_papers(str(papers_path))}

    assert years == {"2024_Smith_Title_10.1000_x": "2024", "plain": None}


def test_scan_sort_by_date(project, fake_ctime):
    _, papers_path = project
    _write(papers_path, "a.pdf", created=3_000)
    _write(papers_path, "b.pdf", created=1_000)
    _write(papers_path, "c.pdf", created=2_000)

    ascending = [e.stem for e in scan_papers(str(papers_path), PaperSortBy.date)]
    descending = [e.stem for e in scan_papers(str(papers_path), PaperSortBy.date, desc=True)]

    assert ascending == ["b", "c", "a"]
    assert descending == ["a", "c", "b"]


def test_scan_sort_by_size(project):
    _, papers_path = project
    _write(papers_path, "a.pdf", 300)
    _write(papers_path, "b.pdf", 100)
    _write(papers_path, "b.md", 900)
    _write(papers_path, "c.md", 200)

    ascending = [e.stem for e in scan_papers(str(papers_path), PaperSortBy.size)]
    descending = [e.stem for e in scan_papers(str(papers_path), PaperSortBy.size, desc=True)]

    # `b` sorts by its PDF (100), not its larger MD; `c` has no PDF and sorts by all its files.
    assert ascending == ["b", "c", "a"]
    assert descending == ["a", "c", "b"]


# --- papers list ---


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], ["Alpha.pdf", "beta.md", "gamma.pdf"]),
        (["--desc"], ["gamma.pdf", "beta.md", "Alpha.pdf"]),
        (["--sort-by", "date"], ["gamma.pdf", "Alpha.pdf", "beta.md"]),
        (["--sort-by", "date", "--desc"], ["beta.md", "Alpha.pdf", "gamma.pdf"]),
        (["--sort-by", "size"], ["beta.md", "gamma.pdf", "Alpha.pdf"]),
        (["--sort-by", "size", "--desc"], ["Alpha.pdf", "gamma.pdf", "beta.md"]),
    ],
)
def test_list_sorting(project, mock_no_dotenv, fake_ctime, args, expected):
    pid, papers_path = project
    _write(papers_path, "Alpha.pdf", 300, created=2_000)
    _write(papers_path, "beta.md", 100, created=3_000)
    _write(papers_path, "gamma.pdf", 200, created=1_000)

    result = runner.invoke(app, ["papers", "list", "--pid", pid, "--created", *args])

    assert result.exit_code == 0, result.output
    assert _order(result.output, expected) == expected
    assert "1970-01-01" in result.output


def test_list_plain_names_by_default(project, mock_no_dotenv):
    pid, papers_path = project
    _write(papers_path, "b.pdf")
    _write(papers_path, "a.md")

    result = runner.invoke(app, ["papers", "list", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["a.md", "b.pdf"]


def test_list_empty_folder(project, mock_no_dotenv):
    pid, _ = project
    result = runner.invoke(app, ["papers", "list", "--pid", pid])
    assert result.exit_code == 0
    assert "No papers found." in result.output


def test_list_missing_papers_folder(projects_dir, mock_no_dotenv):
    os.makedirs(projects_dir / "bare")
    result = runner.invoke(app, ["papers", "list", "--pid", "bare"])
    assert result.exit_code == 0
    assert "No papers found." in result.output


def test_list_missing_project_raises(projects_dir, mock_no_dotenv):
    result = runner.invoke(app, ["papers", "list", "--pid", "missing"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


# --- papers view ---


def test_view_one_row_per_paper(project, mock_no_dotenv):
    pid, papers_path = project
    _write(papers_path, "a.pdf", 2048)
    _write(papers_path, "a.md", 100)

    result = runner.invoke(app, ["papers", "view", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert "1 paper(s), 2 file(s)" in result.output
    assert "2.0 KB" in result.output
    assert "100 B" in result.output
    assert "sorted alphabetically, A to Z" in result.output


def test_view_sort_by_date_desc(project, mock_no_dotenv, fake_ctime):
    pid, papers_path = project
    _write(papers_path, "paper_old.pdf", created=1_000)
    _write(papers_path, "paper_new.pdf", created=5_000)
    _write(papers_path, "paper_mid.pdf", created=3_000)

    result = runner.invoke(app, ["papers", "view", "--pid", pid, "--sort-by", "date", "--desc"])

    assert result.exit_code == 0, result.output
    assert _order(result.output, ["paper_old", "paper_new", "paper_mid"]) == ["paper_new", "paper_mid", "paper_old"]
    assert "newest first" in result.output


@pytest.mark.parametrize("alias", ["show", "table"])
def test_view_aliases(project, mock_no_dotenv, alias):
    pid, papers_path = project
    _write(papers_path, "a.pdf")
    result = runner.invoke(app, ["papers", alias, "--pid", pid])
    assert result.exit_code == 0, result.output
    assert "Papers" in result.output


def test_view_created_is_the_pdf_time(project, mock_no_dotenv, fake_ctime):
    pid, papers_path = project
    _write(papers_path, "with_pdf.pdf", created=86_400)
    _write(papers_path, "with_pdf.md", created=0)
    _write(papers_path, "only_md.md", created=2 * 86_400)

    result = runner.invoke(app, ["papers", "view", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert "Created (PDF)" in result.output
    assert "1970-01-02 00:00:00 " in result.output
    assert "1970-01-02 00:00:00 PDF" not in result.output
    assert "1970-01-03 00:00:00 MD" in result.output


# --- papers stats ---


def test_stats_counts(project, mock_no_dotenv):
    pid, papers_path = project
    _write(papers_path, "2024_A_T_d.pdf")
    _write(papers_path, "2024_A_T_d.txt")
    _write(papers_path, "2023_B_T_d.pdf")
    _write(papers_path, "2023_B_T_d.md")
    _write(papers_path, "plain.pdf")

    result = runner.invoke(app, ["papers", "stats", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert "3 papers, 5 files" in result.output
    lines = result.output.splitlines()
    assert any("2024" in line and "33.3%" in line for line in lines)
    assert any("unknown" in line and "33.3%" in line for line in lines)
    assert any("TXT: existing" in line and " 1 " in line for line in lines)
    assert any("Markdown: missing" in line and " 2 " in line for line in lines)
    assert any("Text (TXT or MD): existing" in line and " 2 " in line for line in lines)
    assert "Files by Type" in result.output


def test_stats_empty_folder(project, mock_no_dotenv):
    pid, _ = project
    result = runner.invoke(app, ["papers", "stats", "--pid", pid])
    assert result.exit_code == 0
    assert "No papers found." in result.output


def test_view_other_files_in_own_table(project, mock_no_dotenv):
    pid, papers_path = project
    _write(papers_path, "a.pdf")
    _write(papers_path, "a.json", 2048)
    _write(papers_path, "notes.docx")

    result = runner.invoke(app, ["papers", "table", "--pid", pid])

    assert result.exit_code == 0, result.output
    papers_part, other_part = result.output.split("Other files")
    assert "1 paper(s), 1 file(s)" in papers_part
    assert "JSON" not in papers_part
    assert "notes" not in papers_part
    assert "a.json" in other_part and "JSON" in other_part and "2.0 KB" in other_part
    assert "notes.docx" in other_part and "DOCX" in other_part
    assert "2 file(s)" in other_part


def test_view_only_other_files(project, mock_no_dotenv):
    pid, papers_path = project
    _write(papers_path, "notes.docx")

    result = runner.invoke(app, ["papers", "table", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert "Other files" in result.output
    assert "paper(s)" not in result.output
