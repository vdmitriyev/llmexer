"""Tests for the `papers extract` command."""

import os
from io import BytesIO
from unittest.mock import Mock, patch

import pypdf
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.commands.papers as papers_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(papers_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def experiment(experiments_dir):
    """Create a test experiment directory and return (eid, exp_path)."""
    eid = "20260402-test-extract"
    exp_path = experiments_dir / eid
    os.makedirs(exp_path)
    return eid, exp_path


def _make_pdf(text: str) -> bytes:
    """Create a minimal PDF with one page containing the given text."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_no_eid(experiments_dir, mock_no_dotenv, monkeypatch):
    """When --eid is not provided and EXPERIMENT_ID is not set, raises ExperimentIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["papers", "extract"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)


def test_extract_experiment_not_exists(experiments_dir, mock_no_dotenv, monkeypatch):
    """When experiment does not exist, raises ExperimentNotExistsException."""
    monkeypatch.setenv("EXPERIMENT_ID", "nonexistent-exp")

    result = runner.invoke(app, ["papers", "extract"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)


def test_extract_no_papers_dir(experiments_dir, mock_no_dotenv, experiment):
    """When experiment exists but has no papers/ dir, exits 0 with a warning."""
    eid, _ = experiment

    result = runner.invoke(app, ["papers", "extract", "--eid", eid])
    assert result.exit_code == 0
    assert "Warning" in result.output


def test_extract_no_pdfs(experiments_dir, mock_no_dotenv, experiment):
    """When papers/ dir exists but has no PDFs, exits 0 with a warning."""
    eid, exp_path = experiment
    os.makedirs(exp_path / "papers")

    result = runner.invoke(app, ["papers", "extract", "--eid", eid])
    assert result.exit_code == 0
    assert "Warning" in result.output


def test_extract_happy_path(experiments_dir, mock_no_dotenv, experiment):
    """When valid PDFs exist, produces .txt and .md files alongside them."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    pdf_bytes = _make_pdf("Hello extraction world")
    pdf_file = papers_path / "mypaper.pdf"
    pdf_file.write_bytes(pdf_bytes)

    result = runner.invoke(app, ["papers", "extract", "--eid", eid])
    assert result.exit_code == 0
    assert (papers_path / "mypaper.txt").exists()
    assert (papers_path / "mypaper.md").exists()
    assert "Extracted" in result.output
    assert "Done:" in result.output


def test_extract_dry_run(experiments_dir, mock_no_dotenv, experiment):
    """With --dry-run, no output files are written."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    pdf_bytes = _make_pdf("dry run content")
    pdf_file = papers_path / "dryrun.pdf"
    pdf_file.write_bytes(pdf_bytes)

    result = runner.invoke(app, ["--dry-run", "papers", "extract", "--eid", eid])
    assert result.exit_code == 0
    assert not (papers_path / "dryrun.txt").exists()
    assert not (papers_path / "dryrun.md").exists()


def test_extract_pdf_failure(experiments_dir, mock_no_dotenv, experiment):
    """When pypdf raises an exception, the PDF is skipped with a warning and exit code is 0."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    bad_pdf = papers_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a real pdf")

    with patch(
        "llmexer.commands.papers.pypdf.PdfReader", side_effect=Exception("corrupt")
    ):
        result = runner.invoke(app, ["papers", "extract", "--eid", eid])

    assert result.exit_code == 0
    assert "Skipped" in result.output
    assert not (papers_path / "bad.txt").exists()
    assert not (papers_path / "bad.md").exists()
