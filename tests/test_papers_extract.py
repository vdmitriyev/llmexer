"""Tests for the `papers extract` command."""

import os
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

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
    """When valid PDFs exist, produces .txt files alongside them."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    pdf_bytes = _make_pdf("Hello extraction world")
    pdf_file = papers_path / "mypaper.pdf"
    pdf_file.write_bytes(pdf_bytes)

    with patch("llmexer.commands.papers.pypdf.PdfReader") as mock_reader:
        mock_page = Mock()
        mock_page.extract_text.return_value = "Hello extraction world"
        mock_reader.return_value.pages = [mock_page]

        result = runner.invoke(app, ["papers", "extract", "--eid", eid])

    assert result.exit_code == 0
    assert (papers_path / "mypaper.txt").exists()
    assert "extracted:" in result.output
    assert "Extracted:" in result.output


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


def test_extract_empty_content(experiments_dir, mock_no_dotenv, experiment):
    """When extracted text is empty (whitespace-only), warns and skips writing .txt file."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    pdf_bytes = _make_pdf("")
    pdf_file = papers_path / "empty.pdf"
    pdf_file.write_bytes(pdf_bytes)

    with patch("llmexer.commands.papers.pypdf.PdfReader") as mock_reader:
        mock_page = Mock()
        mock_page.extract_text.return_value = "   \n\t  "
        mock_reader.return_value.pages = [mock_page]

        result = runner.invoke(app, ["papers", "extract", "--eid", eid])

    assert result.exit_code == 0
    assert "warning:" in result.output
    assert "could not be parsed" in result.output
    assert not (papers_path / "empty.txt").exists()


# ---------------------------------------------------------------------------
# --rewrite flag tests (pypdf)
# ---------------------------------------------------------------------------


def test_extract_skips_existing_txt(experiments_dir, mock_no_dotenv, experiment):
    """Without --rewrite, a pre-existing .txt file is skipped."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    pdf_file = papers_path / "mypaper.pdf"
    pdf_file.write_bytes(_make_pdf("content"))
    txt_file = papers_path / "mypaper.txt"
    txt_file.write_text("original content", encoding="utf-8")

    with patch("llmexer.commands.papers.pypdf.PdfReader") as mock_reader:
        mock_page = Mock()
        mock_page.extract_text.return_value = "new content"
        mock_reader.return_value.pages = [mock_page]

        result = runner.invoke(app, ["papers", "extract", "--eid", eid])

    assert result.exit_code == 0
    assert "skipped" in result.output
    assert txt_file.read_text(encoding="utf-8") == "original content"


def test_extract_rewrite_overwrites_txt(experiments_dir, mock_no_dotenv, experiment):
    """With --rewrite, a pre-existing .txt file is overwritten."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)

    pdf_file = papers_path / "mypaper.pdf"
    pdf_file.write_bytes(_make_pdf("content"))
    txt_file = papers_path / "mypaper.txt"
    txt_file.write_text("original content", encoding="utf-8")

    with patch("llmexer.commands.papers.pypdf.PdfReader") as mock_reader:
        mock_page = Mock()
        mock_page.extract_text.return_value = "new content"
        mock_reader.return_value.pages = [mock_page]

        result = runner.invoke(app, ["papers", "extract", "--eid", eid, "--rewrite"])

    assert result.exit_code == 0
    assert "extracted:" in result.output
    assert txt_file.read_text(encoding="utf-8") == "new content"


# ---------------------------------------------------------------------------
# Docling processor tests
# ---------------------------------------------------------------------------


def _make_docling_response(md_content: str) -> Mock:
    """Return a mock requests.Response with a docling-style JSON body."""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"documents": [{"md_content": md_content}]}
    return mock_resp


def test_extract_docling_happy_path(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """With --processor docling, valid PDF produces a .md file."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))

    monkeypatch.delenv("DOCLING_URL", raising=False)
    monkeypatch.delenv("DOCLING_USER", raising=False)
    monkeypatch.delenv("DOCLING_PASSWORD", raising=False)

    mock_session = MagicMock()
    mock_session.post.return_value = _make_docling_response("# Title\n\nBody text.")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app, ["papers", "extract", "--eid", eid, "--processor", "docling"]
        )

    assert result.exit_code == 0
    assert (papers_path / "paper.md").exists()
    assert (papers_path / "paper.md").read_text(
        encoding="utf-8"
    ) == "# Title\n\nBody text."
    assert "extracted:" in result.output


def test_extract_docling_dry_run(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """With --dry-run and --processor docling, no .md file is written."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))

    monkeypatch.delenv("DOCLING_URL", raising=False)
    monkeypatch.delenv("DOCLING_USER", raising=False)
    monkeypatch.delenv("DOCLING_PASSWORD", raising=False)

    mock_session = MagicMock()
    mock_session.post.return_value = _make_docling_response("# Title")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            ["--dry-run", "papers", "extract", "--eid", eid, "--processor", "docling"],
        )

    assert result.exit_code == 0
    assert not (papers_path / "paper.md").exists()


def test_extract_docling_server_error(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """When docling server returns HTTP 500, the paper is skipped with a warning."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))

    monkeypatch.delenv("DOCLING_URL", raising=False)

    mock_resp = Mock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    mock_session = MagicMock()
    mock_session.post.return_value = mock_resp
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app, ["papers", "extract", "--eid", eid, "--processor", "docling"]
        )

    assert result.exit_code == 0
    assert "skipped" in result.output
    assert not (papers_path / "paper.md").exists()


def test_extract_docling_cli_url_override(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """--docling-url overrides the DOCLING_URL env var."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))

    monkeypatch.setenv("DOCLING_URL", "http://env-server:9999/")

    mock_session = MagicMock()
    mock_session.post.return_value = _make_docling_response("# CLI URL")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            [
                "papers",
                "extract",
                "--eid",
                eid,
                "--processor",
                "docling",
                "--docling-url",
                "http://cli-server:1234/",
            ],
        )

    assert result.exit_code == 0
    call_url = mock_session.post.call_args[0][0]
    assert "cli-server:1234" in call_url
    assert "env-server" not in call_url


def test_extract_docling_default_url(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """When no DOCLING_URL is set and no --docling-url given, uses http://localhost:5001/."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))

    monkeypatch.delenv("DOCLING_URL", raising=False)

    mock_session = MagicMock()
    mock_session.post.return_value = _make_docling_response("# Default")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app, ["papers", "extract", "--eid", eid, "--processor", "docling"]
        )

    assert result.exit_code == 0
    call_url = mock_session.post.call_args[0][0]
    assert "localhost:5001" in call_url


def test_extract_docling_skips_existing_md(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """Without --rewrite, a pre-existing .md file is skipped (docling processor)."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))
    md_file = papers_path / "paper.md"
    md_file.write_text("# Original", encoding="utf-8")

    monkeypatch.delenv("DOCLING_URL", raising=False)
    monkeypatch.delenv("DOCLING_USER", raising=False)
    monkeypatch.delenv("DOCLING_PASSWORD", raising=False)

    mock_session = MagicMock()
    mock_session.post.return_value = _make_docling_response("# New Content")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app, ["papers", "extract", "--eid", eid, "--processor", "docling"]
        )

    assert result.exit_code == 0
    assert "skipped" in result.output
    assert md_file.read_text(encoding="utf-8") == "# Original"


def test_extract_docling_rewrite_overwrites_md(
    experiments_dir, mock_no_dotenv, experiment, monkeypatch
):
    """With --rewrite, a pre-existing .md file is overwritten (docling processor)."""
    eid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(_make_pdf("content"))
    md_file = papers_path / "paper.md"
    md_file.write_text("# Original", encoding="utf-8")

    monkeypatch.delenv("DOCLING_URL", raising=False)
    monkeypatch.delenv("DOCLING_USER", raising=False)
    monkeypatch.delenv("DOCLING_PASSWORD", raising=False)

    mock_session = MagicMock()
    mock_session.post.return_value = _make_docling_response("# New Content")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            ["papers", "extract", "--eid", eid, "--processor", "docling", "--rewrite"],
        )

    assert result.exit_code == 0
    assert "extracted:" in result.output
    assert md_file.read_text(encoding="utf-8") == "# New Content"
