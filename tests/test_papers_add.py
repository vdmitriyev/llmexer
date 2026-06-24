"""Tests for the `papers add` command."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    PaperAddException,
    PaperAlreadyExistsException,
    ProjectIDRequiredException,
    ProjectNotExistsException,
    UnexpectedCLIParamsException,
)

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def experiment(projects_dir):
    """Create a test experiment directory and return its ID and path."""
    pid = "test-exp"
    exp_path = projects_dir / pid
    os.makedirs(exp_path)
    return pid, exp_path


# --- Mutual exclusion ---


def test_add_no_params_raises(projects_dir, mock_no_dotenv, experiment, monkeypatch):
    """No input params → UnexpectedCLIParamsException."""
    pid, _ = experiment
    result = runner.invoke(app, ["papers", "add", "--pid", pid])
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_add_two_params_raises(
    projects_dir, mock_no_dotenv, experiment, tmp_path, monkeypatch
):
    """Two input params → UnexpectedCLIParamsException."""
    pid, _ = experiment
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = runner.invoke(
        app,
        [
            "papers",
            "add",
            "--pid",
            pid,
            "--file",
            str(pdf),
            "--url",
            "http://example.com/x.pdf",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


# --- Experiment ID handling ---


def test_add_missing_eid_raises(projects_dir, mock_no_dotenv, monkeypatch, tmp_path):
    """No --pid and no env var → ProjectIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = runner.invoke(app, ["papers", "add", "--file", str(pdf)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_add_nonexistent_experiment_raises(
    projects_dir, mock_no_dotenv, monkeypatch, tmp_path
):
    """Non-existent experiment → ProjectNotExistsException."""
    monkeypatch.setenv("PROJECT_ID", "no-such-exp")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = runner.invoke(app, ["papers", "add", "--file", str(pdf)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


# --- --file ---


def test_add_file_happy_path(projects_dir, mock_no_dotenv, experiment, tmp_path):
    """--file copies the PDF into papers/."""
    pid, exp_path = experiment
    pdf = tmp_path / "mypaper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    result = runner.invoke(app, ["papers", "add", "--pid", pid, "--file", str(pdf)])
    assert result.exit_code == 0
    assert (exp_path / "papers" / "mypaper.pdf").exists()


def test_add_file_duplicate_skips(projects_dir, mock_no_dotenv, experiment, tmp_path):
    """--file skips with exit 0 if destination already exists."""
    pid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "mypaper.pdf").write_bytes(b"%PDF-1.4")

    pdf = tmp_path / "mypaper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    result = runner.invoke(app, ["papers", "add", "--pid", pid, "--file", str(pdf)])
    assert result.exit_code == 0


def test_add_file_not_pdf_raises(projects_dir, mock_no_dotenv, experiment, tmp_path):
    """--file raises PaperAddException if path is not a PDF."""
    pid, _ = experiment
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")

    result = runner.invoke(app, ["papers", "add", "--pid", pid, "--file", str(txt)])
    assert result.exit_code != 0
    assert isinstance(result.exception, PaperAddException)


def test_add_file_nonexistent_raises(
    projects_dir, mock_no_dotenv, experiment, tmp_path
):
    """--file raises PaperAddException if path does not exist."""
    pid, _ = experiment
    result = runner.invoke(
        app, ["papers", "add", "--pid", pid, "--file", str(tmp_path / "ghost.pdf")]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, PaperAddException)


# --- --directory ---


def test_add_directory_happy_path(projects_dir, mock_no_dotenv, experiment, tmp_path):
    """--directory copies all PDFs found recursively."""
    pid, exp_path = experiment
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (subdir / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "readme.txt").write_text("ignore me")

    result = runner.invoke(
        app, ["papers", "add", "--pid", pid, "--directory", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert (exp_path / "papers" / "a.pdf").exists()
    assert (exp_path / "papers" / "b.pdf").exists()
    assert not (exp_path / "papers" / "readme.txt").exists()


def test_add_directory_duplicate_skips(
    projects_dir, mock_no_dotenv, experiment, tmp_path
):
    """--directory skips duplicates and exits 0 on collision."""
    pid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "a.pdf").write_bytes(b"%PDF-1.4")

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")

    result = runner.invoke(
        app, ["papers", "add", "--pid", pid, "--directory", str(tmp_path)]
    )
    assert result.exit_code == 0


def test_add_directory_invalid_raises(
    projects_dir, mock_no_dotenv, experiment, tmp_path
):
    """--directory raises PaperAddException if path is not a directory."""
    pid, _ = experiment
    result = runner.invoke(
        app,
        ["papers", "add", "--pid", pid, "--directory", str(tmp_path / "nonexistent")],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, PaperAddException)


# --- --url helpers ---


def _make_mock_response(url="http://example.com/paper.pdf", content_disposition=None):
    """Build a mock requests.Response for URL tests."""
    mock_response = MagicMock()
    mock_response.raise_for_status = Mock()
    mock_response.url = url
    mock_response.iter_content = Mock(return_value=[b"%PDF-1.4", b" content"])
    headers = {}
    if content_disposition:
        headers["Content-Disposition"] = content_disposition
    mock_response.headers = headers
    return mock_response


# --- --url ---


def test_add_url_happy_path(projects_dir, mock_no_dotenv, experiment):
    """--url with explicit .pdf in URL downloads the PDF into papers/."""
    pid, exp_path = experiment
    mock_response = _make_mock_response()

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            ["papers", "add", "--pid", pid, "--url", "http://example.com/paper.pdf"],
        )

    assert result.exit_code == 0
    assert (exp_path / "papers" / "paper.pdf").exists()


def test_add_url_resolved_via_final_url(projects_dir, mock_no_dotenv, experiment):
    """--url without .pdf in original URL resolves filename from the final redirected URL."""
    pid, exp_path = experiment
    mock_response = _make_mock_response(url="http://example.com/redirected/mypaper.pdf")

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            [
                "papers",
                "add",
                "--pid",
                pid,
                "--url",
                "http://example.com/download/12345",
            ],
        )

    assert result.exit_code == 0
    assert (exp_path / "papers" / "mypaper.pdf").exists()


def test_add_url_resolved_via_content_disposition(
    projects_dir, mock_no_dotenv, experiment
):
    """--url resolves filename from Content-Disposition header when URL path has no .pdf."""
    pid, exp_path = experiment
    mock_response = _make_mock_response(
        url="http://example.com/download/12345",
        content_disposition='attachment; filename="article.pdf"',
    )

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            [
                "papers",
                "add",
                "--pid",
                pid,
                "--url",
                "http://example.com/download/12345",
            ],
        )

    assert result.exit_code == 0
    assert (exp_path / "papers" / "article.pdf").exists()


def test_add_url_duplicate_raises(projects_dir, mock_no_dotenv, experiment):
    """--url raises PaperAlreadyExistsException if destination already exists."""
    pid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(b"%PDF-1.4")

    mock_response = _make_mock_response()
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            ["papers", "add", "--pid", pid, "--url", "http://example.com/paper.pdf"],
        )
    assert result.exit_code != 0
    assert isinstance(result.exception, PaperAlreadyExistsException)


def test_add_url_not_pdf_raises(projects_dir, mock_no_dotenv, experiment):
    """--url raises PaperAddException if no PDF filename can be resolved from URL or headers."""
    pid, _ = experiment
    mock_response = _make_mock_response(url="http://example.com/file.zip")

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app, ["papers", "add", "--pid", pid, "--url", "http://example.com/file.zip"]
        )
    assert result.exit_code != 0
    assert isinstance(result.exception, PaperAddException)


def test_add_url_request_failure_raises(projects_dir, mock_no_dotenv, experiment):
    """--url raises PaperAddException on network error."""
    import requests as req

    pid, _ = experiment
    mock_session = MagicMock()
    mock_session.get.side_effect = req.RequestException("timeout")
    with patch("llmexer.base.papers.make_http_session", return_value=mock_session):
        result = runner.invoke(
            app,
            ["papers", "add", "--pid", pid, "--url", "http://example.com/paper.pdf"],
        )
    assert result.exit_code != 0
    assert isinstance(result.exception, PaperAddException)
