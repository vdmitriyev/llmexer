"""Tests for the `papers download` command."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    PaperDownloadException,
    ProjectIDRequiredException,
    ProjectNotExistsException,
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
    """Create a test experiment directory and return (pid, exp_path)."""
    pid = "test-exp"
    exp_path = projects_dir / pid
    os.makedirs(exp_path)
    return pid, exp_path


def _make_unpaywall_response(
    pdf_url="http://example.com/paper.pdf", has_oa=True, pdf_url_null=False
):
    mock = MagicMock()
    mock.raise_for_status = Mock()
    if has_oa and not pdf_url_null:
        mock.json.return_value = {"best_oa_location": {"url_for_pdf": pdf_url}}
    elif has_oa and pdf_url_null:
        mock.json.return_value = {"best_oa_location": {"url_for_pdf": None}}
    else:
        mock.json.return_value = {"best_oa_location": None}
    return mock


def _make_pdf_response(url="http://example.com/paper.pdf", filename=None):
    mock = MagicMock()
    mock.raise_for_status = Mock()
    mock.url = url
    if filename:
        mock.headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    else:
        mock.headers = {}
    mock.iter_content = Mock(return_value=[b"%PDF-1.4 fake content"])
    return mock


def _mock_session(*get_side_effects):
    """Return a mock session whose .get() returns the given side effects in order."""
    mock_session = MagicMock()
    if len(get_side_effects) == 1 and not isinstance(
        get_side_effects[0], (list, Exception)
    ):
        mock_session.get.return_value = get_side_effects[0]
    else:
        mock_session.get.side_effect = list(get_side_effects)
    return mock_session


# --- EID and experiment validation ---


def test_download_missing_eid_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """When --pid is not provided and PROJECT_ID is not set, raises ProjectIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["papers", "download", "--doi", "10.1000/xyz"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_download_nonexistent_experiment_raises(projects_dir, mock_no_dotenv):
    """When experiment does not exist, raises ProjectNotExistsException."""
    result = runner.invoke(
        app,
        [
            "papers",
            "download",
            "--pid",
            "nonexistent-exp",
            "--doi",
            "10.1000/xyz",
            "--email",
            "test@example.com",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


# --- Email validation ---


def test_download_email_from_env(projects_dir, mock_no_dotenv, experiment, monkeypatch):
    """When UNPAYWALL_EMAIL env var is set and --email is omitted, it is used automatically."""
    pid, exp_path = experiment
    monkeypatch.setenv("UNPAYWALL_EMAIL", "env@example.com")

    unpaywall_mock = _make_unpaywall_response("http://example.com/paper.pdf")
    pdf_mock = _make_pdf_response("http://example.com/paper.pdf")

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[_mock_session(unpaywall_mock), _mock_session(pdf_mock)],
    ):
        result = runner.invoke(
            app, ["papers", "download", "--pid", pid, "--doi", "10.1000/xyz"]
        )

    assert result.exit_code == 0
    assert (exp_path / "papers" / "paper.pdf").exists()


# --- Happy path ---


def test_download_happy_path_single_doi(projects_dir, mock_no_dotenv, experiment):
    """Single DOI resolves and downloads a PDF into the papers directory."""
    pid, exp_path = experiment

    unpaywall_mock = _make_unpaywall_response("http://example.com/paper.pdf")
    pdf_mock = _make_pdf_response("http://example.com/paper.pdf")

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[_mock_session(unpaywall_mock), _mock_session(pdf_mock)],
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert (exp_path / "papers" / "paper.pdf").exists()
    assert "downloaded" in result.output


def test_download_happy_path_multiple_dois(projects_dir, mock_no_dotenv, experiment):
    """Multiple --doi flags each resolve and download to the papers directory."""
    pid, exp_path = experiment

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[
            _mock_session(_make_unpaywall_response("http://example.com/paper1.pdf")),
            _mock_session(_make_pdf_response("http://example.com/paper1.pdf")),
            _mock_session(_make_unpaywall_response("http://example.com/paper2.pdf")),
            _mock_session(_make_pdf_response("http://example.com/paper2.pdf")),
        ],
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/abc",
                "--doi",
                "10.1000/def",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert (exp_path / "papers" / "paper1.pdf").exists()
    assert (exp_path / "papers" / "paper2.pdf").exists()


# --- Skip / failure scenarios ---


def test_download_no_oa_location_skips(projects_dir, mock_no_dotenv, experiment):
    """When Unpaywall returns no OA location, the DOI is skipped with a warning."""
    pid, exp_path = experiment

    unpaywall_mock = _make_unpaywall_response(has_oa=False)

    with patch(
        "llmexer.base.papers.make_http_session",
        return_value=_mock_session(unpaywall_mock),
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert "skipped" in result.output
    papers_path = exp_path / "papers"
    assert not any(papers_path.iterdir()) if papers_path.exists() else True


def test_download_no_pdf_url_skips(projects_dir, mock_no_dotenv, experiment):
    """When Unpaywall OA location exists but url_for_pdf is null, the DOI is skipped."""
    pid, _ = experiment

    unpaywall_mock = _make_unpaywall_response(has_oa=True, pdf_url_null=True)

    with patch(
        "llmexer.base.papers.make_http_session",
        return_value=_mock_session(unpaywall_mock),
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert "skipped" in result.output


def test_download_unpaywall_request_failure_skips(
    projects_dir, mock_no_dotenv, experiment
):
    """When the Unpaywall API call raises a network error, the DOI is skipped."""
    import requests as req

    pid, _ = experiment

    mock_session = MagicMock()
    mock_session.get.side_effect = req.RequestException("timeout")

    with patch(
        "llmexer.base.papers.make_http_session",
        return_value=mock_session,
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert "skipped" in result.output


def test_download_pdf_request_failure_skips(projects_dir, mock_no_dotenv, experiment):
    """When the PDF download raises a network error, the DOI is marked as failed."""
    import requests as req

    pid, _ = experiment

    unpaywall_mock = _make_unpaywall_response("http://example.com/paper.pdf")
    pdf_error = req.RequestException("connection refused")

    pdf_session = MagicMock()
    pdf_session.get.side_effect = pdf_error

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[_mock_session(unpaywall_mock), pdf_session],
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert "failed" in result.output


def test_download_duplicate_skips(projects_dir, mock_no_dotenv, experiment):
    """When a PDF with the same filename already exists, the DOI is skipped."""
    pid, exp_path = experiment
    papers_path = exp_path / "papers"
    os.makedirs(papers_path)
    (papers_path / "paper.pdf").write_bytes(b"existing")

    unpaywall_mock = _make_unpaywall_response("http://example.com/paper.pdf")
    pdf_mock = _make_pdf_response("http://example.com/paper.pdf")

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[_mock_session(unpaywall_mock), _mock_session(pdf_mock)],
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert "skipped" in result.output
    # Original file should be unchanged
    assert (papers_path / "paper.pdf").read_bytes() == b"existing"


# --- Dry run ---


def test_download_dry_run(projects_dir, mock_no_dotenv, experiment):
    """With --dry-run, no PDF file is written."""
    pid, exp_path = experiment

    unpaywall_mock = _make_unpaywall_response("http://example.com/paper.pdf")
    pdf_mock = _make_pdf_response("http://example.com/paper.pdf")

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[_mock_session(unpaywall_mock), _mock_session(pdf_mock)],
    ):
        result = runner.invoke(
            app,
            [
                "--dry-run",
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert not (exp_path / "papers" / "paper.pdf").exists()


# --- Fallback filename ---


def test_download_fallback_filename_from_doi(projects_dir, mock_no_dotenv, experiment):
    """When the PDF URL has no .pdf extension and no Content-Disposition, the DOI is used as filename."""
    pid, exp_path = experiment

    # URL with no .pdf extension
    unpaywall_mock = _make_unpaywall_response("http://example.com/view?id=123")
    pdf_mock = _make_pdf_response("http://example.com/view?id=123")

    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[_mock_session(unpaywall_mock), _mock_session(pdf_mock)],
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/xyz123",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    papers_path = exp_path / "papers"
    pdf_files = list(papers_path.glob("*.pdf"))
    assert len(pdf_files) == 1
    assert pdf_files[0].name == "10.1000_xyz123.pdf"


# --- Summary output ---


def test_download_summary_output(projects_dir, mock_no_dotenv, experiment):
    """Output contains a summary line showing succeeded/failed counts."""
    pid, _ = experiment

    # First DOI succeeds, second has no OA
    with patch(
        "llmexer.base.papers.make_http_session",
        side_effect=[
            _mock_session(_make_unpaywall_response("http://example.com/paper.pdf")),
            _mock_session(_make_pdf_response("http://example.com/paper.pdf")),
            _mock_session(_make_unpaywall_response(has_oa=False)),
        ],
    ):
        result = runner.invoke(
            app,
            [
                "papers",
                "download",
                "--pid",
                pid,
                "--doi",
                "10.1000/good",
                "--doi",
                "10.1000/bad",
                "--email",
                "test@example.com",
            ],
        )

    assert result.exit_code == 0
    assert "Downloaded:" in result.output
    assert "Skipped/Failed:" in result.output
