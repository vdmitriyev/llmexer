"""Tests for the `experiment compact` command."""

import os
from unittest.mock import Mock

import py7zr
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException
from tests.db_helpers import LITELLM_ROW, OLLAMA_ROW, seed_db

runner = CliRunner()

PID = "compact-test-exp"
_DB_NAME = "experiment_20240101_01.db"
_ARCHIVE_NAME = "experiment_20240101_01.7z"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def experiment_with_results(projects_dir):
    """A database with one finished ollama row and one unrun litellm row."""
    exp_subdir = projects_dir / PID / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {
            "ollama": [
                dict(
                    OLLAMA_ROW,
                    ID=1,
                    response_text='{"relevant": true}',
                    status="success",
                    total_tokens=142,
                )
            ],
            "litellm": [dict(LITELLM_ROW, ID=2)],
        },
    )
    return exp_subdir


def _compact(pid=PID, *options):
    """Invoke `experiment compact` on the seeded database."""
    return runner.invoke(app, ["experiment", "compact", "--pid", pid, "--file", _DB_NAME, *options])


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------


def test_compact_writes_the_archive_next_to_the_database(experiment_with_results):
    """The archive takes the database name with a `.7z` extension."""
    result = _compact()

    assert result.exit_code == 0, result.output
    assert (experiment_with_results / _ARCHIVE_NAME).exists()


def test_compact_keeps_the_database(experiment_with_results):
    """Compacting only adds a file: the database stays where it was."""
    db_path = experiment_with_results / _DB_NAME
    before = db_path.read_bytes()

    result = _compact()

    assert result.exit_code == 0, result.output
    assert db_path.read_bytes() == before


def test_compact_stores_the_database_under_its_own_name(experiment_with_results):
    """The archive holds the `.db` alone, at its bare file name."""
    _compact()

    with py7zr.SevenZipFile(experiment_with_results / _ARCHIVE_NAME) as archive:
        assert archive.getnames() == [_DB_NAME]


def test_compact_roundtrips_the_database_byte_for_byte(experiment_with_results, tmp_path):
    """Extracting the archive gives back exactly the database that went in."""
    _compact()

    target = tmp_path / "extracted"
    with py7zr.SevenZipFile(experiment_with_results / _ARCHIVE_NAME) as archive:
        archive.extractall(path=target)

    assert (target / _DB_NAME).read_bytes() == (experiment_with_results / _DB_NAME).read_bytes()


# ---------------------------------------------------------------------------
# CLI behaviour: dry run, rewrite, errors
# ---------------------------------------------------------------------------


def test_compact_dry_run_writes_nothing(experiment_with_results):
    """`--dry-run` announces the target file but writes nothing."""
    result = runner.invoke(
        app,
        ["--dry-run", "experiment", "compact", "--pid", PID, "--file", _DB_NAME],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not (experiment_with_results / _ARCHIVE_NAME).exists()


def test_compact_does_not_overwrite_without_rewrite(experiment_with_results):
    """An existing archive is kept unless `--rewrite` is passed."""
    archive_path = experiment_with_results / _ARCHIVE_NAME
    archive_path.write_bytes(b"existing")

    result = _compact()

    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert archive_path.read_bytes() == b"existing"

    result = _compact(PID, "--rewrite")

    assert result.exit_code == 0, result.output
    assert archive_path.read_bytes() != b"existing"


def test_compact_accepts_an_absolute_path(experiment_with_results):
    """`--file` may be a full path to the database instead of its name."""
    db_path = experiment_with_results / _DB_NAME

    result = runner.invoke(app, ["experiment", "compact", "--pid", PID, "--file", str(db_path)])

    assert result.exit_code == 0, result.output
    assert (experiment_with_results / _ARCHIVE_NAME).exists()


def test_compact_without_database_raises(projects_dir, mock_no_dotenv):
    """A project with no generated database points the user at `generate`."""
    os.makedirs(projects_dir / "empty-exp" / "experiment")

    result = runner.invoke(app, ["experiment", "compact", "--pid", "empty-exp"])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "generate" in str(result.exception).lower()


def test_compact_missing_file_raises(experiment_with_results):
    """An explicit --file that does not exist is an error, not an empty archive."""
    result = _compact(PID, "--file", "nope.db")

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_compact_defaults_to_the_newest_database(experiment_with_results):
    """With no --file the newest experiment_*.db is compacted."""
    seed_db(experiment_with_results / "experiment_20240101_02.db", {"ollama": [dict(OLLAMA_ROW)]})

    result = runner.invoke(app, ["experiment", "compact", "--pid", PID])

    assert result.exit_code == 0, result.output
    assert (experiment_with_results / "experiment_20240101_02.7z").exists()
    assert not (experiment_with_results / _ARCHIVE_NAME).exists()
