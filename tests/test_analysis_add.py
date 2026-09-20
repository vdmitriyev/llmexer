"""Tests for the `analysis add-agreement` command."""

import os
from unittest.mock import Mock

import nbformat
import pytest
from typer.testing import CliRunner

from llmexer.base.analysis import COPIED_MODULES
from llmexer.base.analysis.notebook import ADDON_NOTEBOOKS
from llmexer.cli import app
from llmexer.constants import ANALYSIS_BACKUP_DIR
from llmexer.exceptions import (
    ProjectIDRequiredException,
    ProjectNotExistsException,
    UnexpectedCLIParamsException,
)
from tests.db_helpers import OLLAMA_ROW, seed_db

runner = CliRunner()

PID = "analysis-add-test-exp"
NOTEBOOK = ADDON_NOTEBOOKS["agreements"]


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
def project(projects_dir):
    """A bare project folder - nothing initialised, nothing generated."""

    path = projects_dir / PID
    os.makedirs(path)
    return path


@pytest.fixture()
def project_with_db(project):
    """A project carrying one generated database, so a CSV name can be derived."""

    exp_subdir = project / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / "experiment_20240101_01.db", {"ollama": [dict(OLLAMA_ROW)]})
    return project


def _add(*options, pid=PID, group="analysis"):
    return runner.invoke(app, [group, "add-agreement", "--pid", pid, *options])


def _analysis_dir(project):
    return project / "analysis"


def _literals(project):
    """The editable literal cell of the written notebook, as source lines."""

    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOK), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and "CSV_FILE" in cell.source:
            return cell.source
    raise AssertionError("no literal cell found")


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def test_add_writes_the_notebook_and_the_modules(project):
    result = _add()

    assert result.exit_code == 0, result.exception
    assert (_analysis_dir(project) / NOTEBOOK).is_file()
    for module in COPIED_MODULES:
        assert (_analysis_dir(project) / module).is_file()


def test_add_does_not_write_the_init_notebooks(project):
    """`add-agreement` adds one notebook; it is not a second `init`."""

    _add()

    written = {path.name for path in _analysis_dir(project).glob("*.ipynb")}
    assert written == {NOTEBOOK}


def test_the_notebook_is_valid_and_every_code_cell_compiles(project):
    _add()

    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOK), as_version=4)
    nbformat.validate(notebook)

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        body = "\n".join(line for line in cell.source.splitlines() if not line.strip().startswith(("%", "!")))
        compile(body, f"cell-{index}", "exec")


def test_the_notebook_carries_no_stored_outputs(project):
    _add()

    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOK), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []


# ---------------------------------------------------------------------------
# Reported and injected values
# ---------------------------------------------------------------------------


def test_the_resolved_values_are_reported(project_with_db):
    result = _add()

    assert "fields:" in result.output
    assert "detected in the notebook" in result.output
    assert "autodetect-fields:" in result.output
    assert "experiment_20240101_01_flattened.csv" in result.output
    assert NOTEBOOK in result.output


def test_explicit_fields_are_reported_and_injected(project_with_db):
    result = _add("--fields", "isTitleProvided,isAbstractProvided")

    assert "isTitleProvided, isAbstractProvided" in result.output
    assert "explicit --fields given" in result.output
    assert "FIELDS = ['isTitleProvided', 'isAbstractProvided']" in _literals(project_with_db)


def test_fields_is_repeatable_and_deduplicated(project_with_db):
    _add("--fields", "a,b", "--fields", "b", "--fields", " c ")

    assert "FIELDS = ['a', 'b', 'c']" in _literals(project_with_db)


def test_no_fields_leaves_the_list_empty_and_autodetect_on(project_with_db):
    _add()

    source = _literals(project_with_db)
    assert "FIELDS = []" in source
    assert "AUTODETECT_FIELDS = True" in source


def test_autodetect_can_be_switched_off_when_fields_are_given(project_with_db):
    _add("--fields", "a", "--no-autodetect-fields")

    assert "AUTODETECT_FIELDS = False" in _literals(project_with_db)


def test_the_database_and_csv_names_follow_the_newest_database(project_with_db):
    _add()

    source = _literals(project_with_db)
    assert "DB_FILE = 'experiment_20240101_01.db'" in source
    assert "CSV_FILE = 'experiment_20240101_01_flattened.csv'" in source


def test_without_a_database_the_csv_name_is_none(project):
    result = _add()

    assert "no experiment database yet" in result.output
    source = _literals(project)
    assert "DB_FILE = None" in source
    assert "CSV_FILE = None" in source


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_no_fields_and_no_autodetect_is_rejected(project):
    result = _add("--no-autodetect-fields")

    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert not _analysis_dir(project).exists()


def test_an_existing_notebook_is_kept_without_rewrite(project):
    _add()
    path = _analysis_dir(project) / NOTEBOOK
    path.write_text("edited by hand", encoding="utf-8")

    result = _add()

    assert result.exit_code == 0
    assert path.read_text(encoding="utf-8") == "edited by hand"
    assert "--rewrite" in result.output


def test_rewrite_backs_the_notebook_up_first(project):
    _add()
    path = _analysis_dir(project) / NOTEBOOK
    path.write_text("edited by hand", encoding="utf-8")

    result = _add("--rewrite")

    assert result.exit_code == 0
    backups = list((_analysis_dir(project) / ANALYSIS_BACKUP_DIR).glob("*.ipynb"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "edited by hand"
    assert path.read_text(encoding="utf-8") != "edited by hand"


def test_no_backup_folder_without_a_rewrite(project):
    _add()

    assert not (_analysis_dir(project) / ANALYSIS_BACKUP_DIR).exists()


def test_dry_run_writes_nothing(project):
    result = runner.invoke(app, ["--dry-run", "analysis", "add-agreement", "--pid", PID])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert not _analysis_dir(project).exists()


def test_unknown_project_is_rejected(projects_dir, mock_no_dotenv):
    result = _add(pid="nope")

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


def test_no_project_id_is_rejected(projects_dir, mock_no_dotenv, monkeypatch):
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["analysis", "add-agreement"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


@pytest.mark.parametrize("group", ("analysis", "analyse", "analyze"))
def test_every_alias_adds_the_same_notebook(project, group):
    result = _add(group=group)

    assert result.exit_code == 0, result.exception
    assert (_analysis_dir(project) / NOTEBOOK).is_file()
