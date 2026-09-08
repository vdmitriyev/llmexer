"""Tests for the `analysis init` command."""

import os
from unittest.mock import Mock

import nbformat
import pytest
from typer.testing import CliRunner

from llmexer.base.analysis import COPIED_MODULES
from llmexer.base.analysis.notebook import NOTEBOOKS, module_source_path
from llmexer.cli import app
from llmexer.constants import ANALYSIS_BACKUP_DIR
from llmexer.exceptions import ProjectIDRequiredException, ProjectNotExistsException
from tests.db_helpers import OLLAMA_ROW, seed_db

runner = CliRunner()

PID = "analysis-test-exp"

# `analysis` plus the two aliases registered in cli.py; all three are the same app.
GROUP_NAMES = ("analysis", "analyse", "analyze")


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
def full_project(project):
    """A project with a generated database, a search CSV and a paper."""

    exp_subdir = project / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / "experiment_20240101_01.db", {"ollama": [dict(OLLAMA_ROW)]})

    searches = project / "searches"
    os.makedirs(searches)
    (searches / "s1.yaml").write_text("query: a\n", encoding="utf-8")
    (searches / "s1__results.csv").write_text("entry_source;title\nOpenAlex;A\n", encoding="utf-8")

    papers = project / "papers"
    os.makedirs(papers)
    (papers / "a.pdf").write_text("x", encoding="utf-8")

    return project


def _init(*options, pid=PID, group="analysis"):
    return runner.invoke(app, [group, "init", "--pid", pid, *options])


def _analysis_dir(project):
    return project / "analysis"


def _notebooks(project):
    return [_analysis_dir(project) / name for name in NOTEBOOKS.values()]


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def test_init_creates_the_analysis_folder(project):
    result = _init()

    assert result.exit_code == 0, result.exception
    assert _analysis_dir(project).is_dir()


def test_init_writes_both_notebooks(project):
    _init()

    for path in _notebooks(project):
        assert path.is_file()


def test_init_copies_the_analysis_modules(project):
    _init()

    for filename in COPIED_MODULES:
        assert (_analysis_dir(project) / filename).is_file()


def test_copied_modules_are_byte_identical_to_the_shipped_ones(project):
    _init()

    for filename in COPIED_MODULES:
        copied = (_analysis_dir(project) / filename).read_bytes()
        shipped = open(module_source_path(filename), "rb").read()
        assert copied == shipped


def test_init_does_not_copy_the_package_only_modules(project):
    """`notebook.py` imports llmexer, so it must stay in the package."""

    _init()
    copied = {path.name for path in _analysis_dir(project).iterdir()}

    assert "notebook.py" not in copied
    assert "__init__.py" not in copied


def test_init_reports_the_created_files(project):
    result = _init()

    assert "analysis" in result.output
    for filename in NOTEBOOKS.values():
        assert filename in result.output


# ---------------------------------------------------------------------------
# Notebook validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_written_notebook_is_valid(project, name):
    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)

    nbformat.validate(notebook)


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_notebook_declares_a_kernel(project, name):
    """Without a kernelspec JupyterLab prompts for a kernel on every open."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)

    assert notebook.metadata["kernelspec"]["name"] == "python3"


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_notebook_holds_no_business_logic(project, name):
    """Cells configure paths, import, call and display - nothing more."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert "def " not in cell.source
            assert "class " not in cell.source


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_every_code_cell_compiles(project, name):
    """Catches a syntax error in a template without starting a kernel."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        body = "\n".join(line for line in cell.source.splitlines() if not line.strip().startswith(("%", "!")))
        compile(body, f"<{name}>", "exec")


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_notebook_carries_no_stored_output(project, name):
    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == []


def test_notebook_injects_the_project_and_paths(project):
    _init()
    source = (_analysis_dir(project) / NOTEBOOKS["experiment"]).read_text(encoding="utf-8")

    assert PID in source
    assert "DIR_EXPERIMENT" in source


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_notebook_code_never_imports_llmexer(project, name):
    """The analysis folder is self-contained: cells import only their siblings."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "llmexer" not in code


def _exec_folders_cell(project, name):
    """Run the part of the setup cell that defines the injected literals."""

    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)
    body = notebook.cells[2].source.split("# --- imports")[0]
    namespace = {}
    exec(compile(body, "<setup>", "exec"), namespace)  # nosec B102 - our own generated cell

    return namespace


def test_experiment_notebook_names_the_newest_database(full_project):
    """`analysis init` writes the database filename in as an editable literal."""

    _init()

    assert _exec_folders_cell(full_project, NOTEBOOKS["experiment"])["DB_FILE"] == "experiment_20240101_01.db"


def test_experiment_notebook_handles_a_project_with_no_database(project):
    """`None`, not JSON `null` - the cell has to stay valid Python."""

    _init()

    assert _exec_folders_cell(project, NOTEBOOKS["experiment"])["DB_FILE"] is None


def test_searches_notebook_lists_one_entry_per_search(full_project):
    """Each entry names the search YAML and the CSVs written from it."""

    _init()
    searches = _exec_folders_cell(full_project, NOTEBOOKS["searches"])["SEARCHES"]

    assert searches == [{"filter": None, "results": "s1__results.csv", "search": "s1.yaml"}]


def test_searches_notebook_handles_a_project_with_no_searches(project):
    _init()

    assert _exec_folders_cell(project, NOTEBOOKS["searches"])["SEARCHES"] == []


def test_notebook_derives_the_data_folders_from_its_own_parent(project):
    """No absolute paths are baked in, so the project stays relocatable."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOKS["experiment"]), as_version=4)
    setup = notebook.cells[2].source

    assert "ANALYSIS_DIR = Path.cwd()" in setup
    assert "PROJECT_DIR = ANALYSIS_DIR.parent" in setup
    assert str(_analysis_dir(project)) not in setup


@pytest.mark.parametrize("name", list(NOTEBOOKS.values()))
def test_notebook_documents_how_to_change_the_folders(project, name):
    """A reader with a different layout has to be told they can edit the paths."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / name), as_version=4)
    folders = notebook.cells[1].source

    assert folders.startswith("## Folders")
    assert "Change them by hand" in folders
    # The folder tree was dropped: the table below already names every path.
    assert "<- this notebook" not in folders


def test_searches_notebook_analyses_one_search_at_a_time(project):
    """Rows from two different queries are two populations, never one."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOKS["searches"]), as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "SEARCH_INDEX" in code
    assert "load_search_frame(" in code


def test_searches_notebook_exports_no_csv(project):
    """Flattening is for experiment answers; a search CSV has nothing to flatten."""

    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOKS["searches"]), as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "export_as_csv" not in code
    assert "searches_flattened" not in code


def test_searches_notebook_declares_the_search_and_paper_folders(project):
    _init()
    notebook = nbformat.read(str(_analysis_dir(project) / NOTEBOOKS["searches"]), as_version=4)
    setup = notebook.cells[2].source

    assert "DIR_SEARCHES" in setup
    assert "DIR_PAPERS" in setup


def test_a_project_id_with_quotes_still_renders_valid_json(projects_dir):
    """Every injected value goes through `tojson`, so odd names cannot break it."""

    pid = "odd 'quoted' name"
    os.makedirs(projects_dir / pid)
    result = _init(pid=pid)

    assert result.exit_code == 0, result.exception
    nbformat.read(str(projects_dir / pid / "analysis" / NOTEBOOKS["experiment"]), as_version=4)


# ---------------------------------------------------------------------------
# Overwrite guard and backups
# ---------------------------------------------------------------------------


def test_rerun_without_rewrite_keeps_existing_files(project):
    _init()
    edited = _analysis_dir(project) / "stats.py"
    edited.write_text("# researcher's own version\n", encoding="utf-8")

    result = _init()

    assert result.exit_code == 0, result.exception
    assert edited.read_text(encoding="utf-8") == "# researcher's own version\n"
    assert "already exists" in result.output


def test_rewrite_replaces_the_files(project):
    _init()
    edited = _analysis_dir(project) / "stats.py"
    edited.write_text("# researcher's own version\n", encoding="utf-8")

    result = _init("--rewrite")

    assert result.exit_code == 0, result.exception
    assert edited.read_text(encoding="utf-8") != "# researcher's own version\n"


def test_rewrite_backs_up_the_previous_files_into_the_backup_folder(project):
    """Backups live in `.backup/`, so the analysis folder shows only live files."""

    _init()
    (_analysis_dir(project) / "stats.py").write_text("# researcher's own version\n", encoding="utf-8")

    _init("--rewrite")
    backups = sorted((_analysis_dir(project) / ANALYSIS_BACKUP_DIR).glob("stats_backup_*.py"))

    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "# researcher's own version\n"


def test_rewrite_backs_up_the_notebooks_too(project):
    _init()
    _init("--rewrite")

    assert list((_analysis_dir(project) / ANALYSIS_BACKUP_DIR).glob("analyse_experiment_backup_*.ipynb"))


def test_backups_never_sit_next_to_the_notebooks(project):
    _init()
    _init("--rewrite")
    beside = [path.name for path in _analysis_dir(project).iterdir() if "_backup_" in path.name]

    assert beside == []


def test_no_backup_folder_is_created_without_a_rewrite(project):
    _init()

    assert not (_analysis_dir(project) / ANALYSIS_BACKUP_DIR).exists()


def test_a_missing_file_is_restored_without_rewrite(project):
    """The guard is per file, so a deleted module comes back on its own."""

    _init()
    (_analysis_dir(project) / "plots.py").unlink()

    result = _init()

    assert result.exit_code == 0, result.exception
    assert (_analysis_dir(project) / "plots.py").is_file()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(project):
    result = runner.invoke(app, ["--dry-run", "analysis", "init", "--pid", PID])

    assert result.exit_code == 0, result.exception
    assert not _analysis_dir(project).exists()
    assert "dry run" in result.output.lower()


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", GROUP_NAMES)
def test_every_alias_scaffolds_the_same_workspace(projects_dir, group):
    pid = f"{PID}-{group}"
    os.makedirs(projects_dir / pid)

    result = _init(pid=pid, group=group)

    assert result.exit_code == 0, result.exception
    written = sorted(path.name for path in (projects_dir / pid / "analysis").iterdir())
    assert written == sorted(list(NOTEBOOKS.values()) + list(COPIED_MODULES))


# ---------------------------------------------------------------------------
# Readiness reporting
# ---------------------------------------------------------------------------


def test_init_succeeds_on_a_project_with_no_experiment(project):
    """Scaffolding is free; requiring `experiment generate` first would be a
    made-up ordering constraint."""

    result = _init()

    assert result.exit_code == 0, result.exception
    for filename in NOTEBOOKS.values():
        assert (_analysis_dir(project) / filename).is_file()


def test_init_on_a_populated_project_scaffolds_the_same_files(full_project):
    """The command reports what it wrote, not what the notebooks will find."""

    result = _init()

    assert result.exit_code == 0, result.exception
    written = sorted(path.name for path in _analysis_dir(full_project).iterdir())
    assert written == sorted(list(NOTEBOOKS.values()) + list(COPIED_MODULES))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unknown_project_raises(projects_dir, mock_no_dotenv):
    result = _init(pid="does-not-exist")

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


def test_no_project_id_raises(projects_dir, mock_no_dotenv, monkeypatch):
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.setattr("llmexer.configs.settings.project_id", None)

    result = runner.invoke(app, ["analysis", "init"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)
