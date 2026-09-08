"""Tests for the package-only `notebook` builder, plus a no-kernel smoke run."""

import builtins
import os

import matplotlib

matplotlib.use("Agg")  # no display; must precede any pyplot import

import nbformat  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from llmexer.base.analysis.notebook import (  # noqa: E402
    NOTEBOOKS,
    TEMPLATES,
    render_notebook,
    validate_notebook,
    write_notebook,
)
from llmexer.cli import app  # noqa: E402
from llmexer.exceptions import LLMExerException  # noqa: E402
from tests.db_helpers import OLLAMA_ROW, seed_db  # noqa: E402

runner = CliRunner()

PID = "notebook-test-exp"

CONTEXT = {
    "project_id": PID,
    "package_version": "0.4.0",
    "generated_at": "2026-09-08 12:00:00 UTC",
    "db_file": "experiment_20240101_01.db",
    "searches": [{"search": "s1.yaml", "results": "s1__results.csv", "filter": None}],
}


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Rendering and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(TEMPLATES))
def test_render_produces_a_valid_notebook(kind):
    notebook = validate_notebook(render_notebook(kind, CONTEXT), kind)

    assert notebook.cells
    assert notebook.nbformat == 4


@pytest.mark.parametrize("kind", list(TEMPLATES))
def test_render_injects_the_context(kind):
    source = render_notebook(kind, CONTEXT)

    assert PID in source
    assert "0.4.0" in source


@pytest.mark.parametrize(
    "value",
    ['a "quoted" project', r"C:\windows\path", "line\nbreak", "emoji 🎯", "back\\slash"],
)
def test_awkward_values_still_render_valid_json(value):
    """Every injected value goes through `tojson`, so nothing can break the JSON."""

    notebook = validate_notebook(render_notebook("experiment", {**CONTEXT, "project_id": value}), "experiment")

    assert notebook.cells


def test_render_rejects_an_unknown_kind():
    with pytest.raises(LLMExerException):
        render_notebook("nope", CONTEXT)


def test_validate_rejects_malformed_json():
    with pytest.raises(LLMExerException) as excinfo:
        validate_notebook("{not json", "experiment")

    assert "experiment" in str(excinfo.value)


def test_validate_rejects_a_json_document_that_is_not_a_notebook():
    with pytest.raises(LLMExerException):
        validate_notebook('{"cells": "not a list"}', "experiment")


def test_write_notebook_writes_and_returns_the_path(tmp_path):
    path = tmp_path / "out.ipynb"
    returned = write_notebook("experiment", CONTEXT, str(path))

    assert returned == str(path)
    nbformat.validate(nbformat.read(str(path), as_version=4))


def test_write_notebook_leaves_nothing_behind_when_rendering_fails(tmp_path):
    """Validation happens before the file is opened, so a bad template writes nothing."""

    path = tmp_path / "out.ipynb"
    with pytest.raises(LLMExerException):
        write_notebook("nope", CONTEXT, str(path))

    assert not path.exists()


# ---------------------------------------------------------------------------
# Smoke run: execute the notebook's code without starting a Jupyter kernel
# ---------------------------------------------------------------------------


def _project_with_data(projects_dir):
    project = projects_dir / PID
    exp_subdir = project / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / "experiment_20240101_01.db",
        {
            "ollama": [
                dict(
                    OLLAMA_ROW,
                    ID=1,
                    status="success",
                    state="finished",
                    total_tokens=100,
                    usage_tokens=100,
                    elapsed_seconds=1.5,
                    response_text='{"verdict": "relevant"}',
                )
            ]
        },
    )

    searches = project / "searches"
    os.makedirs(searches)
    (searches / "s1.yaml").write_text("query: a\n", encoding="utf-8")
    (searches / "s1__results.csv").write_text(
        "search_engine_internal_id;year;title;authors;abstract;isOpenAccess;doi;language;"
        "citationCount;referenceCount;entry_source;pdf_filename;txt_filename;markdown_filename;pdf_downloaded\n"
        "a1;2024;A title;An author;An abstract;True;10.1/x;en;3;4;OpenAlex;;;;False\n",
        encoding="utf-8",
    )

    papers = project / "papers"
    os.makedirs(papers)
    (papers / "a.pdf").write_text("x", encoding="utf-8")

    return project


def _run_notebook_cells(path, workdir, monkeypatch):
    """Execute every code cell in one namespace, minus the IPython magics.

    A real kernel would add ~5s per notebook and pull `nbclient` into the test
    requirements for very little more coverage: this still exercises the
    `import stats` resolution, every call the notebook makes and every name it
    relies on.
    """

    notebook = nbformat.read(str(path), as_version=4)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(builtins, "display", print, raising=False)

    namespace = {"__name__": "__main__", "display": print}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        body = "\n".join(line for line in cell.source.splitlines() if not line.strip().startswith(("%", "!")))
        exec(compile(body, str(path), "exec"), namespace)  # nosec B102 - our own generated cells

    return namespace


@pytest.mark.parametrize("kind", list(NOTEBOOKS))
def test_scaffolded_notebook_runs_end_to_end(projects_dir, monkeypatch, kind):
    project = _project_with_data(projects_dir)
    result = runner.invoke(app, ["analysis", "init", "--pid", PID])
    assert result.exit_code == 0, result.exception

    analysis_dir = project / "analysis"
    namespace = _run_notebook_cells(analysis_dir / NOTEBOOKS[kind], analysis_dir, monkeypatch)

    assert namespace["PROJECT_ID"] == PID
    # Only the experiment notebook exports: a search CSV has no JSON to flatten.
    assert bool(list(analysis_dir.glob("*_flattened.csv"))) is (kind == "experiment")


def test_scaffolded_notebook_runs_against_an_empty_project(projects_dir, monkeypatch):
    """A project with nothing generated must still execute top to bottom."""

    project = projects_dir / PID
    os.makedirs(project)
    assert runner.invoke(app, ["analysis", "init", "--pid", PID]).exit_code == 0

    analysis_dir = project / "analysis"
    namespace = _run_notebook_cells(analysis_dir / NOTEBOOKS["experiment"], analysis_dir, monkeypatch)

    assert len(namespace["df"]) == 0
