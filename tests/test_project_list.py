"""Tests for the `project list` command."""

import os
import re
import time

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.configs import settings

runner = CliRunner()

# The parts table needs more than the 80 columns the runner defaults to.
WIDE = {"COLUMNS": "200"}


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.setattr("llmexer.cli.load_dotenv", lambda *args, **kwargs: None)
    return tmp_path


def _row_columns(output: str) -> list[list[str]]:
    """Return the cells of every data row, found by the date column."""
    rows = []
    for line in output.splitlines():
        if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
            columns = [c.strip() for c in re.split(r"[│┃|]", line)]
            rows.append(columns[1:-1])
    return rows


def test_list_empty(projects_dir):
    """Listing with no projects should print a no-projects message."""
    result = runner.invoke(app, ["project", "list"], env=WIDE)
    assert result.exit_code == 0
    assert "No projects found" in result.output


def test_list_parts(projects_dir):
    """A part counts only when its folder exists and holds something."""
    project_path = projects_dir / "my-project"
    os.makedirs(project_path / "experiment")
    (project_path / "experiment" / "data.csv").write_text("ID;Title;Abstract\n", encoding="utf-8")
    os.makedirs(project_path / "papers")

    result = runner.invoke(app, ["project", "list"], env=WIDE)
    assert result.exit_code == 0

    # #, Name, Created, Search, Experiment, Analysis, Papers
    row = _row_columns(result.output)[0]
    assert row[1] == "my-project"
    assert row[3] == "NO"  # searches/ missing
    assert row[4] == "YES"  # experiment/ holds a file
    assert row[5] == "NO"  # analysis/ missing
    assert row[6] == "NO"  # papers/ exists but is empty


def test_list_alpha_default(projects_dir):
    """Default sort should be alphabetical."""
    for name in ["c-proj", "a-proj", "b-proj"]:
        os.makedirs(projects_dir / name)

    result = runner.invoke(app, ["project", "list"], env=WIDE)
    assert result.exit_code == 0
    assert [row[1] for row in _row_columns(result.output)] == ["a-proj", "b-proj", "c-proj"]


def test_list_sort_by_date_desc(projects_dir):
    """--sort-by date --desc should list the newest project first."""
    for name in ["first", "second"]:
        os.makedirs(projects_dir / name)
        time.sleep(0.01)

    result = runner.invoke(app, ["project", "list", "--sort-by", "date", "--desc"], env=WIDE)
    assert result.exit_code == 0
    assert [row[1] for row in _row_columns(result.output)] == ["second", "first"]


def test_list_highlights_current_project(projects_dir, monkeypatch):
    """The current project is still listed when PROJECT_ID is set."""
    os.makedirs(projects_dir / "current-project")
    os.makedirs(projects_dir / "other-project")
    monkeypatch.setattr(settings, "project_id", "current-project")

    result = runner.invoke(app, ["project", "list"], env=WIDE)
    assert result.exit_code == 0
    assert [row[1] for row in _row_columns(result.output)] == ["current-project", "other-project"]
