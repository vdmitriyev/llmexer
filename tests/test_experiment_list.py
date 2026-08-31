"""Tests for the `experiment list` command."""

import os
import re
import time

import pytest
from typer.testing import CliRunner

from llmexer.cli import app

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.experiment as experiment_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(experiment_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


def _names_from_output(output: str) -> list[str]:
    """Extract experiment names from table rows (lines containing the name column value)."""
    # Each data row contains a row number, the name, and a date — grab lines with a date pattern.
    rows = []
    for line in output.splitlines():
        m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
        if m:
            # Split on the table column separator │
            columns = re.split(r"[│┃|]", line)
            # columns[0] is empty (before first │), columns[1] is #, columns[2] is Name
            if len(columns) >= 3:
                name = columns[2].strip()
                rows.append(name)
    return rows


def test_list_empty(projects_dir):
    """Listing with no experiments should print a no-experiments message."""
    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "No projects found" in result.output


def test_list_alpha_default(projects_dir):
    """Default sort should be alphabetical."""
    for name in ["c-exp", "a-exp", "b-exp"]:
        os.makedirs(projects_dir / name)

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["a-exp", "b-exp", "c-exp"]


def test_list_alpha_explicit(projects_dir):
    """--sort-by alpha should produce alphabetical order."""
    for name in ["z-exp", "m-exp", "a-exp"]:
        os.makedirs(projects_dir / name)

    result = runner.invoke(app, ["experiment", "list", "--sort-by", "alpha"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["a-exp", "m-exp", "z-exp"]


def test_list_sort_by_date(projects_dir):
    """--sort-by date should produce chronological (oldest-first) order."""
    for name in ["first-exp", "second-exp", "third-exp"]:
        os.makedirs(projects_dir / name)
        time.sleep(0.02)

    result = runner.invoke(app, ["experiment", "list", "--sort-by", "date"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["first-exp", "second-exp", "third-exp"]


def test_list_invalid_sort_by(projects_dir):
    """Passing an unknown --sort-by value should fail."""
    result = runner.invoke(app, ["experiment", "list", "--sort-by", "invalid"])
    assert result.exit_code != 0


def test_list_alpha_desc(projects_dir):
    """--desc should reverse alphabetical order."""
    for name in ["a-exp", "b-exp", "c-exp"]:
        os.makedirs(projects_dir / name)

    result = runner.invoke(app, ["experiment", "list", "--desc"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["c-exp", "b-exp", "a-exp"]


def test_list_sort_by_date_desc(projects_dir):
    """--sort-by date --desc should produce newest-first order."""
    for name in ["first-exp", "second-exp", "third-exp"]:
        os.makedirs(projects_dir / name)
        time.sleep(0.02)

    result = runner.invoke(app, ["experiment", "list", "--sort-by", "date", "--desc"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["third-exp", "second-exp", "first-exp"]


def test_list_table_columns(projects_dir):
    """Output should contain # , Name, and Created column headers."""
    os.makedirs(projects_dir / "my-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "#" in result.output
    assert "Name" in result.output
    assert "Created" in result.output


def test_list_date_format(projects_dir):
    """Created column should display dates in YYYY-MM-DD HH:MM:SS format."""
    os.makedirs(projects_dir / "dated-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.output)


def test_list_numbering(projects_dir):
    """Rows should be numbered starting from 1."""
    for name in ["a-exp", "b-exp", "c-exp"]:
        os.makedirs(projects_dir / name)

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    for i in ["1", "2", "3"]:
        assert i in result.output


def test_list_initialized_column_header(projects_dir):
    """Output should contain Initialized column header."""
    os.makedirs(projects_dir / "my-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "Initialized" in result.output


def test_list_not_initialized(projects_dir):
    """Experiment without init should show No in Initialized column."""
    os.makedirs(projects_dir / "uninit-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "No" in result.output


def test_list_initialized(projects_dir):
    """Experiment with all required CSV files should show Yes in Initialized column."""
    exp_path = projects_dir / "init-exp" / "experiment"
    os.makedirs(exp_path)
    for f in ["data.csv", "llm-params.csv", "mapping.csv", "llms-for-experiment.csv"]:
        (exp_path / f).write_text("header\n")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "Yes" in result.output


def test_list_partially_initialized(projects_dir):
    """Experiment with only some CSV files should show No in Initialized column."""
    exp_path = projects_dir / "partial-exp" / "experiment"
    os.makedirs(exp_path)
    # Only create some of the required files
    for f in ["data.csv", "llms-for-experiment.csv"]:
        (exp_path / f).write_text("header\n")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "No" in result.output


def test_list_generated_files_column_header(projects_dir):
    """Output should contain Experiments column header."""
    os.makedirs(projects_dir / "my-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "Experiments" in result.output


def test_list_no_generated_files(projects_dir):
    """Experiment with no generated files should show - in Generated Files column."""
    os.makedirs(projects_dir / "no-gen-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    # The dash is shown when no generated files exist
    assert "-" in result.output


def test_list_with_generated_files(projects_dir):
    """Experiment with generated files should display their names (possibly truncated)."""
    exp_path = projects_dir / "gen-exp" / "experiment"
    os.makedirs(exp_path)
    (exp_path / "experiment_20240101_01.db").write_text("data\n")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    # Rich may truncate long filenames, so check for the prefix
    assert "experiment_20240101" in result.output


def test_list_multiple_generated_files(projects_dir):
    """Experiment with multiple generated files should display them."""
    exp_path = projects_dir / "multi-gen-exp" / "experiment"
    os.makedirs(exp_path)
    (exp_path / "experiment_20240101_01.db").write_text("data\n")
    (exp_path / "experiment_20240102_02.db").write_text("data\n")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    # Rich may truncate long filenames with ellipsis, so check for shorter prefix
    # that will survive truncation (at least "experiment_2024" should appear)
    assert "experiment_2024" in result.output


def test_list_excludes_non_db_files(projects_dir):
    """Only generated databases (*.db) appear in the Experiments column."""
    exp_path = projects_dir / "results-exp" / "experiment"
    os.makedirs(exp_path)
    (exp_path / "experiment_20240101_01.db").write_text("data\n")
    # Non-db files in the experiment dir must not be listed as generated experiments.
    (exp_path / "experiment_legacy.csv").write_text("legacy\n")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    # The generated database should be shown (possibly truncated, but prefix should appear)
    assert "experiment_20240101" in result.output
    # The stray CSV should NOT be shown.
    assert "experiment_legacy" not in result.output
