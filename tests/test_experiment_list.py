"""Tests for the `experiment list` command."""

import os
import re
import time

import pytest
from typer.testing import CliRunner

from llmexer.cli import app

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.commands.experiment as exp_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(exp_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


def _names_from_output(output: str) -> list[str]:
    """Extract experiment names from table rows (lines containing the name column value)."""
    # Each data row contains a row number, the name, and a date — grab lines with a date pattern.
    rows = []
    for line in output.splitlines():
        m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
        if m:
            # Name sits between the row number and the date; strip table border characters.
            # Strip Rich markup / box-drawing chars and split on whitespace runs.
            clean = re.sub(r"[│┃|]", " ", line)
            parts = clean.split()
            # parts: [row_num, ...name_parts..., date, time]
            # date and time are the last two tokens; row_num is the first
            name = " ".join(parts[1:-2])
            rows.append(name)
    return rows


def test_list_empty(experiments_dir):
    """Listing with no experiments should print a no-experiments message."""
    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "No experiments found" in result.output


def test_list_alpha_default(experiments_dir):
    """Default sort should be alphabetical."""
    for name in ["c-exp", "a-exp", "b-exp"]:
        os.makedirs(experiments_dir / name)

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["a-exp", "b-exp", "c-exp"]


def test_list_alpha_explicit(experiments_dir):
    """--sort-by alpha should produce alphabetical order."""
    for name in ["z-exp", "m-exp", "a-exp"]:
        os.makedirs(experiments_dir / name)

    result = runner.invoke(app, ["experiment", "list", "--sort-by", "alpha"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["a-exp", "m-exp", "z-exp"]


def test_list_sort_by_date(experiments_dir):
    """--sort-by date should produce chronological (oldest-first) order."""
    for name in ["first-exp", "second-exp", "third-exp"]:
        os.makedirs(experiments_dir / name)
        time.sleep(0.02)

    result = runner.invoke(app, ["experiment", "list", "--sort-by", "date"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["first-exp", "second-exp", "third-exp"]


def test_list_invalid_sort_by(experiments_dir):
    """Passing an unknown --sort-by value should fail."""
    result = runner.invoke(app, ["experiment", "list", "--sort-by", "invalid"])
    assert result.exit_code != 0


def test_list_alpha_desc(experiments_dir):
    """--desc should reverse alphabetical order."""
    for name in ["a-exp", "b-exp", "c-exp"]:
        os.makedirs(experiments_dir / name)

    result = runner.invoke(app, ["experiment", "list", "--desc"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["c-exp", "b-exp", "a-exp"]


def test_list_sort_by_date_desc(experiments_dir):
    """--sort-by date --desc should produce newest-first order."""
    for name in ["first-exp", "second-exp", "third-exp"]:
        os.makedirs(experiments_dir / name)
        time.sleep(0.02)

    result = runner.invoke(app, ["experiment", "list", "--sort-by", "date", "--desc"])
    assert result.exit_code == 0
    assert _names_from_output(result.output) == ["third-exp", "second-exp", "first-exp"]


def test_list_table_columns(experiments_dir):
    """Output should contain # , Name, and Created column headers."""
    os.makedirs(experiments_dir / "my-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert "#" in result.output
    assert "Name" in result.output
    assert "Created" in result.output


def test_list_date_format(experiments_dir):
    """Created column should display dates in YYYY-MM-DD HH:MM:SS format."""
    os.makedirs(experiments_dir / "dated-exp")

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.output)


def test_list_numbering(experiments_dir):
    """Rows should be numbered starting from 1."""
    for name in ["a-exp", "b-exp", "c-exp"]:
        os.makedirs(experiments_dir / name)

    result = runner.invoke(app, ["experiment", "list"])
    assert result.exit_code == 0
    for i in ["1", "2", "3"]:
        assert i in result.output
