"""Tests for how the `llmexer` console entry point shows llmexer's own errors."""

import sys
from unittest.mock import Mock

import pytest

import llmexer.cli as cli
from llmexer.exceptions import LLMExerException, ProjectNotExistsException


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))
    monkeypatch.setattr(cli.settings, "verbose", False)
    monkeypatch.setenv("COLUMNS", "300")


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["llmexer", *args])
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    return exit_info.value.code


def test_error_is_a_message_not_a_traceback(monkeypatch, capsys, tmp_path):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))

    code = _run(monkeypatch, "experiment", "view", "--pid", "no-such-project")

    err = capsys.readouterr().err
    assert code == 1
    assert "Error: Project 'no-such-project' does not exist." in err
    assert "ProjectNotExistsException" in err
    assert "Traceback" not in err and "╭" not in err


def test_multiline_message_keeps_its_lines(monkeypatch, capsys):
    message = "First line.\nCommand for the rename:\nmv /a/experiment /a/experiments"
    monkeypatch.setattr(cli, "app", Mock(side_effect=LLMExerException(message)))

    code = _run(monkeypatch)

    err = capsys.readouterr().err.splitlines()
    assert code == 1
    assert err[0] == "Error: First line."
    assert err[1] == "Command for the rename:"
    assert err[2] == "  mv /a/experiment /a/experiments"


def test_verbose_keeps_the_exception(monkeypatch):
    monkeypatch.setattr(cli, "app", Mock(side_effect=ProjectNotExistsException("gone")))
    monkeypatch.setattr(cli.settings, "verbose", True)
    monkeypatch.setattr(sys, "argv", ["llmexer"])

    with pytest.raises(ProjectNotExistsException):
        cli.run()


def test_other_exceptions_are_not_caught(monkeypatch):
    monkeypatch.setattr(cli, "app", Mock(side_effect=ValueError("bug")))
    monkeypatch.setattr(sys, "argv", ["llmexer"])

    with pytest.raises(ValueError):
        cli.run()


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "Run `experiment init --pid x` first.",
            "Run [bold yellow]experiment init --pid x[/bold yellow] first.",
        ),
        ("Project 'p1' does not exist.", "Project [cyan]'p1'[/cyan] does not exist."),
        ("The project's folder is gone.", "The project's folder is gone."),
        ("mv /a/experiment /a/experiments", "  [bold yellow]mv /a/experiment /a/experiments[/bold yellow]"),
        ("llmexer experiment init", "  [bold yellow]llmexer experiment init[/bold yellow]"),
        ("Values [x] stay literal", "Values \\[x] stay literal"),
    ],
)
def test_error_line_colors(line, expected):
    assert cli._style_error_line(line) == expected
