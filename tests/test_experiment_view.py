"""Tests for the `experiment view` command (alias `show`)."""

import os
import re
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException

runner = CliRunner()

_LLM_PARAMS_CSV = (
    "provider;model_name;profile_name;temperature;top_p;max_tokens;"
    "ollama_context_window;ollama_repeat_penalty;openai_seed\n"
    "ollama;llama3.3:latest;ollama-default;0.7;1.0;512;4096;;\n"
    "openai;gpt-4o;openai-default;0.2;1.0;256;;;42\n"
    "acme;acme-1;acme-default;0.5;0.9;128;8192;1.1;7\n"
)

_LLMS_CSV = (
    "provider;model_name;profile_name;notes\n"
    "ollama;llama3.3:latest;ollama-default;local model\n"
    "openai;gpt-4o;openai-default;\n"
)

_DATA_CSV = (
    "ID;Title;Abstract\n" "D01;One;First.\n" "D02;Two;Second.\n" "D02;Two again;Duplicate.\n" "D03;Three;Third.\n"
)

_MAPPING_CSV = "data_id;prompt_id\n" "D01;prompt01\n" "D02;prompt01\n" "D03;prompt01\n" "D01;prompt02\n"


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def mock_no_dotenv(monkeypatch):
    """Keep a developer's .env out of the tests."""
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))


@pytest.fixture()
def experiment(projects_dir):
    """A project whose experiment/ folder holds all the files the view reads."""
    pid = "test-view"
    exp_subdir = projects_dir / pid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "llm-params.csv").write_text(_LLM_PARAMS_CSV, encoding="utf-8")
    (exp_subdir / "llms-for-experiment.csv").write_text(_LLMS_CSV, encoding="utf-8")
    (exp_subdir / "data.csv").write_text(_DATA_CSV, encoding="utf-8")
    (exp_subdir / "mapping.csv").write_text(_MAPPING_CSV, encoding="utf-8")
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")
    (prompts_dir / "prompt02.txt").write_text("Abstract: {{abstract}}.", encoding="utf-8")

    return pid, exp_subdir


def _invoke(*args):
    return runner.invoke(app, ["experiment", "view", *args], env={"COLUMNS": "300"})


def _table_block(output: str, provider: str) -> str:
    """The lines of one provider's table, from its title to the next title or hint."""
    match = re.search(rf"Provider: {provider} *\n(.*?)(?=Provider: |To edit use:)", output, re.S)
    assert match, f"no table for provider '{provider}'"
    return match.group(1)


def test_view_default_shows_params_per_provider(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid)

    assert result.exit_code == 0, result.output
    ollama = _table_block(result.output, "ollama")
    openai = _table_block(result.output, "openai")
    assert "ollama_context_window" in ollama and "openai_seed" not in ollama
    assert "openai_seed" in openai and "ollama_context_window" not in openai
    assert "4096" in ollama and "4096.0" not in ollama
    assert "llms-for-experiment.csv" not in result.output


def test_view_params_column_order(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid)

    header = next(line for line in _table_block(result.output, "ollama").splitlines() if "temperature" in line)
    order = ["provider", "model_name", "profile_name", "temperature", "top_p", "max_tokens", "ollama_context_window"]
    positions = [header.index(column) for column in order]
    assert positions == sorted(positions)


def test_view_default_equals_params_flag(experiment):
    pid, _ = experiment

    assert _invoke("--pid", pid).output == _invoke("--pid", pid, "--params").output


def test_view_unknown_provider_shows_common_columns_only(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid)

    acme = _table_block(result.output, "acme")
    assert "max_tokens" in acme
    assert "ollama_context_window" not in acme and "openai_seed" not in acme


def test_view_params_edit_hint(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid)

    expected = os.path.join(".projects", pid, "experiment", "llm-params.csv")
    assert f"To edit use:\nnano {expected}" in result.output


def test_view_llms(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid, "--llms")

    assert result.exit_code == 0, result.output
    assert "llama3.3:latest" in result.output and "gpt-4o" in result.output
    assert "local model" in result.output
    assert "Provider: ollama" not in result.output
    assert f"nano {os.path.join('.projects', pid, 'experiment', 'llms-for-experiment.csv')}" in result.output


def test_view_data_counts(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid, "--data")

    assert result.exit_code == 0, result.output
    assert re.search(r"Unique IDs in data\.csv\s*│\s*3\s*│", result.output)
    assert re.search(r"Prompts in prompts/\s*│\s*2\s*│", result.output)
    assert re.search(r"Mapping: prompt01\s*│\s*3\s*│", result.output)
    assert re.search(r"Mapping: prompt02\s*│\s*1\s*│", result.output)
    assert f"nano {os.path.join('.projects', pid, 'experiment', 'data.csv')}" in result.output
    assert f"nano {os.path.join('.projects', pid, 'experiment', 'mapping.csv')}" in result.output


def test_view_combined_flags(experiment):
    pid, _ = experiment

    result = _invoke("--pid", pid, "--params", "--llms", "--data")

    assert result.exit_code == 0, result.output
    assert "Provider: ollama" in result.output
    assert "llms-for-experiment.csv" in result.output
    assert "Unique IDs in data.csv" in result.output


def test_show_alias(experiment):
    pid, _ = experiment

    result = runner.invoke(app, ["exp", "show", "--pid", pid, "--llms"], env={"COLUMNS": "300"})

    assert result.exit_code == 0, result.output
    assert "gpt-4o" in result.output


def test_view_missing_file(experiment):
    pid, exp_subdir = experiment
    os.remove(exp_subdir / "llm-params.csv")

    result = _invoke("--pid", pid)

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "llm-params.csv" in str(result.exception)
