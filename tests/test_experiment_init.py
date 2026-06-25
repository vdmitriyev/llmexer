"""Tests for the `experiment init` command."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    LLMExerException,
    ProjectIDRequiredException,
    ProjectNotExistsException,
)

runner = CliRunner()


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
def experiment(projects_dir):
    """Create a test experiment directory and return (pid, exp_path)."""
    pid = "test-exp"
    exp_path = projects_dir / pid
    os.makedirs(exp_path)
    return pid, exp_path


def test_init_creates_experiment_subfolder(experiment, projects_dir):
    """init should create an `experiment/` subfolder inside the experiment."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    assert (exp_path / "experiment").is_dir()


def test_init_creates_prompts_subfolder(experiment, projects_dir):
    """init should create an `experiment/prompts/` subfolder."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    assert (exp_path / "experiment" / "prompts").is_dir()


def test_init_creates_models_csv(experiment, projects_dir):
    """init should create llm-models.csv with the correct header and example row."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    models_file = exp_path / "experiment" / "llm-models.csv"
    assert models_file.exists()
    lines = models_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "name;provider;notes"
    assert "gemma4:31b" in lines[1]
    assert "ollama" in lines[1]


def test_init_creates_data_csv(experiment, projects_dir):
    """init should create data.csv with the correct header and two example rows."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    data_file = exp_path / "experiment" / "data.csv"
    assert data_file.exists()
    lines = data_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ID;Title;Abstract"
    assert len(lines) == 3  # header + 2 example rows
    assert lines[1].startswith("D01;")
    assert lines[2].startswith("D02;")


def test_init_creates_mapping_csv(experiment, projects_dir):
    """init should create mapping.csv with the correct header and two example rows."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    mapping_file = exp_path / "experiment" / "mapping.csv"
    assert mapping_file.exists()
    lines = mapping_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "data_id;prompt_id"
    assert lines[1] == "D01;prompt01"


def test_init_creates_prompt_file(experiment, projects_dir):
    """init should create prompts/prompt01.txt with a template instruction."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    prompt_file = exp_path / "experiment" / "prompts" / "prompt01.txt"
    assert prompt_file.exists()
    content = prompt_file.read_text(encoding="utf-8")
    assert "{{title}}" in content
    assert "{{abstract}}" in content


def test_init_prints_success_message(experiment, projects_dir):
    """init should print a success message containing the experiment ID."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    assert pid in result.output


def test_init_nonexistent_experiment_raises(projects_dir):
    """init should raise ProjectNotExistsException for a non-existent experiment."""
    result = runner.invoke(app, ["experiment", "init", "--pid", "nonexistent"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


def test_init_nonexistent_experiment_error_message(projects_dir):
    """The exception message should mention the non-existent experiment ID."""
    result = runner.invoke(app, ["experiment", "init", "--pid", "nonexistent"])

    assert "nonexistent" in str(result.exception)


def test_init_already_initialised_raises(experiment, projects_dir):
    """Calling init on an already-initialised experiment should raise LLMExerException."""
    pid, exp_path = experiment

    runner.invoke(app, ["experiment", "init", "--pid", pid])
    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_init_already_initialised_error_message(experiment, projects_dir):
    """The exception message should mention the experiment ID."""
    pid, exp_path = experiment

    runner.invoke(app, ["experiment", "init", "--pid", pid])
    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert pid in str(result.exception)
    assert "already been initialised" in str(result.exception)


def test_init_uses_current_experiment_from_env(
    projects_dir, mock_no_dotenv, monkeypatch
):
    """When --pid is omitted, init should use PROJECT_ID from the environment."""
    pid = "env-exp"
    exp_path = projects_dir / pid
    os.makedirs(exp_path)
    monkeypatch.setenv("PROJECT_ID", pid)

    result = runner.invoke(app, ["experiment", "init"])

    assert result.exit_code == 0
    assert (exp_path / "experiment").is_dir()
    assert pid in result.output


def test_init_without_eid_and_no_env_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """When --pid is omitted and PROJECT_ID is not set, should raise an error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "init"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_init_creates_llm_params_csv(experiment, projects_dir):
    """init should create llm-params.csv with the correct header and example rows."""
    pid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--pid", pid])

    assert result.exit_code == 0
    params_file = exp_path / "experiment" / "llm-params.csv"
    assert params_file.exists()
    lines = params_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "provider;model_name;profile_name;temperature;top_p;max_tokens;"
        "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level"
    )
    assert len(lines) >= 2
    assert any("ollama" in line for line in lines[1:])


def test_init_eid_overrides_env(projects_dir, mock_no_dotenv, monkeypatch):
    """When --pid is provided, it should override PROJECT_ID from the environment."""
    env_eid = "env-exp"
    cli_eid = "cli-exp"
    os.makedirs(projects_dir / env_eid)
    os.makedirs(projects_dir / cli_eid)
    monkeypatch.setenv("PROJECT_ID", env_eid)

    result = runner.invoke(app, ["experiment", "init", "--pid", cli_eid])

    assert result.exit_code == 0
    # env-exp should NOT have been initialised
    assert not (projects_dir / env_eid / "experiment").exists()
    # cli-exp should have been initialised
    assert (projects_dir / cli_eid / "experiment").is_dir()
