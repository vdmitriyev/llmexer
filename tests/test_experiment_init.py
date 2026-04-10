"""Tests for the `experiment init` command."""

import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
    LLMExerException,
)

runner = CliRunner()


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    """Redirect EXPERIMENTS_PATH to a temporary directory for each test."""
    import llmexer.commands.experiment as exp_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(exp_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


@pytest.fixture()
def experiment(experiments_dir):
    """Create a test experiment directory and return (eid, exp_path)."""
    eid = "test-exp"
    exp_path = experiments_dir / eid
    os.makedirs(exp_path)
    return eid, exp_path


def test_init_creates_experiment_subfolder(experiment, experiments_dir):
    """init should create an `experiment/` subfolder inside the experiment."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    assert (exp_path / "experiment").is_dir()


def test_init_creates_prompts_subfolder(experiment, experiments_dir):
    """init should create an `experiment/prompts/` subfolder."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    assert (exp_path / "experiment" / "prompts").is_dir()


def test_init_creates_models_csv(experiment, experiments_dir):
    """init should create models.csv with the correct header and example row."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    models_file = exp_path / "experiment" / "models.csv"
    assert models_file.exists()
    lines = models_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "name;provider;notes"
    assert "llama3.3:latest" in lines[1]
    assert "ollama" in lines[1]


def test_init_creates_data_csv(experiment, experiments_dir):
    """init should create data.csv with the correct header and two example rows."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    data_file = exp_path / "experiment" / "data.csv"
    assert data_file.exists()
    lines = data_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ID;Title;Abstract"
    assert len(lines) == 3  # header + 2 example rows
    assert lines[1].startswith("D01;")
    assert lines[2].startswith("D02;")


def test_init_creates_mapping_csv(experiment, experiments_dir):
    """init should create mapping.csv with the correct header and two example rows."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    mapping_file = exp_path / "experiment" / "mapping.csv"
    assert mapping_file.exists()
    lines = mapping_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "data_id;prompt_id"
    assert lines[1] == "D01;prompt01"


def test_init_creates_prompt_file(experiment, experiments_dir):
    """init should create prompts/prompt01.txt with a template instruction."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    prompt_file = exp_path / "experiment" / "prompts" / "prompt01.txt"
    assert prompt_file.exists()
    content = prompt_file.read_text(encoding="utf-8")
    assert "{{title}}" in content
    assert "{{abstract}}" in content


def test_init_prints_success_message(experiment, experiments_dir):
    """init should print a success message containing the experiment ID."""
    eid, exp_path = experiment

    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code == 0
    assert eid in result.output


def test_init_nonexistent_experiment_raises(experiments_dir):
    """init should raise ExperimentNotExistsException for a non-existent experiment."""
    result = runner.invoke(app, ["experiment", "init", "--eid", "nonexistent"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)


def test_init_nonexistent_experiment_error_message(experiments_dir):
    """The exception message should mention the non-existent experiment ID."""
    result = runner.invoke(app, ["experiment", "init", "--eid", "nonexistent"])

    assert "nonexistent" in str(result.exception)


def test_init_already_initialised_raises(experiment, experiments_dir):
    """Calling init on an already-initialised experiment should raise LLMExerException."""
    eid, exp_path = experiment

    runner.invoke(app, ["experiment", "init", "--eid", eid])
    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_init_already_initialised_error_message(experiment, experiments_dir):
    """The exception message should mention the experiment ID."""
    eid, exp_path = experiment

    runner.invoke(app, ["experiment", "init", "--eid", eid])
    result = runner.invoke(app, ["experiment", "init", "--eid", eid])

    assert eid in str(result.exception)
    assert "already been initialised" in str(result.exception)


def test_init_uses_current_experiment_from_env(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is omitted, init should use EXPERIMENT_ID from the environment."""
    eid = "env-exp"
    exp_path = experiments_dir / eid
    os.makedirs(exp_path)
    monkeypatch.setenv("EXPERIMENT_ID", eid)

    result = runner.invoke(app, ["experiment", "init"])

    assert result.exit_code == 0
    assert (exp_path / "experiment").is_dir()
    assert eid in result.output


def test_init_without_eid_and_no_env_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is omitted and EXPERIMENT_ID is not set, should raise an error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "init"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)


def test_init_eid_overrides_env(experiments_dir, mock_no_dotenv, monkeypatch):
    """When --eid is provided, it should override EXPERIMENT_ID from the environment."""
    env_eid = "env-exp"
    cli_eid = "cli-exp"
    os.makedirs(experiments_dir / env_eid)
    os.makedirs(experiments_dir / cli_eid)
    monkeypatch.setenv("EXPERIMENT_ID", env_eid)

    result = runner.invoke(app, ["experiment", "init", "--eid", cli_eid])

    assert result.exit_code == 0
    # env-exp should NOT have been initialised
    assert not (experiments_dir / env_eid / "experiment").exists()
    # cli-exp should have been initialised
    assert (experiments_dir / cli_eid / "experiment").is_dir()
