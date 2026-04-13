"""Tests for the `experiment generate` command."""

import json
import os
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
    LLMExerException,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def initialised_experiment(experiments_dir):
    """Create and initialise a test experiment with standard template files.

    Returns (eid, exp_subdir) where exp_subdir is the 'experiment/' subfolder.
    Uses Jinja2 double-brace syntax in the prompt file.
    """
    eid = "test-exp"
    exp_path = experiments_dir / eid
    exp_subdir = exp_path / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nllama3.3:latest;ollama;local model\n",
        encoding="utf-8",
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\n"
        "D01;Sample Paper Title One;This is the abstract of the first sample paper.\n"
        "D02;Sample Paper Title Two;This is the abstract of the second sample paper.\n",
        encoding="utf-8",
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;prompt01\n",
        encoding="utf-8",
    )
    (prompts_dir / "prompt01.txt").write_text(
        "Here is the title: {{title}}.\n\nHere is the abstract: {{abstract}}.\n\nCount words.",
        encoding="utf-8",
    )
    return eid, exp_subdir


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_generate_creates_output_csv(initialised_experiment, experiments_dir):
    """generate should create a experiment_*.csv file in the experiment/ subfolder."""
    eid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code == 0
    csv_files = list(exp_subdir.glob("experiment_*.csv"))
    assert len(csv_files) == 1


def test_generate_output_has_correct_columns(initialised_experiment, experiments_dir):
    """The output CSV should have exactly the required 8 columns in order."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert list(df.columns) == [
        "ID",
        "code",
        "prompt",
        "original_data",
        "model_name",
        "provider_name",
        "prompt_hash",
        "original_data_hash",
    ]


def test_generate_code_field_format(initialised_experiment, experiments_dir):
    """code field should be DATAID_PROMPTID_MODELNAME."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert df.iloc[0]["code"] == "D01_prompt01_llama3.3:latest"


def test_generate_row_count(initialised_experiment, experiments_dir):
    """One mapping row × one model = one result row."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert len(df) == 1


def test_generate_prompt_is_rendered(initialised_experiment, experiments_dir):
    """The prompt column should contain the rendered template (variables substituted)."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert "Sample Paper Title One" in df.iloc[0]["prompt"]
    assert "abstract of the first" in df.iloc[0]["prompt"]
    assert "{{title}}" not in df.iloc[0]["prompt"]
    assert "{{abstract}}" not in df.iloc[0]["prompt"]


def test_generate_model_name_and_provider(initialised_experiment, experiments_dir):
    """model_name and provider_name columns should match models.csv."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert df.iloc[0]["model_name"] == "llama3.3:latest"
    assert df.iloc[0]["provider_name"] == "ollama"


def test_generate_original_data_is_json(initialised_experiment, experiments_dir):
    """original_data column should be a parseable JSON string containing the data row fields."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    original_data = json.loads(df.iloc[0]["original_data"])
    assert original_data["ID"] == "D01"
    assert "Title" in original_data
    assert "Abstract" in original_data


def test_generate_prompt_hash_is_sha256(initialised_experiment, experiments_dir):
    """prompt_hash should be a 64-character hex string (SHA256)."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert len(df.iloc[0]["prompt_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in df.iloc[0]["prompt_hash"])


def test_generate_original_data_hash_is_sha256(initialised_experiment, experiments_dir):
    """original_data_hash should be a 64-character hex string (SHA256)."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert len(df.iloc[0]["original_data_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in df.iloc[0]["original_data_hash"])


def test_generate_multiple_models_multiple_rows(experiments_dir):
    """With 2 mapping rows and 2 models, should produce 4 result rows."""
    eid = "multi-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nmodel-a;provider-a;\nmodel-b;provider-b;\n",
        encoding="utf-8",
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;Title One;Abstract One.\nD02;Title Two;Abstract Two.\n",
        encoding="utf-8",
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;prompt01\nD02;prompt01\n",
        encoding="utf-8",
    )
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])
    assert result.exit_code == 0

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert len(df) == 4  # 2 data rows × 2 models
    assert list(df["ID"]) == [1, 2, 3, 4]
    # rows are sorted by model order from models.csv, then by mapping order within each model
    assert list(df["model_name"]) == ["model-a", "model-a", "model-b", "model-b"]


def test_generate_sorted_by_model_order(experiments_dir):
    """Output rows should be grouped by model in models.csv order (model-first, then mapping order)."""
    eid = "sort-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    # model-b is listed second in models.csv — its rows must appear after model-a's
    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nmodel-a;p;\nmodel-b;p;\n",
        encoding="utf-8",
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;T1;A1.\nD02;T2;A2.\n",
        encoding="utf-8",
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;prompt01\nD02;prompt01\n",
        encoding="utf-8",
    )
    (prompts_dir / "prompt01.txt").write_text("{{title}}", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])
    assert result.exit_code == 0

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")

    # Expected order: D01+model-a, D02+model-a, D01+model-b, D02+model-b
    assert list(df["model_name"]) == ["model-a", "model-a", "model-b", "model-b"]
    assert list(df["ID"]) == [1, 2, 3, 4]


def test_generate_row_id_format(initialised_experiment, experiments_dir):
    """Each row ID should be an incrementing integer starting from 1."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert df.iloc[0]["ID"] == 1


def test_generate_output_filename_format(initialised_experiment, experiments_dir):
    """Output filename should start with 'experiment_' followed by YYYYMMDD-XXXXXXXX."""
    eid, exp_subdir = initialised_experiment

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_files = list(exp_subdir.glob("experiment_*.csv"))
    assert len(csv_files) == 1
    assert csv_files[0].name.startswith("experiment_")


def test_generate_prints_success_message(initialised_experiment, experiments_dir):
    """generate should print a success message with the row count."""
    eid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code == 0
    assert "1" in result.output


def test_generate_prompt_hash_deterministic(experiments_dir):
    """Two rows from the same prompt+data but different models share the same prompt_hash."""
    eid = "hash-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nmodel-a;p;\nmodel-b;p;\n",
        encoding="utf-8",
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;Sample Title;Sample Abstract.\n",
        encoding="utf-8",
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;prompt01\n",
        encoding="utf-8",
    )
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")

    runner.invoke(app, ["experiment", "generate", "--eid", eid])

    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert list(df["ID"]) == [1, 2]
    assert df.iloc[0]["prompt_hash"] == df.iloc[1]["prompt_hash"]
    assert df.iloc[0]["original_data_hash"] == df.iloc[1]["original_data_hash"]


# ---------------------------------------------------------------------------
# Dry run tests
# ---------------------------------------------------------------------------


def test_generate_dry_run_no_files_written(initialised_experiment, experiments_dir):
    """With --dry-run, no output CSV should be written."""
    eid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["--dry-run", "experiment", "generate", "--eid", eid])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert not list(exp_subdir.glob("experiment_*.csv"))


def test_generate_dry_run_shows_row_count(initialised_experiment, experiments_dir):
    """With --dry-run, output should indicate how many rows would be written."""
    eid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["--dry-run", "experiment", "generate", "--eid", eid])

    assert result.exit_code == 0
    assert "1" in result.output


# ---------------------------------------------------------------------------
# Error-case tests
# ---------------------------------------------------------------------------


def test_generate_nonexistent_experiment_raises(experiments_dir):
    """generate on a nonexistent experiment should raise ExperimentNotExistsException."""
    result = runner.invoke(app, ["experiment", "generate", "--eid", "no-such-exp"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)


def test_generate_uninitialised_experiment_raises(experiments_dir):
    """generate on an experiment without 'experiment/' subfolder should raise LLMExerException."""
    eid = "bare-exp"
    os.makedirs(experiments_dir / eid)

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "not been initialised" in str(result.exception)


def test_generate_missing_models_csv_raises(experiments_dir):
    """Missing models.csv should raise LLMExerException."""
    eid = "no-models-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)
    (exp_subdir / "data.csv").write_text("ID;Title\nD01;T\n", encoding="utf-8")
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;p\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "models.csv" in str(result.exception)


def test_generate_missing_data_csv_raises(experiments_dir):
    """Missing data.csv should raise LLMExerException."""
    eid = "no-data-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)
    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nm;p;\n", encoding="utf-8"
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;p\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "data.csv" in str(result.exception)


def test_generate_missing_mapping_csv_raises(experiments_dir):
    """Missing mapping.csv should raise LLMExerException."""
    eid = "no-mapping-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)
    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nm;p;\n", encoding="utf-8"
    )
    (exp_subdir / "data.csv").write_text("ID;Title\nD01;T\n", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "mapping.csv" in str(result.exception)


def test_generate_missing_data_id_skips_row(experiments_dir):
    """When a mapping data_id is not in data.csv, that row is skipped with a warning."""
    eid = "bad-data-id-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nm;p;\n", encoding="utf-8"
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;Title;Abstract.\n", encoding="utf-8"
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;prompt01\nBAD-ID;prompt01\n", encoding="utf-8"
    )
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code == 0
    assert "Warning" in result.output
    csv_file = next(exp_subdir.glob("experiment_*.csv"))
    df = pd.read_csv(csv_file, sep=";")
    assert len(df) == 1  # Only D01 row is included


def test_generate_missing_prompt_file_skips_row(experiments_dir):
    """When a mapping prompt_id has no matching .txt file, that row is skipped."""
    eid = "bad-prompt-id-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nm;p;\n", encoding="utf-8"
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;Title;Abstract.\n", encoding="utf-8"
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;nonexistent-prompt\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["experiment", "generate", "--eid", eid])

    assert result.exit_code == 0
    assert "Warning" in result.output
    assert not list(exp_subdir.glob("experiment_*.csv"))


def test_generate_uses_current_experiment_from_env(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is omitted, generate should use EXPERIMENT_ID from the environment."""
    eid = "env-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "models.csv").write_text(
        "name;provider;notes\nm;p;\n", encoding="utf-8"
    )
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;Title;Abstract.\n", encoding="utf-8"
    )
    (exp_subdir / "mapping.csv").write_text(
        "data_id;prompt_id\nD01;prompt01\n", encoding="utf-8"
    )
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")

    monkeypatch.setenv("EXPERIMENT_ID", eid)

    result = runner.invoke(app, ["experiment", "generate"])
    assert result.exit_code == 0
    assert list(exp_subdir.glob("experiment_*.csv"))


def test_generate_without_eid_and_no_env_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is omitted and EXPERIMENT_ID is not set, should raise an error."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "generate"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)
