"""Tests for the `experiment map` command."""

import os
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import LLMExerException
from tests.db_helpers import find_db

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


_LLM_PARAMS_HEADER = (
    "provider;model_name;profile_name;temperature;top_p;max_tokens;"
    "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
)
_LLM_PARAMS_ROW = "ollama;llama3.3:latest;ollama-default;0.7;1.0;512;4096;1.1;;;;\n"

_DATA_CSV = (
    "ID;Title;Abstract\n"
    "D01;Sample Paper Title One;This is the abstract of the first sample paper.\n"
    "D02;Sample Paper Title Two;This is the abstract of the second sample paper.\n"
    "D03;Sample Paper Title Three;This is the abstract of the third sample paper.\n"
)


@pytest.fixture()
def initialised_experiment(projects_dir):
    """Create a project with data.csv and two prompt templates, but no mapping.csv."""
    pid = "test-map"
    exp_subdir = projects_dir / pid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "data.csv").write_text(_DATA_CSV, encoding="utf-8")
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")
    (prompts_dir / "prompt02.txt").write_text("Abstract: {{abstract}}.", encoding="utf-8")

    return pid, exp_subdir


def _read_mapping(exp_subdir):
    return pd.read_csv(exp_subdir / "mapping.csv", sep=";", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------


def test_map_without_prompt_uses_all_prompts(initialised_experiment, mock_no_dotenv):
    """With no --prompt, every prompts/*.txt is paired with every data row."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert result.exception is None

    df = _read_mapping(exp_subdir)
    assert list(df.columns) == ["data_id", "prompt_id"]
    assert len(df) == 6  # 3 data rows x 2 prompts
    # Prompt-major: all data rows for prompt01, then all for prompt02.
    assert list(df["prompt_id"]) == ["prompt01"] * 3 + ["prompt02"] * 3
    assert list(df["data_id"]) == ["D01", "D02", "D03"] * 2


def test_map_with_single_prompt(initialised_experiment, mock_no_dotenv):
    """A single --prompt maps only that prompt."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt02"])

    assert result.exit_code == 0, result.output
    df = _read_mapping(exp_subdir)
    assert len(df) == 3
    assert set(df["prompt_id"]) == {"prompt02"}


def test_map_repeated_and_comma_separated_prompts_agree(initialised_experiment, mock_no_dotenv):
    """`--prompt a --prompt b` and `--prompt a,b` produce the same mapping."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt01", "--prompt", "prompt02"])
    assert result.exit_code == 0, result.output
    repeated = (exp_subdir / "mapping.csv").read_text(encoding="utf-8")

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt01,prompt02"])
    assert result.exit_code == 0, result.output
    comma = (exp_subdir / "mapping.csv").read_text(encoding="utf-8")

    assert repeated == comma


def test_map_accepts_txt_extension_and_strips_it(initialised_experiment, mock_no_dotenv):
    """`prompt01.txt` resolves to the extension-less ID `generate` expects."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt01.txt"])

    assert result.exit_code == 0, result.output
    df = _read_mapping(exp_subdir)
    assert set(df["prompt_id"]) == {"prompt01"}


def test_map_deduplicates_prompts(initialised_experiment, mock_no_dotenv):
    """The same prompt named twice is only mapped once."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt01,prompt01.txt"])

    assert result.exit_code == 0, result.output
    assert len(_read_mapping(exp_subdir)) == 3


def test_map_preserves_prompt_order_given(initialised_experiment, mock_no_dotenv):
    """Prompts are written in the order they were passed, not sorted."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt02,prompt01"])

    assert result.exit_code == 0, result.output
    df = _read_mapping(exp_subdir)
    assert list(df["prompt_id"]) == ["prompt02"] * 3 + ["prompt01"] * 3


# ---------------------------------------------------------------------------
# Backup behaviour
# ---------------------------------------------------------------------------


def test_map_backs_up_existing_mapping(initialised_experiment, mock_no_dotenv):
    """An existing mapping.csv is copied to mapping_backup_<date>_<NN>.csv."""
    pid, exp_subdir = initialised_experiment
    old_content = "data_id;prompt_id\nD01;prompt01\n"
    (exp_subdir / "mapping.csv").write_text(old_content, encoding="utf-8")

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code == 0, result.output
    backups = list(exp_subdir.glob("mapping_backup_*.csv"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == old_content
    assert len(_read_mapping(exp_subdir)) == 6

    output = " ".join(result.output.split())
    assert "backed up previous mapping.csv" in output


def test_map_second_run_creates_second_backup(initialised_experiment, mock_no_dotenv):
    """Repeated runs on the same day get their own numbered backups."""
    pid, exp_subdir = initialised_experiment

    for _ in range(3):
        result = runner.invoke(app, ["experiment", "map", "--pid", pid])
        assert result.exit_code == 0, result.output

    # First run had nothing to back up; the next two each produced one.
    assert len(list(exp_subdir.glob("mapping_backup_*.csv"))) == 2


def test_map_without_existing_mapping_creates_no_backup(initialised_experiment, mock_no_dotenv):
    """A first run reports no backup note and leaves no backup file."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert list(exp_subdir.glob("mapping_backup_*.csv")) == []
    assert "backed up" not in " ".join(result.output.split())


def test_map_does_not_back_up_data_csv(initialised_experiment, mock_no_dotenv):
    """`map` touches mapping.csv only — data.csv is left exactly as it was."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert (exp_subdir / "data.csv").read_text(encoding="utf-8") == _DATA_CSV
    assert list(exp_subdir.glob("data_backup_*.csv")) == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_map_missing_prompt_aborts_and_keeps_mapping(initialised_experiment, mock_no_dotenv):
    """A named prompt that does not exist aborts, leaving mapping.csv untouched."""
    pid, exp_subdir = initialised_experiment
    old_content = "data_id;prompt_id\nD01;prompt01\n"
    (exp_subdir / "mapping.csv").write_text(old_content, encoding="utf-8")

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "prompt01,nope"])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "nope.txt" in str(result.exception)
    assert (exp_subdir / "mapping.csv").read_text(encoding="utf-8") == old_content
    assert list(exp_subdir.glob("mapping_backup_*.csv")) == []


def test_map_reports_all_missing_prompts_at_once(initialised_experiment, mock_no_dotenv):
    """Every missing prompt name is listed in a single error."""
    pid, _ = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "nope", "--prompt", "alsonope"])

    assert result.exit_code != 0
    message = str(result.exception)
    assert "nope.txt" in message
    assert "alsonope.txt" in message


def test_map_rejects_path_traversal(initialised_experiment, mock_no_dotenv):
    """A prompt name may not escape the prompts/ folder."""
    pid, _ = initialised_experiment

    result = runner.invoke(app, ["experiment", "map", "--pid", pid, "--prompt", "../data"])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "Invalid prompt name" in str(result.exception)


def test_map_errors_when_prompts_folder_is_empty(initialised_experiment, mock_no_dotenv):
    """With no --prompt and no templates at all, the command explains what to do."""
    pid, exp_subdir = initialised_experiment
    for prompt_file in (exp_subdir / "prompts").glob("*.txt"):
        prompt_file.unlink()

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "No prompt templates found" in str(result.exception)


def test_map_errors_when_data_csv_missing(initialised_experiment, mock_no_dotenv):
    """A missing data.csv is reported in the same style as `generate`."""
    pid, exp_subdir = initialised_experiment
    (exp_subdir / "data.csv").unlink()

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "data.csv" in str(result.exception)


def test_map_errors_when_data_csv_has_no_id_column(initialised_experiment, mock_no_dotenv):
    """data.csv without an ID column cannot be mapped."""
    pid, exp_subdir = initialised_experiment
    (exp_subdir / "data.csv").write_text("Title;Abstract\nT1;A1\n", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "'ID' column" in str(result.exception)


def test_map_warns_on_empty_data_csv(initialised_experiment, mock_no_dotenv):
    """A header-only data.csv warns and leaves mapping.csv alone."""
    pid, exp_subdir = initialised_experiment
    (exp_subdir / "data.csv").write_text("ID;Title;Abstract\n", encoding="utf-8")
    old_content = "data_id;prompt_id\nD01;prompt01\n"
    (exp_subdir / "mapping.csv").write_text(old_content, encoding="utf-8")

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert "Warning" in " ".join(result.output.split())
    assert (exp_subdir / "mapping.csv").read_text(encoding="utf-8") == old_content


def test_map_errors_when_project_not_initialised(projects_dir, mock_no_dotenv):
    """A project without an experiment/ folder is rejected."""
    pid = "not-initialised"
    os.makedirs(projects_dir / pid)

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "has not been initialised" in str(result.exception)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_map_dry_run_writes_nothing(initialised_experiment, mock_no_dotenv):
    """--dry-run reports the row count without writing or backing up."""
    pid, exp_subdir = initialised_experiment
    old_content = "data_id;prompt_id\nD01;prompt01\n"
    (exp_subdir / "mapping.csv").write_text(old_content, encoding="utf-8")

    result = runner.invoke(app, ["--dry-run", "experiment", "map", "--pid", pid])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Dry run:" in output
    assert "would write 6 row(s)" in output
    assert (exp_subdir / "mapping.csv").read_text(encoding="utf-8") == old_content
    assert list(exp_subdir.glob("mapping_backup_*.csv")) == []


# ---------------------------------------------------------------------------
# Integration with `generate`
# ---------------------------------------------------------------------------


def test_map_output_is_consumable_by_generate(initialised_experiment, mock_no_dotenv):
    """A mapping produced by `map` drives `generate` end to end."""
    pid, exp_subdir = initialised_experiment
    (exp_subdir / "llms-for-experiment.csv").write_text(
        "provider;model_name;profile_name;notes\nollama;llama3.3:latest;ollama-default;local model\n",
        encoding="utf-8",
    )
    (exp_subdir / "llm-params.csv").write_text(_LLM_PARAMS_HEADER + _LLM_PARAMS_ROW, encoding="utf-8")

    result = runner.invoke(app, ["experiment", "map", "--pid", pid])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["experiment", "generate", "--pid", pid])
    assert result.exit_code == 0, result.output
    assert result.exception is None

    # 3 data rows x 2 prompts x 1 model x 1 profile.
    assert "Generated 6 row(s)" in " ".join(result.output.split())
    assert find_db(exp_subdir) is not None


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


def test_map_available_under_exp_alias(initialised_experiment, mock_no_dotenv):
    """The command is reachable through the hidden `exp` alias."""
    pid, exp_subdir = initialised_experiment

    result = runner.invoke(app, ["exp", "map", "--pid", pid, "--prompt", "prompt01"])

    assert result.exit_code == 0, result.output
    assert len(_read_mapping(exp_subdir)) == 3
