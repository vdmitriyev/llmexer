"""Tests for the `experiment update` command."""

import os
import sqlite3
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.base.dao import ExperimentDAO
from llmexer.cli import app
from llmexer.exceptions import LLMExerException
from tests.db_helpers import find_db, list_dbs, read_experiment_df, read_params_rows

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
_OLLAMA_PARAMS_ROW = "ollama;llama3.3:latest;ollama-default;0.7;1.0;512;4096;1.1;;;;\n"
_MODELS_HEADER = "provider;model_name;profile_name;notes\n"
_OLLAMA_MODEL_ROW = "ollama;llama3.3:latest;ollama-default;local model\n"


def _write_models(exp_subdir, *rows):
    (exp_subdir / "llms-for-experiment.csv").write_text(_MODELS_HEADER + "".join(rows), encoding="utf-8")


def _write_params(exp_subdir, *rows):
    (exp_subdir / "llm-params.csv").write_text(_LLM_PARAMS_HEADER + "".join(rows), encoding="utf-8")


def _write_mapping(exp_subdir, *pairs):
    body = "".join(f"{data_id};{prompt_id}\n" for data_id, prompt_id in pairs)
    (exp_subdir / "mapping.csv").write_text("data_id;prompt_id\n" + body, encoding="utf-8")


@pytest.fixture()
def generated_experiment(projects_dir):
    """Initialise a project, run `generate`, return (pid, exp_subdir, db_path).

    The generated database holds two ollama rows (D01 and D02 under prompt01).
    """
    pid = "update-test"
    exp_subdir = projects_dir / pid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW)
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\n"
        "D01;Sample Paper Title One;This is the abstract of the first sample paper.\n"
        "D02;Sample Paper Title Two;This is the abstract of the second sample paper.\n",
        encoding="utf-8",
    )
    _write_mapping(exp_subdir, ("D01", "prompt01"), ("D02", "prompt01"))
    (prompts_dir / "prompt01.txt").write_text(
        "Here is the title: {{title}}.\n\nHere is the abstract: {{abstract}}.",
        encoding="utf-8",
    )
    (prompts_dir / "prompt02.txt").write_text("Summarise: {{title}}.", encoding="utf-8")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW)

    result = runner.invoke(app, ["experiment", "generate", "--pid", pid])
    assert result.exit_code == 0, result.output

    return pid, exp_subdir, find_db(exp_subdir)


# ---------------------------------------------------------------------------
# Adding new combinations
# ---------------------------------------------------------------------------


def test_update_adds_rows_for_a_new_model(generated_experiment):
    """A model added to llms-for-experiment.csv brings in its combinations only."""
    pid, exp_subdir, db_path = generated_experiment
    before = read_experiment_df(db_path)

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "ollama;phi4:14b;ollama-fast;second model\n")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW, "ollama;phi4:14b;ollama-fast;0.2;0.9;256;8192;1.0;;;;\n")

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    after = read_experiment_df(db_path)
    assert len(after) == len(before) + 2
    added = after[after["model_name"] == "phi4:14b"]
    assert sorted(added["code"]) == ["D01_prompt01_phi4:14b_ollama-fast", "D02_prompt01_phi4:14b_ollama-fast"]
    # The parameter set of the new profile came from llm-params.csv.
    profiles = {row["profile_name"]: row for row in read_params_rows(db_path, "ollama")}
    assert profiles["ollama-fast"]["temperature"] == 0.2
    assert profiles["ollama-fast"]["ollama_context_window"] == 8192
    # Untouched rows keep their identity.
    assert list(before["ID"]) == list(after["ID"])[: len(before)]
    assert list(before["code"]) == list(after["code"])[: len(before)]


def test_update_creates_tables_for_a_new_provider(generated_experiment):
    """A model of a provider absent from the database gets its own table pair."""
    pid, exp_subdir, db_path = generated_experiment

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "openai;gpt-4o;openai-default;hosted model\n")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW, "openai;gpt-4o;openai-default;0.5;1.0;256;;;;;42;\n")

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    with ExperimentDAO(str(db_path)) as dao:
        assert sorted(dao.provider_tables()) == ["ollama", "openai"]
        assert sorted(dao.params_tables()) == ["ollama", "openai"]
    assert read_params_rows(db_path, "openai")[0]["openai_seed"] == 42
    assert len(read_experiment_df(db_path)) == 4


def test_update_adds_rows_for_new_mapping_rows(generated_experiment):
    """Extra mapping.csv rows bring in the matching combinations."""
    pid, exp_subdir, db_path = generated_experiment

    _write_mapping(
        exp_subdir,
        ("D01", "prompt01"),
        ("D02", "prompt01"),
        ("D01", "prompt02"),
    )

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    codes = set(read_experiment_df(db_path)["code"])
    assert "D01_prompt02_llama3.3:latest_ollama-default" in codes
    assert len(codes) == 3


def test_update_adds_a_renamed_profile_without_drift(generated_experiment):
    """Changed parameters published under a NEW profile name are added normally."""
    pid, exp_subdir, db_path = generated_experiment

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "ollama;llama3.3:latest;ollama-default-v2;retuned\n")
    _write_params(
        exp_subdir,
        _OLLAMA_PARAMS_ROW,
        "ollama;llama3.3:latest;ollama-default-v2;0.9;1.0;512;4096;1.1;;;;\n",
    )

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    profiles = {row["profile_name"]: row for row in read_params_rows(db_path, "ollama")}
    assert sorted(profiles) == ["ollama-default", "ollama-default-v2"]
    assert profiles["ollama-default"]["temperature"] == 0.7
    assert profiles["ollama-default-v2"]["temperature"] == 0.9
    assert len(read_experiment_df(db_path)) == 4


def test_update_appends_ids_and_preserves_results(generated_experiment):
    """New IDs continue after the highest one and stored results survive."""
    pid, exp_subdir, db_path = generated_experiment

    with ExperimentDAO(str(db_path)) as dao:
        dao.update_result("ollama", 1, {"status": "success", "response_text": "done", "total_tokens": 11})

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "ollama;phi4:14b;ollama-fast;second model\n")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW, "ollama;phi4:14b;ollama-fast;0.2;0.9;256;8192;1.0;;;;\n")

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    by_id = read_experiment_df(db_path).set_index("ID")
    assert list(by_id.index) == [1, 2, 3, 4]
    assert by_id.loc[1, "status"] == "success"
    assert by_id.loc[1, "response_text"] == "done"
    assert by_id.loc[3, "model_name"] == "phi4:14b"


def test_update_without_changes_adds_nothing(generated_experiment):
    """Running update on unchanged CSVs reports it and leaves the database alone."""
    pid, _exp_subdir, db_path = generated_experiment
    before = read_experiment_df(db_path)

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert "already up to date" in " ".join(result.output.split())
    assert read_experiment_df(db_path).equals(before)


# ---------------------------------------------------------------------------
# Parameter drift
# ---------------------------------------------------------------------------


def test_update_aborts_on_parameter_drift(generated_experiment):
    """Changed parameters under an UNCHANGED profile name abort the whole update."""
    pid, exp_subdir, db_path = generated_experiment
    before = read_experiment_df(db_path)

    # Same profile name, different temperature — plus a genuinely new model that
    # must NOT be added because the update aborts as a whole.
    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "ollama;phi4:14b;ollama-fast;second model\n")
    _write_params(
        exp_subdir,
        "ollama;llama3.3:latest;ollama-default;0.9;1.0;512;4096;1.1;;;;\n",
        "ollama;phi4:14b;ollama-fast;0.2;0.9;256;8192;1.0;;;;\n",
    )

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    output = " ".join(result.output.split())
    assert "parameter drift for ollama / llama3.3:latest / ollama-default" in output
    assert "temperature: db=0.7 csv=0.9" in output
    assert "profile_name" in str(result.exception)
    # Nothing was written: no new rows, no new profile, old values kept.
    assert read_experiment_df(db_path).equals(before)
    profiles = {row["profile_name"]: row for row in read_params_rows(db_path, "ollama")}
    assert sorted(profiles) == ["ollama-default"]
    assert profiles["ollama-default"]["temperature"] == 0.7


def test_update_ignores_equivalent_parameter_formatting(generated_experiment):
    """A parameter rewritten as 512.0 instead of 512 is not drift."""
    pid, exp_subdir, db_path = generated_experiment

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "ollama;phi4:14b;ollama-fast;second model\n")
    _write_params(
        exp_subdir,
        "ollama;llama3.3:latest;ollama-default;0.70;1.0;512.0;4096.0;1.1;;;;\n",
        "ollama;phi4:14b;ollama-fast;0.2;0.9;256;8192;1.0;;;;\n",
    )

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert len(read_experiment_df(db_path)) == 4


# ---------------------------------------------------------------------------
# Reporting, dry run and database selection
# ---------------------------------------------------------------------------


def test_update_reports_changed_prompt_text(generated_experiment):
    """An edited prompt template is reported; stored rows are not rewritten."""
    pid, exp_subdir, db_path = generated_experiment
    before = read_experiment_df(db_path)

    (exp_subdir / "prompts" / "prompt01.txt").write_text("Rewritten: {{title}}.", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "2 stored row(s) no longer match" in output
    assert read_experiment_df(db_path).equals(before)


def test_update_dry_run_writes_nothing(generated_experiment):
    """--dry-run reports the row count without touching the database."""
    pid, exp_subdir, db_path = generated_experiment
    before = read_experiment_df(db_path)

    _write_models(exp_subdir, _OLLAMA_MODEL_ROW, "ollama;phi4:14b;ollama-fast;second model\n")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW, "ollama;phi4:14b;ollama-fast;0.2;0.9;256;8192;1.0;;;;\n")

    result = runner.invoke(app, ["--dry-run", "experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Dry run:" in output
    assert "would add 2 row(s)" in output
    assert read_experiment_df(db_path).equals(before)


def test_update_file_option_targets_one_database(generated_experiment):
    """--file updates the chosen database; the newest one stays untouched."""
    pid, exp_subdir, first_db = generated_experiment

    assert runner.invoke(app, ["experiment", "generate", "--pid", pid]).exit_code == 0
    assert len(list_dbs(exp_subdir)) == 2
    newest_db = exp_subdir / list_dbs(exp_subdir)[-1]
    newest_before = read_experiment_df(newest_db)

    _write_mapping(exp_subdir, ("D01", "prompt01"), ("D02", "prompt01"), ("D01", "prompt02"))

    result = runner.invoke(app, ["experiment", "update", "--pid", pid, "--file", os.path.basename(first_db)])

    assert result.exit_code == 0, result.output
    assert len(read_experiment_df(first_db)) == 3
    assert read_experiment_df(newest_db).equals(newest_before)


def test_update_defaults_to_newest_database(generated_experiment):
    """Without --file the newest database is updated."""
    pid, exp_subdir, first_db = generated_experiment

    assert runner.invoke(app, ["experiment", "generate", "--pid", pid]).exit_code == 0
    newest_db = exp_subdir / list_dbs(exp_subdir)[-1]

    _write_mapping(exp_subdir, ("D01", "prompt01"), ("D02", "prompt01"), ("D01", "prompt02"))

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert len(read_experiment_df(newest_db)) == 3
    assert len(read_experiment_df(first_db)) == 2


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_update_without_database_raises(projects_dir, mock_no_dotenv):
    """A project with no generated database points the user at `generate`."""
    pid = "no-db"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir / "prompts")
    _write_models(exp_subdir, _OLLAMA_MODEL_ROW)
    (exp_subdir / "data.csv").write_text("ID;Title;Abstract\nD01;T;A.\n", encoding="utf-8")
    _write_mapping(exp_subdir, ("D01", "prompt01"))
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW)

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "experiment generate" in str(result.exception)


def test_update_rejects_legacy_database(generated_experiment):
    """A database predating the params table is rejected, as it is for `run`."""
    pid, exp_subdir, _db_path = generated_experiment
    legacy_db = exp_subdir / "experiment_20200101_09.db"
    connection = sqlite3.connect(legacy_db)
    connection.execute("CREATE TABLE experiment_ollama (ID INTEGER PRIMARY KEY, code TEXT)")
    connection.commit()
    connection.close()

    result = runner.invoke(app, ["experiment", "update", "--pid", pid, "--file", legacy_db.name])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "params_ollama" in str(result.exception)


def test_update_missing_csv_raises(generated_experiment):
    """A missing input CSV is reported before the database is opened."""
    pid, exp_subdir, _db_path = generated_experiment
    (exp_subdir / "mapping.csv").unlink()

    result = runner.invoke(app, ["experiment", "update", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "mapping.csv" in str(result.exception)


def test_update_alias_group(generated_experiment):
    """The hidden `exp` alias exposes the command too."""
    pid, exp_subdir, db_path = generated_experiment

    _write_mapping(exp_subdir, ("D01", "prompt01"), ("D02", "prompt01"), ("D02", "prompt02"))

    result = runner.invoke(app, ["exp", "update", "--pid", pid])

    assert result.exit_code == 0, result.output
    assert len(read_experiment_df(db_path)) == 3
