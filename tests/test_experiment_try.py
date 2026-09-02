"""Tests for the `experiment try` command."""

import json
import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.base.dao import ExperimentDAO
from llmexer.cli import app
from llmexer.exceptions import LLMExerException
from tests.db_helpers import find_db, read_experiment_df, read_try_rows, try_table_names

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


@pytest.fixture()
def mock_ollama(monkeypatch):
    """Replace OllamaProvider with a fake returning a canned response."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42)

    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)
    return FakeOllamaProvider


@pytest.fixture()
def mock_failing_ollama(monkeypatch):
    """Replace OllamaProvider with one whose call fails."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FailingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.STARTED

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(text="", usage_tokens=None, raw="connection refused")

    monkeypatch.setattr(llm_module, "OllamaProvider", FailingOllamaProvider)
    return FailingOllamaProvider


_LLM_PARAMS_HEADER = (
    "provider;model_name;profile_name;temperature;top_p;max_tokens;"
    "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
)
_OLLAMA_PARAMS_ROW = "ollama;llama3.3:latest;ollama-default;0.7;1.0;512;4096;1.1;;;;\n"
_MODELS_HEADER = "provider;model_name;profile_name;notes\n"
_OLLAMA_MODEL_ROW = "ollama;llama3.3:latest;ollama-default;local model\n"


def _write_params(exp_subdir, *rows):
    (exp_subdir / "llm-params.csv").write_text(_LLM_PARAMS_HEADER + "".join(rows), encoding="utf-8")


@pytest.fixture()
def generated_experiment(projects_dir):
    """Initialise a project, run `generate`, return (pid, exp_subdir, db_path)."""
    pid = "try-test"
    exp_subdir = projects_dir / pid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "llms-for-experiment.csv").write_text(_MODELS_HEADER + _OLLAMA_MODEL_ROW, encoding="utf-8")
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\n"
        "D01;Sample Paper Title One;This is the abstract of the first sample paper.\n"
        "D02;Sample Paper Title Two;This is the abstract of the second sample paper.\n",
        encoding="utf-8",
    )
    (exp_subdir / "mapping.csv").write_text("data_id;prompt_id\nD01;prompt01\n", encoding="utf-8")
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}. Abstract: {{abstract}}.", encoding="utf-8")
    (prompts_dir / "prompt02.txt").write_text("Summarise: {{title}}.", encoding="utf-8")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW)

    result = runner.invoke(app, ["experiment", "generate", "--pid", pid])
    assert result.exit_code == 0, result.output

    return pid, exp_subdir, find_db(exp_subdir)


def _try(pid, *options):
    """Invoke `experiment try` with the three names filled in by default."""
    args = ["experiment", "try", "--pid", pid]
    if "--prompt" not in options:
        args += ["--prompt", "prompt01"]
    if "--profile" not in options:
        args += ["--profile", "ollama-default"]
    if "--data-id" not in options:
        args += ["--data-id", "D01"]
    return runner.invoke(app, args + list(options))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_try_appends_to_both_try_tables(generated_experiment, mock_ollama):
    """A successful try lands in try_experiment_<provider> with its result."""
    pid, _exp_subdir, db_path = generated_experiment

    result = _try(pid)

    assert result.exit_code == 0, result.output
    tables = try_table_names(db_path)
    assert "try_experiment_ollama" in tables
    assert "try_param_ollama" in tables

    rows = read_try_rows(db_path, "ollama")
    assert len(rows) == 1
    row = rows[0]
    assert row["code"] == "D01_prompt01_llama3.3:latest_ollama-default"
    assert row["status"] == "success"
    assert row["response_text"] == "mocked response"
    assert row["usage_tokens"] == 42
    assert row["model_name"] == "llama3.3:latest"
    assert row["provider_name"] == "ollama"
    # The parameters the try ran with are joined back in.
    assert row["temperature"] == 0.7
    assert row["ollama_context_window"] == 4096


def test_try_prints_header_and_response(generated_experiment, mock_ollama):
    """The header names model, provider and usage tokens; then the response."""
    pid, _exp_subdir, _db_path = generated_experiment

    result = _try(pid)

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Model: llama3.3:latest" in output
    assert "Provider: ollama" in output
    assert "Usage tokens: 42" in output
    assert "mocked response" in output


def test_try_renders_the_selected_data_row(generated_experiment, mock_ollama):
    """The prompt is rendered with the --data-id row, not the first one."""
    pid, _exp_subdir, db_path = generated_experiment

    assert _try(pid, "--data-id", "D02").exit_code == 0

    row = read_try_rows(db_path, "ollama")[0]
    assert "Sample Paper Title Two" in row["prompt"]
    assert row["code"].startswith("D02_prompt01_")


def test_try_uses_the_selected_prompt(generated_experiment, mock_ollama):
    """--prompt selects the template, with or without the .txt extension."""
    pid, _exp_subdir, db_path = generated_experiment

    assert _try(pid, "--prompt", "prompt02.txt").exit_code == 0

    row = read_try_rows(db_path, "ollama")[0]
    assert row["prompt"].startswith("Summarise:")
    assert row["code"] == "D01_prompt02_llama3.3:latest_ollama-default"


def test_try_appends_each_run(generated_experiment, mock_ollama):
    """A second try appends a second row to both tables, keeping the first."""
    pid, _exp_subdir, db_path = generated_experiment

    assert _try(pid).exit_code == 0
    assert _try(pid, "--data-id", "D02").exit_code == 0

    rows = read_try_rows(db_path, "ollama")
    assert [row["ID"] for row in rows] == [1, 2]
    assert rows[0]["code"].startswith("D01_")
    assert rows[1]["code"].startswith("D02_")


def test_try_records_the_parameters_of_each_try(generated_experiment, mock_ollama):
    """Editing a profile between tries: each try keeps the values it ran with."""
    pid, exp_subdir, db_path = generated_experiment

    assert _try(pid).exit_code == 0
    _write_params(exp_subdir, "ollama;llama3.3:latest;ollama-default;0.1;1.0;256;4096;1.1;;;;\n")
    assert _try(pid).exit_code == 0

    rows = read_try_rows(db_path, "ollama")
    assert [row["temperature"] for row in rows] == [0.7, 0.1]
    assert [row["max_tokens"] for row in rows] == [512, 256]


def test_try_leaves_the_generated_experiment_untouched(generated_experiment, mock_ollama):
    """The generated rows, their results and `stats` are unaffected by a try."""
    pid, _exp_subdir, db_path = generated_experiment
    before = read_experiment_df(db_path)

    assert _try(pid).exit_code == 0

    assert read_experiment_df(db_path).equals(before)
    with ExperimentDAO(str(db_path)) as dao:
        assert sorted(dao.provider_tables()) == ["ollama"]
        assert dao.stats()["total"] == len(before)


def test_try_writes_the_response_json(generated_experiment, mock_ollama):
    """The per-call payload is saved under experiment/responses/, as `run` does."""
    pid, exp_subdir, _db_path = generated_experiment

    assert _try(pid).exit_code == 0

    files = list((exp_subdir / "responses").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["model"] == "llama3.3:latest"
    assert payload["provider"] == "ollama"
    assert payload["response_text"] == "mocked response"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_try_dry_run_writes_nothing(generated_experiment, mock_ollama):
    """--dry-run shows the rendered prompt and touches neither DB nor disk."""
    pid, exp_subdir, db_path = generated_experiment

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "try",
            "--pid",
            pid,
            "--prompt",
            "prompt01",
            "--profile",
            "ollama-default",
            "--data-id",
            "D01",
        ],
    )

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Dry run:" in output
    assert "Sample Paper Title One" in output
    assert "try_experiment_ollama" not in try_table_names(db_path)
    assert not (exp_subdir / "responses").exists()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_try_unknown_prompt_aborts(generated_experiment, mock_ollama):
    """An unknown prompt name aborts before anything is written."""
    pid, _exp_subdir, db_path = generated_experiment

    result = _try(pid, "--prompt", "nope")

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "nope.txt" in str(result.exception)
    assert "try_experiment_ollama" not in try_table_names(db_path)


def test_try_unknown_data_id_aborts(generated_experiment, mock_ollama):
    """An unknown --data-id aborts and names the available IDs."""
    pid, _exp_subdir, _db_path = generated_experiment

    result = _try(pid, "--data-id", "D99")

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    message = str(result.exception)
    assert "D99" in message and "D01" in message


def test_try_unknown_profile_aborts(generated_experiment, mock_ollama):
    """An unknown profile aborts and names the profiles that do exist."""
    pid, _exp_subdir, _db_path = generated_experiment

    result = _try(pid, "--profile", "nope")

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "ollama-default" in str(result.exception)


def test_try_profile_match_is_case_sensitive(generated_experiment, mock_ollama):
    """Profiles are matched in full and case-sensitively."""
    pid, _exp_subdir, _db_path = generated_experiment

    result = _try(pid, "--profile", "OLLAMA-DEFAULT")

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_try_ambiguous_profile_aborts_with_candidates(generated_experiment, mock_ollama):
    """A profile name covering two models is reported, not guessed at."""
    pid, exp_subdir, _db_path = generated_experiment
    _write_params(
        exp_subdir,
        _OLLAMA_PARAMS_ROW,
        "ollama;phi4:14b;ollama-default;0.2;0.9;256;8192;1.0;;;;\n",
    )

    result = _try(pid)

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    output = " ".join(result.output.split())
    assert "ollama / llama3.3:latest" in output
    assert "ollama / phi4:14b" in output
    assert "--model" in str(result.exception)


def test_try_model_option_disambiguates(generated_experiment, mock_ollama):
    """--model picks one of several rows sharing a profile name."""
    pid, exp_subdir, db_path = generated_experiment
    _write_params(
        exp_subdir,
        _OLLAMA_PARAMS_ROW,
        "ollama;phi4:14b;ollama-default;0.2;0.9;256;8192;1.0;;;;\n",
    )

    result = _try(pid, "--model", "phi4:14b")

    assert result.exit_code == 0, result.output
    row = read_try_rows(db_path, "ollama")[0]
    assert row["model_name"] == "phi4:14b"
    assert row["temperature"] == 0.2


def test_try_without_database_aborts(projects_dir, mock_no_dotenv):
    """A project with no generated database points the user at `generate`."""
    pid = "no-db"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir / "prompts")
    (exp_subdir / "llms-for-experiment.csv").write_text(_MODELS_HEADER + _OLLAMA_MODEL_ROW, encoding="utf-8")
    (exp_subdir / "data.csv").write_text("ID;Title;Abstract\nD01;T;A.\n", encoding="utf-8")
    (exp_subdir / "mapping.csv").write_text("data_id;prompt_id\nD01;prompt01\n", encoding="utf-8")
    (exp_subdir / "prompts" / "prompt01.txt").write_text("Title: {{title}}.", encoding="utf-8")
    _write_params(exp_subdir, _OLLAMA_PARAMS_ROW)

    result = _try(pid)

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "experiment generate" in str(result.exception)


# ---------------------------------------------------------------------------
# Failed calls
# ---------------------------------------------------------------------------


def test_try_records_a_failed_call(generated_experiment, mock_failing_ollama):
    """A failed provider call is printed as an error and still appended."""
    pid, _exp_subdir, db_path = generated_experiment

    result = _try(pid)

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "connection refused" in output
    assert "mocked response" not in output

    rows = read_try_rows(db_path, "ollama")
    assert len(rows) == 1
    assert rows[0]["status"].startswith("Error:")
    assert rows[0]["response_text"] == ""


# ---------------------------------------------------------------------------
# DAO level
# ---------------------------------------------------------------------------


def test_dao_try_tables_are_ignored_by_the_experiment_api(generated_experiment, mock_ollama):
    """try_* tables never leak into provider_tables()/fetch_rows()/stats()."""
    pid, _exp_subdir, db_path = generated_experiment
    assert _try(pid).exit_code == 0

    with ExperimentDAO(str(db_path)) as dao:
        assert sorted(dao.provider_tables()) == ["ollama"]
        assert sorted(dao.params_tables()) == ["ollama"]
        assert sorted(dao.try_tables()) == ["ollama"]
        assert sorted(dao.try_params_tables()) == ["ollama"]
        assert all("try" not in row["code"] for row in dao.fetch_rows())
        # The generated database holds one mapping row x one model; the try that
        # just ran must not show up in the totals.
        assert dao.stats()["total"] == 1


def test_dao_append_try_row_assigns_increasing_ids(tmp_path):
    """append_try_row lets SQLite number the tries and returns the new ID."""
    db_path = tmp_path / "experiment_20240101_01.db"
    row = {
        "code": "D01_prompt01_llama3.3:latest_ollama-default",
        "prompt": "Hello",
        "model_name": "llama3.3:latest",
        "provider_name": "ollama",
        "profile_name": "ollama-default",
        "temperature": 0.7,
        "ollama_context_window": 4096,
        "status": "success",
    }

    with ExperimentDAO(str(db_path), create=True) as dao:
        assert dao.append_try_row("ollama", row) == 1
        assert dao.append_try_row("ollama", row) == 2
        rows = dao.fetch_try_rows("ollama")

    assert [r["ID"] for r in rows] == [1, 2]
    assert rows[0]["params_code"] == "llama3.3:latest_ollama"
    assert rows[0]["temperature"] == 0.7
