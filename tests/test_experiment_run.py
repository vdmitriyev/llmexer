"""Tests for the `experiment run` command."""

import json
import os
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    LLMExerException,
    ProjectIDRequiredException,
    ProjectNotExistsException,
)
from tests.db_helpers import (
    OLLAMA_ROW,
    OPENAI_ROW,
    find_db,
    read_experiment_df,
    seed_db,
)

runner = CliRunner()

# Name of the generated experiment database used throughout these tests.
_EXPERIMENT_DB_NAME = "experiment_20240101_01.db"


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
def experiment_with_db(projects_dir):
    """Create a test experiment with a generated single-ollama-row database."""
    pid = "run-test-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / _EXPERIMENT_DB_NAME, {"ollama": [dict(OLLAMA_ROW)]})
    return pid, exp_subdir


@pytest.fixture()
def mock_providers(monkeypatch):
    """Replace the provider classes with fakes that return canned results."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FakeCompletion:
        """Stands in for an SDK response: serialize_response() calls model_dump."""

        def model_dump(self, mode=None):
            return {"id": "cmpl-1", "usage": {"prompt_tokens": 10, "total_tokens": 42}}

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42)

    class FakeOpenAIProvider(FakeOllamaProvider):
        """Same canned behaviour, but carries a raw response worth serialising."""

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42, raw=FakeCompletion())

    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)
    monkeypatch.setattr(llm_module, "OpenAIProvider", FakeOpenAIProvider)
    return FakeOpenAIProvider


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


def test_run_dry_run_no_files_written(experiment_with_db, mock_providers):
    """With --dry-run, no responses/ dir should be created and rows stay pending."""
    pid, exp_subdir = experiment_with_db

    result = runner.invoke(
        app,
        ["--dry-run", "experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    # No responses directory created
    assert not (exp_subdir / "responses").exists()
    # Row remains unrun (no status).
    df = read_experiment_df(find_db(exp_subdir))
    assert pd.isna(df.iloc[0]["status"])


def test_run_dry_run_shows_row_count(experiment_with_db, mock_providers):
    """Dry run output should mention the total number of rows to run."""
    pid, _ = experiment_with_db

    result = runner.invoke(
        app,
        ["--dry-run", "experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    assert "1" in result.output  # 1 row in the experiment database


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_run_writes_results_into_db(experiment_with_db, mock_providers):
    """run should write results back into the same database row."""
    pid, exp_subdir = experiment_with_db

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    df = read_experiment_df(find_db(exp_subdir))
    assert df.iloc[0]["status"] == "success"
    assert df.iloc[0]["response_text"] == "mocked response"


def test_run_result_row_has_result_columns(experiment_with_db, mock_providers):
    """The run row should carry the result columns, including response_json."""
    pid, exp_subdir = experiment_with_db

    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    df = read_experiment_df(find_db(exp_subdir))
    for col in [
        "response_text",
        "usage_tokens",
        "status",
        "timestamp",
        "response_json",
    ]:
        assert col in df.columns
    # response_json holds a parseable JSON payload of the call.
    payload = json.loads(df.iloc[0]["response_json"])
    assert payload["response_text"] == "mocked response"
    assert payload["provider"] == "ollama"


def test_run_results_row_count(experiment_with_db, mock_providers):
    """Result row count equals the number of rows in the experiment database."""
    pid, exp_subdir = experiment_with_db

    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    df = read_experiment_df(find_db(exp_subdir))
    assert len(df) == 1


def test_run_creates_responses_directory(experiment_with_db, mock_providers):
    """run should create an experiment/responses/ directory."""
    pid, exp_subdir = experiment_with_db

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    assert (exp_subdir / "responses").is_dir()


def test_run_creates_individual_json_files(experiment_with_db, mock_providers):
    """run should save one JSON file per LLM call in responses/."""
    pid, exp_subdir = experiment_with_db

    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    json_files = list((exp_subdir / "responses").glob("*.json"))
    assert len(json_files) == 1
    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)
    assert "response_text" in data


def test_run_uses_current_experiment_from_env(projects_dir, mock_no_dotenv, mock_providers, monkeypatch):
    """When --pid is omitted, run should use PROJECT_ID from the environment."""
    pid = "env-run-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / _EXPERIMENT_DB_NAME, {"ollama": [dict(OLLAMA_ROW)]})
    monkeypatch.setenv("PROJECT_ID", pid)

    result = runner.invoke(app, ["experiment", "run", "--file", _EXPERIMENT_DB_NAME])

    assert result.exit_code == 0


def test_run_custom_experiment_db(experiment_with_db, mock_providers, tmp_path):
    """--file should override the auto-detected database."""
    pid, _ = experiment_with_db
    custom_db = tmp_path / "custom_experiment.db"
    seed_db(custom_db, {"ollama": [dict(OLLAMA_ROW)]})

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", str(custom_db)],
    )

    assert result.exit_code == 0
    df = read_experiment_df(custom_db)
    assert df.iloc[0]["status"] == "success"


def test_run_defaults_to_newest_db(experiment_with_db, mock_providers):
    """With no --file, run uses the newest experiment database."""
    pid, exp_subdir = experiment_with_db
    # A second generation with a higher counter.
    seed_db(exp_subdir / "experiment_20240101_02.db", {"ollama": [dict(OLLAMA_ROW)]})

    result = runner.invoke(app, ["experiment", "run", "--pid", pid])

    assert result.exit_code == 0
    # The newest DB (counter 02) was the one run.
    newest = read_experiment_df(exp_subdir / "experiment_20240101_02.db")
    assert newest.iloc[0]["status"] == "success"
    oldest = read_experiment_df(exp_subdir / _EXPERIMENT_DB_NAME)
    assert pd.isna(oldest.iloc[0]["status"])


def test_run_failed_call_still_writes_row(experiment_with_db, monkeypatch):
    """A failing LLM call should produce an Error status row; run should not abort."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class ErrorOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.ERROR

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(text="", usage_tokens=None, raw="connection refused")

    monkeypatch.setattr(llm_module, "OllamaProvider", ErrorOllamaProvider)

    pid, exp_subdir = experiment_with_db

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    df = read_experiment_df(find_db(exp_subdir))
    assert len(df) == 1
    assert "Error" in str(df["status"].iloc[0])


# ---------------------------------------------------------------------------
# Skip already-successful rows
# ---------------------------------------------------------------------------


def test_run_skips_already_successful_row(projects_dir, mock_providers):
    """A row already in 'success' state is skipped (no LLM call, result untouched)."""
    pid = "skip-success-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _EXPERIMENT_DB_NAME,
        {"ollama": [{**OLLAMA_ROW, "status": "success", "response_text": "prior result"}]},
    )

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    assert "skipped" in result.output.lower()
    # The existing result is preserved; the mock's "mocked response" was not written.
    df = read_experiment_df(find_db(exp_subdir))
    assert df.iloc[0]["status"] == "success"
    assert df.iloc[0]["response_text"] == "prior result"
    # Nothing ran, so no per-call JSON file was created.
    assert not list((exp_subdir / "responses").glob("*.json"))


def test_run_skips_success_runs_only_pending(projects_dir, mock_providers):
    """With one success and one pending row, only the pending row is executed."""
    pid = "skip-mixed-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _EXPERIMENT_DB_NAME,
        {
            "ollama": [{**OLLAMA_ROW, "status": "success", "response_text": "prior"}],
            "openai": [dict(OPENAI_ROW)],
        },
    )

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0
    assert "skipped" in result.output.lower()
    df = read_experiment_df(find_db(exp_subdir)).set_index("ID")
    # ollama row untouched, openai row freshly run.
    assert df.loc[1, "response_text"] == "prior"
    assert df.loc[2, "status"] == "success"
    assert df.loc[2, "response_text"] == "mocked response"


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


def test_run_nonexistent_experiment_raises(projects_dir):
    """run with a non-existent experiment ID should raise ProjectNotExistsException."""
    result = runner.invoke(app, ["experiment", "run", "--pid", "does-not-exist"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotExistsException)


def test_run_uninitialised_experiment_raises(projects_dir):
    """run on an experiment without an experiment/ subdir should raise LLMExerException."""
    pid = "not-init"
    os.makedirs(projects_dir / pid)

    result = runner.invoke(app, ["experiment", "run", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "init" in str(result.exception).lower()


def test_run_no_experiment_db_raises(projects_dir):
    """run when experiment/ has no experiment_*.db should raise LLMExerException."""
    pid = "no-db"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)

    result = runner.invoke(app, ["experiment", "run", "--pid", pid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "generate" in str(result.exception).lower()


def test_run_without_eid_and_no_env_raises(projects_dir, mock_no_dotenv, monkeypatch):
    """When --pid is omitted and PROJECT_ID is not set, should raise ProjectIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "project_id", None)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "run"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectIDRequiredException)


def test_run_missing_openai_package_raises(experiment_with_db, monkeypatch):
    """When openai is not installed, run should raise LLMExerException with install hint."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "llmexer.base.llm_provider":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    pid, _ = experiment_with_db

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "pip install" in str(result.exception).lower()


# ---------------------------------------------------------------------------
# Per-provider env var tests
# ---------------------------------------------------------------------------


def test_run_passes_joined_params_to_the_provider(experiment_with_db, monkeypatch):
    """Parameters stored in params_<provider> reach the provider on the row.

    They live in a separate table now, so `run` only works if the DAO joins them
    back onto the flat row dict the provider reads by name.
    """
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            pass

        def execute(self, prompt, row):
            captured["row"] = dict(row)
            self.state = CallerState.FINISHED
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)

    pid, _ = experiment_with_db
    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    row = captured.get("row")
    assert row is not None
    assert row["temperature"] == 0.7
    assert row["top_p"] == 1.0
    assert row["max_tokens"] == 512
    assert row["ollama_context_window"] == 4096
    assert row["ollama_repeat_penalty"] == 1.1
    assert row["profile_name"] == "ollama-default"
    assert row["params_code"] == "llama3.3:latest_ollama"


def test_run_uses_provider_url_from_env(experiment_with_db, monkeypatch):
    """PROVIDER_OLLAMA_URL in env should override the built-in URL_MAP default."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["base_url"] = base_url

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.setenv("PROVIDER_OLLAMA_URL", "http://custom-ollama:9999/v1")

    pid, _ = experiment_with_db
    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert captured.get("base_url") == "http://custom-ollama:9999/v1"


def test_run_provider_url_falls_back_to_url_map(experiment_with_db, monkeypatch):
    """When PROVIDER_OLLAMA_URL is not set, the built-in URL_MAP default is used."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["base_url"] = base_url

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.delenv("PROVIDER_OLLAMA_URL", raising=False)

    pid, _ = experiment_with_db
    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert captured.get("base_url") == "http://localhost:11434/v1"


def test_run_uses_provider_key_from_env(experiment_with_db, monkeypatch):
    """PROVIDER_OLLAMA_KEY in env should take precedence over LLM_API_KEY."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["api_key"] = auth.api_key if auth else "na"

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.setenv("PROVIDER_OLLAMA_KEY", "provider-specific-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")

    pid, _ = experiment_with_db
    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert captured.get("api_key") == "provider-specific-key"


def test_run_provider_key_defaults_to_na_when_absent(experiment_with_db, monkeypatch):
    """When PROVIDER_OLLAMA_KEY is absent, api_key defaults to 'na'."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["api_key"] = auth.api_key if auth else "na"

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.delenv("PROVIDER_OLLAMA_KEY", raising=False)

    pid, _ = experiment_with_db
    runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert captured.get("api_key") == "na"


# ---------------------------------------------------------------------------
# --filter-provider tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiment_with_two_provider_rows(projects_dir):
    """Experiment database with one ollama row and one openai row."""
    pid = "filter-test-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _EXPERIMENT_DB_NAME,
        {"ollama": [dict(OLLAMA_ROW)], "openai": [dict(OPENAI_ROW)]},
    )
    return pid, exp_subdir


def test_run_filter_provider_runs_only_matching_rows(experiment_with_two_provider_rows, mock_providers):
    """--filter-provider ollama runs only ollama rows; openai stays pending."""
    pid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            _EXPERIMENT_DB_NAME,
            "--filter-provider",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    df = read_experiment_df(find_db(exp_subdir))
    # The single database keeps every row; only the ollama row was run.
    assert len(df) == 2
    ollama_row = df[df["provider_name"].str.lower() == "ollama"].iloc[0]
    openai_row = df[df["provider_name"].str.lower() == "openai"].iloc[0]
    assert ollama_row["status"] == "success"
    assert pd.isna(openai_row["status"])


def test_run_filter_provider_no_match_exits_cleanly(experiment_with_two_provider_rows, mock_providers):
    """--filter-provider gemini on a DB with no gemini rows exits 0, runs nothing."""
    pid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            _EXPERIMENT_DB_NAME,
            "--filter-provider",
            "gemini",
        ],
    )

    assert result.exit_code == 0
    assert "nothing to run" in result.output.lower()
    # Nothing was run: both rows stay pending.
    df = read_experiment_df(find_db(exp_subdir))
    assert df["status"].isna().all()


def test_run_filter_provider_case_insensitive(experiment_with_two_provider_rows, mock_providers):
    """--filter-provider OLLAMA (upper-case) should match the ollama table."""
    pid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            _EXPERIMENT_DB_NAME,
            "--filter-provider",
            "OLLAMA",
        ],
    )

    assert result.exit_code == 0
    df = read_experiment_df(find_db(exp_subdir))
    assert len(df) == 2
    ollama_row = df[df["provider_name"].str.lower() == "ollama"].iloc[0]
    assert ollama_row["status"] == "success"


def test_run_filter_provider_dry_run_shows_filtered_count(experiment_with_two_provider_rows, mock_providers):
    """Dry-run with --filter-provider should print the filtered row count (1)."""
    pid, _ = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            _EXPERIMENT_DB_NAME,
            "--filter-provider",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    # Should mention 1 (filtered count), not 2 (total)
    assert "1" in result.output


def test_run_sequential_filtered_runs_persist_into_one_db(experiment_with_two_provider_rows, mock_providers):
    """Two filtered runs leave a single database with both providers' rows filled."""
    pid, exp_subdir = experiment_with_two_provider_rows

    for provider in ("ollama", "openai"):
        result = runner.invoke(
            app,
            [
                "experiment",
                "run",
                "--pid",
                pid,
                "--file",
                _EXPERIMENT_DB_NAME,
                "--filter-provider",
                provider,
            ],
        )
        assert result.exit_code == 0

    df = read_experiment_df(find_db(exp_subdir))
    assert len(df) == 2
    # Both rows are now populated (ollama from run 1, openai from run 2).
    assert df[df["provider_name"].str.lower() == "ollama"].iloc[0]["status"] == "success"
    assert df[df["provider_name"].str.lower() == "openai"].iloc[0]["status"] == "success"


# ---------------------------------------------------------------------------
# --filter-model / --filter-profile tests
# ---------------------------------------------------------------------------


def _ollama_row(row_id, model_name, profile_name):
    """Build one ollama row for a (model, profile) pair of the cross join."""
    row = dict(OLLAMA_ROW)
    row.update(
        {
            "ID": row_id,
            "code": f"D01_prompt01_{model_name}_{profile_name}",
            "model_name": model_name,
            "profile_name": profile_name,
        }
    )
    return row


@pytest.fixture()
def experiment_with_model_profile_matrix(projects_dir):
    """Database with one ollama table holding two models × two profiles."""
    pid = "filter-matrix-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _EXPERIMENT_DB_NAME,
        {
            "ollama": [
                _ollama_row(1, "llama3.3:latest", "ollama-default"),
                _ollama_row(2, "llama3.3:latest", "ollama-creative"),
                _ollama_row(3, "phi4:14b", "ollama-default"),
                _ollama_row(4, "phi4:14b", "ollama-creative"),
            ]
        },
    )
    return pid, exp_subdir


def _run_with(pid, *options):
    """Invoke `experiment run` on the matrix database with extra options."""
    return runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME, *options],
    )


def _finished_ids(exp_subdir):
    """IDs whose status is 'success' in the matrix database."""
    df = read_experiment_df(find_db(exp_subdir))
    return sorted(df[df["status"] == "success"]["ID"])


def test_run_filter_model_runs_only_matching_rows(experiment_with_model_profile_matrix, mock_providers):
    """--filter-model runs both profiles of that model; the other model stays pending."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-model", "llama3.3:latest")

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == [1, 2]


def test_run_filter_profile_runs_only_matching_rows(experiment_with_model_profile_matrix, mock_providers):
    """--filter-profile runs that profile across every model."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-profile", "ollama-creative")

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == [2, 4]


def test_run_filter_model_and_profile_combine(experiment_with_model_profile_matrix, mock_providers):
    """--filter-model and --filter-profile combine with AND (one row)."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-model", "phi4:14b", "--filter-profile", "ollama-default")

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == [3]


def test_run_filter_model_is_case_sensitive(experiment_with_model_profile_matrix, mock_providers):
    """A model name in the wrong case matches nothing and runs nothing."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-model", "LLAMA3.3:LATEST")

    assert result.exit_code == 0, result.output
    assert "nothing to run" in result.output.lower()
    assert _finished_ids(exp_subdir) == []


def test_run_filter_model_needs_the_full_name(experiment_with_model_profile_matrix, mock_providers):
    """A prefix of a model name is not a match — the filter is a full match."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-model", "llama3.3")

    assert result.exit_code == 0, result.output
    assert "nothing to run" in result.output.lower()
    assert _finished_ids(exp_subdir) == []


def test_run_filter_profile_needs_the_full_name(experiment_with_model_profile_matrix, mock_providers):
    """A prefix of a profile name is not a match either."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-profile", "ollama")

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == []


def test_run_filter_no_match_names_the_filters(experiment_with_model_profile_matrix, mock_providers):
    """The 'nothing to run' warning names every filter that was applied."""
    pid, _exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-model", "phi4:14b", "--filter-profile", "nope")

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "model 'phi4:14b'" in output
    assert "profile 'nope'" in output


def test_run_filter_surrounding_whitespace_is_ignored(experiment_with_model_profile_matrix, mock_providers):
    """A value padded with spaces still matches the stored, stripped value."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = _run_with(pid, "--filter-profile", " ollama-default ")

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == [1, 3]


def test_run_filter_profile_dry_run_writes_nothing(experiment_with_model_profile_matrix, mock_providers):
    """--dry-run with a profile filter reports the filtered count and runs nothing."""
    pid, exp_subdir = experiment_with_model_profile_matrix

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            _EXPERIMENT_DB_NAME,
            "--filter-profile",
            "ollama-default",
        ],
    )

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Total experiments to run: 2" in output
    assert _finished_ids(exp_subdir) == []


def test_run_filter_provider_and_model_combine(experiment_with_two_provider_rows, mock_providers):
    """--filter-provider and --filter-model combine; a mismatch runs nothing."""
    pid, exp_subdir = experiment_with_two_provider_rows

    matching = _run_with(pid, "--filter-provider", "ollama", "--filter-model", "llama3.3:latest")

    assert matching.exit_code == 0, matching.output
    df = read_experiment_df(find_db(exp_subdir))
    assert df[df["provider_name"].str.lower() == "ollama"].iloc[0]["status"] == "success"
    assert pd.isna(df[df["provider_name"].str.lower() == "openai"].iloc[0]["status"])

    # The openai model does not live in the ollama table: no rows, nothing run.
    mismatched = _run_with(pid, "--filter-provider", "ollama", "--filter-model", "gpt-4o")

    assert mismatched.exit_code == 0, mismatched.output
    assert "nothing to run" in mismatched.output.lower()
