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

runner = CliRunner()

# The experiment CSV contains all 21 columns (prompt + tokens_estimate + embedded param columns).
_EXPERIMENT_CSV_HEADER = (
    "ID;code;prompt;tokens_estimate;original_data;model_name;provider_name;prompt_hash;original_data_hash;"
    "profile_name;param_model_name;param_provider;temperature;top_p;max_tokens;"
    "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
)
_EXPERIMENT_CSV_ROW = (
    "1;D01_prompt01_llama3.3:latest_ollama-default;Hello world;2;"
    '{"ID":"D01"};llama3.3:latest;ollama;abc123;def456;'
    "ollama-default;llama3.3:latest;ollama;0.7;1.0;512;4096;1.1;;;;\n"
)
_EXPERIMENT_CSV_ROW_OPENAI = (
    "2;D01_prompt01_gpt-4o_openai-default;Hello world;2;"
    '{"ID":"D01"};gpt-4o;openai;abc123;def456;'
    "openai-default;gpt-4o;openai;0.7;1.0;512;;;;42;\n"
)

# Results are named after the generated input file passed to `run`.
_RESULTS_CSV_NAME = "experiment_20240101-abcd1234_results.csv"


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
def experiment_with_csvs(projects_dir):
    """Create a test experiment with a pre-built 20-column experiment CSV."""
    pid = "run-test-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)

    (exp_subdir / "experiment_20240101-abcd1234.csv").write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW,
        encoding="utf-8",
    )
    return pid, exp_subdir


@pytest.fixture()
def mock_llm_mapper(monkeypatch):
    """Replace LLMRequestsMapper and OllamaProvider with fakes that return canned results."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FakeResult:
        response_text = "mocked response"
        usage_tokens = 42
        status = "success"
        timestamp = "2024-01-01T00:00:00"

        def model_dump(self):
            return {
                "response_text": self.response_text,
                "usage_tokens": self.usage_tokens,
                "status": self.status,
                "timestamp": self.timestamp,
            }

    class FakeMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            self.provider = provider

        def execute(self, prompt, row):
            return FakeResult()

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.SUCCESS

        def execute(self, prompt, row):
            self.state = CallerState.SUCCESS
            return ProviderResponse(text="mocked response", usage_tokens=42)

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", FakeMapper)
    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)
    return FakeMapper


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


def test_run_dry_run_no_files_written(experiment_with_csvs, mock_llm_mapper):
    """With --dry-run, no results CSV or responses/ dir should be created."""
    pid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    # No results CSV written
    assert not any(f == _RESULTS_CSV_NAME for f in os.listdir(exp_subdir))
    # No responses directory created
    assert not (exp_subdir / "responses").exists()


def test_run_dry_run_shows_row_count(experiment_with_csvs, mock_llm_mapper):
    """Dry run output should mention the total number of rows to run."""
    pid, _ = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    assert "1" in result.output  # 1 row in the experiment CSV


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_run_creates_results_csv(experiment_with_csvs, mock_llm_mapper):
    """run should create a results CSV after execution."""
    pid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    assert len(results_files) == 1


def test_run_results_csv_has_correct_columns(experiment_with_csvs, mock_llm_mapper):
    """Results CSV should contain all experiment CSV columns plus 4 result fields."""
    pid, exp_subdir = experiment_with_csvs

    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")

    # Columns from experiment CSV (prompt + embedded params)
    for col in [
        "ID",
        "code",
        "prompt",
        "model_name",
        "provider_name",
        "profile_name",
        "param_model_name",
        "param_provider",
    ]:
        assert col in df.columns

    # Result columns appended by run
    for col in ["response_text", "usage_tokens", "status", "timestamp"]:
        assert col in df.columns


def test_run_results_csv_row_count(experiment_with_csvs, mock_llm_mapper):
    """Result row count equals the number of rows in the experiment CSV."""
    pid, exp_subdir = experiment_with_csvs

    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1


def test_run_creates_responses_directory(experiment_with_csvs, mock_llm_mapper):
    """run should create an experiment/responses/ directory."""
    pid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    assert (exp_subdir / "responses").is_dir()


def test_run_creates_individual_json_files(experiment_with_csvs, mock_llm_mapper):
    """run should save one JSON file per LLM call in responses/."""
    pid, exp_subdir = experiment_with_csvs

    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    json_files = list((exp_subdir / "responses").glob("*.json"))
    assert len(json_files) == 1
    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)
    assert "response_text" in data


def test_run_uses_current_experiment_from_env(
    projects_dir, mock_no_dotenv, mock_llm_mapper, monkeypatch
):
    """When --pid is omitted, run should use PROJECT_ID from the environment."""
    pid = "env-run-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    (exp_subdir / "experiment_20240101-abcd1234.csv").write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW, encoding="utf-8"
    )
    monkeypatch.setenv("PROJECT_ID", pid)

    result = runner.invoke(
        app, ["experiment", "run", "--file", "experiment_20240101-abcd1234.csv"]
    )

    assert result.exit_code == 0


def test_run_custom_experiment_csv(experiment_with_csvs, mock_llm_mapper, tmp_path):
    """--file should override the auto-detected CSV."""
    pid, exp_subdir = experiment_with_csvs
    custom_csv = tmp_path / "custom_experiment.csv"
    custom_csv.write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW, encoding="utf-8"
    )

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", str(custom_csv)],
    )

    assert result.exit_code == 0


def test_run_failed_call_still_writes_row(experiment_with_csvs, monkeypatch):
    """A failing LLM call should produce an Error status row; run should not abort."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class ErrorOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.ERROR

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(
                text="", usage_tokens=None, raw="connection refused"
            )

    monkeypatch.setattr(llm_module, "OllamaProvider", ErrorOllamaProvider)

    pid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1
    assert "Error" in str(df["status"].iloc[0])


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


def test_run_no_experiment_csv_raises(projects_dir):
    """run when experiment/ has no experiment_*.csv should raise LLMExerException."""
    pid = "no-csv"
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


def test_run_missing_openai_package_raises(experiment_with_csvs, monkeypatch):
    """When openai is not installed, run should raise LLMExerException with install hint."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "llmexer.base.llm_provider":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    pid, _ = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "pip install" in str(result.exception).lower()


# ---------------------------------------------------------------------------
# Per-provider env var tests
# ---------------------------------------------------------------------------


def test_run_uses_provider_url_from_env(experiment_with_csvs, monkeypatch):
    """PROVIDER_OLLAMA_URL in env should override the built-in URL_MAP default."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["base_url"] = base_url

        def execute(self, prompt, row):
            self.state = CallerState.SUCCESS
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.setenv("PROVIDER_OLLAMA_URL", "http://custom-ollama:9999/v1")

    pid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("base_url") == "http://custom-ollama:9999/v1"


def test_run_provider_url_falls_back_to_url_map(experiment_with_csvs, monkeypatch):
    """When PROVIDER_OLLAMA_URL is not set, the built-in URL_MAP default is used."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["base_url"] = base_url

        def execute(self, prompt, row):
            self.state = CallerState.SUCCESS
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.delenv("PROVIDER_OLLAMA_URL", raising=False)

    pid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("base_url") == "http://localhost:11434/v1"


def test_run_uses_provider_key_from_env(experiment_with_csvs, monkeypatch):
    """PROVIDER_OLLAMA_KEY in env should take precedence over LLM_API_KEY."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["api_key"] = auth.api_key if auth else "na"

        def execute(self, prompt, row):
            self.state = CallerState.SUCCESS
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.setenv("PROVIDER_OLLAMA_KEY", "provider-specific-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")

    pid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("api_key") == "provider-specific-key"


def test_run_provider_key_defaults_to_na_when_absent(experiment_with_csvs, monkeypatch):
    """When PROVIDER_OLLAMA_KEY is absent, api_key defaults to 'na'."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    captured = {}

    class CapturingOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            captured["api_key"] = auth.api_key if auth else "na"

        def execute(self, prompt, row):
            self.state = CallerState.SUCCESS
            return ProviderResponse(text="x", usage_tokens=1)

    monkeypatch.setattr(llm_module, "OllamaProvider", CapturingOllamaProvider)
    monkeypatch.delenv("PROVIDER_OLLAMA_KEY", raising=False)

    pid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("api_key") == "na"


# ---------------------------------------------------------------------------
# --filter-provider tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiment_with_two_provider_rows(projects_dir, mock_llm_mapper):
    """Experiment CSV with one ollama row and one openai row."""
    pid = "filter-test-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    (exp_subdir / "experiment_20240101-abcd1234.csv").write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW + _EXPERIMENT_CSV_ROW_OPENAI,
        encoding="utf-8",
    )
    return pid, exp_subdir


def test_run_filter_provider_runs_only_matching_rows(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """--filter-provider ollama runs only ollama rows but persists the full set."""
    pid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
            "--filter-provider",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    # The single results file keeps every row; only the ollama row was run.
    assert len(df) == 2
    ollama_row = df[df["param_provider"].str.lower() == "ollama"].iloc[0]
    openai_row = df[df["param_provider"].str.lower() == "openai"].iloc[0]
    assert ollama_row["status"] == "success"
    assert pd.isna(openai_row["status"])


def test_run_filter_provider_no_match_exits_cleanly(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """--filter-provider gemini on a CSV with no gemini rows exits 0, writes no results file."""
    pid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
            "--filter-provider",
            "gemini",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    assert len(results_files) == 0
    assert "nothing to run" in result.output.lower()


def test_run_filter_provider_case_insensitive(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """--filter-provider OLLAMA (upper-case) should match rows with param_provider=ollama."""
    pid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--pid",
            pid,
            "--file",
            "experiment_20240101-abcd1234.csv",
            "--filter-provider",
            "OLLAMA",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 2
    ollama_row = df[df["param_provider"].str.lower() == "ollama"].iloc[0]
    assert ollama_row["status"] == "success"


def test_run_filter_provider_dry_run_shows_filtered_count(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """Dry-run with --filter-provider should print the filtered row count (1), not the total (2)."""
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
            "experiment_20240101-abcd1234.csv",
            "--filter-provider",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    # Should mention 1 (filtered count), not 2 (total)
    assert "1" in result.output


def test_run_sequential_filtered_runs_merge_into_one_file(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """Two filtered runs leave a single results file with both providers' rows filled."""
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
                "experiment_20240101-abcd1234.csv",
                "--filter-provider",
                provider,
            ],
        )
        assert result.exit_code == 0

    # Exactly one results file exists after both runs.
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f == _RESULTS_CSV_NAME and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 2
    # Both rows are now populated (ollama from run 1, openai from run 2).
    assert (
        df[df["param_provider"].str.lower() == "ollama"].iloc[0]["status"] == "success"
    )
    assert (
        df[df["param_provider"].str.lower() == "openai"].iloc[0]["status"] == "success"
    )
