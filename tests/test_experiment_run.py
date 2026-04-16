"""Tests for the `experiment run` command."""

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

# The experiment CSV now contains all 20 columns (prompt + embedded param columns).
_EXPERIMENT_CSV_HEADER = (
    "ID;code;prompt;original_data;model_name;provider_name;prompt_hash;original_data_hash;"
    "profile_name;param_model_name;param_provider;temperature;top_p;max_tokens;"
    "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
)
_EXPERIMENT_CSV_ROW = (
    "1;D01_prompt01_llama3.3:latest_ollama-default;Hello world;"
    '{"ID":"D01"};llama3.3:latest;ollama;abc123;def456;'
    "ollama-default;llama3.3:latest;ollama;0.7;1.0;512;4096;1.1;;;;\n"
)
_EXPERIMENT_CSV_ROW_OPENAI = (
    "2;D01_prompt01_gpt-4o_openai-default;Hello world;"
    '{"ID":"D01"};gpt-4o;openai;abc123;def456;'
    "openai-default;gpt-4o;openai;0.7;1.0;512;;;;42;\n"
)


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
def experiment_with_csvs(experiments_dir):
    """Create a test experiment with a pre-built 20-column experiment CSV."""
    eid = "run-test-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    os.makedirs(exp_subdir)

    (exp_subdir / "experiment_20240101-abcd1234.csv").write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW,
        encoding="utf-8",
    )
    return eid, exp_subdir


@pytest.fixture()
def mock_llm_mapper(monkeypatch):
    """Replace LLMRequestsMapper in the llm module with a fake that returns a canned result."""
    import llmexer.llm as llm_module

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

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", FakeMapper)
    return FakeMapper


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


def test_run_dry_run_no_files_written(experiment_with_csvs, mock_llm_mapper):
    """With --dry-run, no results CSV or responses/ dir should be created."""
    eid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    # No results CSV written
    assert not any(
        f.startswith(f"experiment_{eid}_results_") for f in os.listdir(exp_subdir)
    )
    # No responses directory created
    assert not (exp_subdir / "responses").exists()


def test_run_dry_run_shows_row_count(experiment_with_csvs, mock_llm_mapper):
    """Dry run output should mention the total number of rows to run."""
    eid, _ = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--eid",
            eid,
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
    eid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    assert len(results_files) == 1


def test_run_results_csv_has_correct_columns(experiment_with_csvs, mock_llm_mapper):
    """Results CSV should contain all experiment CSV columns plus 4 result fields."""
    eid, exp_subdir = experiment_with_csvs

    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
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
    eid, exp_subdir = experiment_with_csvs

    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1


def test_run_creates_responses_directory(experiment_with_csvs, mock_llm_mapper):
    """run should create an experiment/responses/ directory."""
    eid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    assert (exp_subdir.parent / "responses").is_dir()


def test_run_creates_individual_json_files(experiment_with_csvs, mock_llm_mapper):
    """run should save one JSON file per LLM call in responses/."""
    eid, exp_subdir = experiment_with_csvs

    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    json_files = list((exp_subdir.parent / "responses").glob("*.json"))
    assert len(json_files) == 1
    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)
    assert "response_text" in data


def test_run_uses_current_experiment_from_env(
    experiments_dir, mock_no_dotenv, mock_llm_mapper, monkeypatch
):
    """When --eid is omitted, run should use EXPERIMENT_ID from the environment."""
    eid = "env-run-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    os.makedirs(exp_subdir)
    (exp_subdir / "experiment_20240101-abcd1234.csv").write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW, encoding="utf-8"
    )
    monkeypatch.setenv("EXPERIMENT_ID", eid)

    result = runner.invoke(
        app, ["experiment", "run", "--file", "experiment_20240101-abcd1234.csv"]
    )

    assert result.exit_code == 0


def test_run_custom_experiment_csv(experiment_with_csvs, mock_llm_mapper, tmp_path):
    """--file should override the auto-detected CSV."""
    eid, exp_subdir = experiment_with_csvs
    custom_csv = tmp_path / "custom_experiment.csv"
    custom_csv.write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW, encoding="utf-8"
    )

    result = runner.invoke(
        app,
        ["experiment", "run", "--eid", eid, "--file", str(custom_csv)],
    )

    assert result.exit_code == 0


def test_run_failed_call_still_writes_row(experiment_with_csvs, monkeypatch):
    """A failing LLM call should produce an Error status row; run should not abort."""
    import llmexer.llm as llm_module

    class ErrorResult:
        response_text = ""
        usage_tokens = None
        status = "Error: connection refused"
        timestamp = "2024-01-01T00:00:00"

        def model_dump(self):
            return {
                "response_text": self.response_text,
                "usage_tokens": self.usage_tokens,
                "status": self.status,
                "timestamp": self.timestamp,
            }

    class ErrorMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            pass

        def execute(self, prompt, row):
            return ErrorResult()

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", ErrorMapper)

    eid, exp_subdir = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert result.exit_code == 0
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1
    assert "Error" in str(df["status"].iloc[0])


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


def test_run_nonexistent_experiment_raises(experiments_dir):
    """run with a non-existent experiment ID should raise ExperimentNotExistsException."""
    result = runner.invoke(app, ["experiment", "run", "--eid", "does-not-exist"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentNotExistsException)


def test_run_uninitialised_experiment_raises(experiments_dir):
    """run on an experiment without an experiment/ subdir should raise LLMExerException."""
    eid = "not-init"
    os.makedirs(experiments_dir / eid)

    result = runner.invoke(app, ["experiment", "run", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "init" in str(result.exception).lower()


def test_run_no_experiment_csv_raises(experiments_dir):
    """run when experiment/ has no experiment_*.csv should raise LLMExerException."""
    eid = "no-csv"
    exp_subdir = experiments_dir / eid / "experiment"
    os.makedirs(exp_subdir)

    result = runner.invoke(app, ["experiment", "run", "--eid", eid])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "generate" in str(result.exception).lower()


def test_run_without_eid_and_no_env_raises(
    experiments_dir, mock_no_dotenv, monkeypatch
):
    """When --eid is omitted and EXPERIMENT_ID is not set, should raise ExperimentIDRequiredException."""
    from llmexer.configs import settings

    monkeypatch.setattr(settings, "experiment_id", None)
    monkeypatch.delenv("EXPERIMENT_ID", raising=False)

    result = runner.invoke(app, ["experiment", "run"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ExperimentIDRequiredException)


def test_run_missing_openai_package_raises(experiment_with_csvs, monkeypatch):
    """When openai is not installed, run should raise LLMExerException with install hint."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "llmexer.llm":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    eid, _ = experiment_with_csvs

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
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
    import llmexer.llm as llm_module

    captured = {}

    class CapturingMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            captured["base_url"] = base_url

        def execute(self, prompt, row):
            from unittest.mock import Mock

            r = Mock()
            r.response_text = "x"
            r.usage_tokens = 1
            r.status = "success"
            r.timestamp = "2024-01-01T00:00:00"
            r.model_dump.return_value = {
                "response_text": "x",
                "usage_tokens": 1,
                "status": "success",
                "timestamp": "2024-01-01T00:00:00",
            }
            return r

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", CapturingMapper)
    monkeypatch.setenv("PROVIDER_OLLAMA_URL", "http://custom-ollama:9999/v1")

    eid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("base_url") == "http://custom-ollama:9999/v1"


def test_run_provider_url_falls_back_to_url_map(experiment_with_csvs, monkeypatch):
    """When PROVIDER_OLLAMA_URL is not set, the built-in URL_MAP default is used."""
    import llmexer.llm as llm_module

    captured = {}

    class CapturingMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            captured["base_url"] = base_url

        def execute(self, prompt, row):
            from unittest.mock import Mock

            r = Mock()
            r.response_text = "x"
            r.usage_tokens = 1
            r.status = "success"
            r.timestamp = "2024-01-01T00:00:00"
            r.model_dump.return_value = {
                "response_text": "x",
                "usage_tokens": 1,
                "status": "success",
                "timestamp": "2024-01-01T00:00:00",
            }
            return r

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", CapturingMapper)
    monkeypatch.delenv("PROVIDER_OLLAMA_URL", raising=False)

    eid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("base_url") == "http://localhost:11434/v1"


def test_run_uses_provider_key_from_env(experiment_with_csvs, monkeypatch):
    """PROVIDER_OLLAMA_KEY in env should take precedence over LLM_API_KEY."""
    import llmexer.llm as llm_module

    captured = {}

    class CapturingMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            captured["api_key"] = api_key

        def execute(self, prompt, row):
            from unittest.mock import Mock

            r = Mock()
            r.response_text = "x"
            r.usage_tokens = 1
            r.status = "success"
            r.timestamp = "2024-01-01T00:00:00"
            r.model_dump.return_value = {
                "response_text": "x",
                "usage_tokens": 1,
                "status": "success",
                "timestamp": "2024-01-01T00:00:00",
            }
            return r

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", CapturingMapper)
    monkeypatch.setenv("PROVIDER_OLLAMA_KEY", "provider-specific-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")

    eid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("api_key") == "provider-specific-key"


def test_run_provider_key_defaults_to_na_when_absent(experiment_with_csvs, monkeypatch):
    """When PROVIDER_OLLAMA_KEY is absent, api_key defaults to 'na'."""
    import llmexer.llm as llm_module

    captured = {}

    class CapturingMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            captured["api_key"] = api_key

        def execute(self, prompt, row):
            from unittest.mock import Mock

            r = Mock()
            r.response_text = "x"
            r.usage_tokens = 1
            r.status = "success"
            r.timestamp = "2024-01-01T00:00:00"
            r.model_dump.return_value = {
                "response_text": "x",
                "usage_tokens": 1,
                "status": "success",
                "timestamp": "2024-01-01T00:00:00",
            }
            return r

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", CapturingMapper)
    monkeypatch.delenv("PROVIDER_OLLAMA_KEY", raising=False)

    eid, _ = experiment_with_csvs
    runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
        ],
    )

    assert captured.get("api_key") == "na"


# ---------------------------------------------------------------------------
# --filter-provider tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiment_with_two_provider_rows(experiments_dir, mock_llm_mapper):
    """Experiment CSV with one ollama row and one openai row."""
    eid = "filter-test-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    os.makedirs(exp_subdir)
    (exp_subdir / "experiment_20240101-abcd1234.csv").write_text(
        _EXPERIMENT_CSV_HEADER + _EXPERIMENT_CSV_ROW + _EXPERIMENT_CSV_ROW_OPENAI,
        encoding="utf-8",
    )
    return eid, exp_subdir


def test_run_filter_provider_runs_only_matching_rows(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """--filter-provider ollama should produce a results CSV with only ollama rows."""
    eid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
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
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1
    assert str(df["param_provider"].iloc[0]).lower() == "ollama"


def test_run_filter_provider_no_match_exits_cleanly(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """--filter-provider gemini on a CSV with no gemini rows exits 0, writes no results file."""
    eid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
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
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    assert len(results_files) == 0
    assert "nothing to run" in result.output.lower()


def test_run_filter_provider_case_insensitive(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """--filter-provider OLLAMA (upper-case) should match rows with param_provider=ollama."""
    eid, exp_subdir = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
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
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1


def test_run_filter_provider_dry_run_shows_filtered_count(
    experiment_with_two_provider_rows, mock_llm_mapper
):
    """Dry-run with --filter-provider should print the filtered row count (1), not the total (2)."""
    eid, _ = experiment_with_two_provider_rows

    result = runner.invoke(
        app,
        [
            "--dry-run",
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_20240101-abcd1234.csv",
            "--filter-provider",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    # Should mention 1 (filtered count), not 2 (total)
    assert "1" in result.output
