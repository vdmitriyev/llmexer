"""Tests for the Experiment data class and ExperimentsManager mapper."""

import json
import os

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.base.llm_manager import Experiment, ExperimentsManager
from llmexer.base.llm_provider import CallerState, ProviderResponse
from llmexer.cli import app
from llmexer.exceptions import LLMExerException

runner = CliRunner()

_HEADER = (
    "ID;code;prompt;tokens_estimate;original_data;model_name;provider_name;prompt_hash;"
    "original_data_hash;profile_name;param_model_name;param_provider;temperature;top_p;"
    "max_tokens;ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;"
    "openai_seed;gemini_thinking_level\n"
)
_ROW_OLLAMA = (
    "1;D01_prompt01_llama3.3:latest_ollama-default;Hello world;2;"
    '{"ID":"D01"};llama3.3:latest;ollama;abc123;def456;'
    "ollama-default;llama3.3:latest;ollama;0.7;1.0;512;4096;1.1;;;;\n"
)
_ROW_OPENAI = (
    "2;D01_prompt01_gpt-4o_openai-default;Hello world;2;"
    '{"ID":"D01"};gpt-4o;openai;abc123;def456;'
    "openai-default;gpt-4o;openai;0.7;1.0;512;;;;42;\n"
)


@pytest.fixture()
def csv_file(tmp_path):
    path = tmp_path / "experiment_test.csv"
    path.write_text(_HEADER + _ROW_OLLAMA + _ROW_OPENAI, encoding="utf-8")
    return str(path)


@pytest.fixture()
def experiments_dir(tmp_path, monkeypatch):
    import llmexer.commands.experiment as exp_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "EXPERIMENTS_PATH", str(tmp_path))
    monkeypatch.setattr(exp_module, "EXPERIMENTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_providers(monkeypatch):
    """Replace LLMRequestsMapper / OllamaProvider with canned fakes."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerStats

    class FakeResult:
        response_text = "mocked response"
        usage_tokens = 42
        status = "success"
        timestamp = "2024-01-01T00:00:00"

    class FakeMapper:
        def __init__(self, provider, base_url=None, api_key="na"):
            self.provider = provider

        def execute(self, prompt, row):
            return FakeResult()

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.SUCCESS
            self.stats = CallerStats(call_count=1, total_tokens=42, elapsed_seconds=0.5)

        def execute(self, prompt, row):
            self.state = CallerState.SUCCESS
            return ProviderResponse(text="mocked response", usage_tokens=42)

    monkeypatch.setattr(llm_module, "LLMRequestsMapper", FakeMapper)
    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)


# ---------------------------------------------------------------------------
# Experiment dataclass
# ---------------------------------------------------------------------------


def test_experiment_from_row_and_to_dict_roundtrip():
    row = {
        "ID": 1,
        "code": "D01_p01",
        "prompt": "Hi",
        "param_provider": "ollama",
        "param_model_name": "llama3.3:latest",
        "temperature": 0.7,
    }
    exp = Experiment.from_row(row)
    assert exp.experiment_id == "D01_p01"
    assert exp.row_id == 1
    assert exp.param_provider == "ollama"

    merged = exp.to_dict()
    assert merged["code"] == "D01_p01"
    assert "status" in merged and "state" in merged


def test_experiment_from_row_handles_missing_result_columns():
    exp = Experiment.from_row({"ID": 5, "code": "x", "prompt": "p"})
    assert exp.status is None
    assert exp.call_count == 0
    assert exp.total_tokens == 0


def test_experiment_to_json_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = Experiment(experiment_id="D01_llama3.3:latest", response_text="hi")
    path = exp.to_json()
    # Colons made filesystem-safe.
    assert path == "D01_llama3.3-latest.json"
    with open(tmp_path / path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["response_text"] == "hi"
    assert "raw" not in data


def test_experiment_to_json_explicit_file_is_indented(tmp_path):
    out = tmp_path / "out.json"
    exp = Experiment(experiment_id="abc", response_text="hi")
    exp.to_json(str(out))
    text = out.read_text(encoding="utf-8")
    # indent=4 produces leading 4-space indentation.
    assert "\n    " in text


def test_experiment_to_yaml(tmp_path):
    out = tmp_path / "out.yaml"
    exp = Experiment(experiment_id="abc", response_text="hi", usage_tokens=7)
    exp.to_yaml(str(out))
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["experiment_id"] == "abc"
    assert data["usage_tokens"] == 7
    assert "raw" not in data


# ---------------------------------------------------------------------------
# ExperimentsManager I/O
# ---------------------------------------------------------------------------


def test_load_returns_dataframe(csv_file):
    mgr = ExperimentsManager()
    df = mgr.load(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert mgr.file == csv_file


def test_load_without_file_raises():
    with pytest.raises(LLMExerException):
        ExperimentsManager().load()


def test_unload_roundtrip_preserves_data(csv_file, tmp_path):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    out = str(tmp_path / "dump.csv")
    mgr.unload(out)
    reloaded = pd.read_csv(out, sep=";", encoding="utf-8")
    assert list(reloaded["code"]) == list(mgr.df["code"])
    assert len(reloaded) == 2


def test_sync_writes_back_to_source(csv_file):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    mgr.df.loc[0, "prompt"] = "changed"
    mgr.sync()
    reloaded = pd.read_csv(csv_file, sep=";", encoding="utf-8")
    assert reloaded.loc[0, "prompt"] == "changed"


def test_operations_before_load_raise(csv_file):
    mgr = ExperimentsManager()
    with pytest.raises(LLMExerException):
        mgr.stats()
    with pytest.raises(LLMExerException):
        mgr.unload()


# ---------------------------------------------------------------------------
# ExperimentsManager.run
# ---------------------------------------------------------------------------


def test_run_ollama_writes_state_back(csv_file, mock_providers):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    exp = mgr.run(1)

    assert exp.status == "success"
    assert exp.state == CallerState.SUCCESS.value
    assert exp.response_text == "mocked response"
    assert exp.call_count == 1
    # State written back into the DataFrame.
    assert mgr.df.loc[0, "status"] == "success"
    assert mgr.df.loc[0, "state"] == CallerState.SUCCESS.value
    assert mgr.df.loc[0, "response_text"] == "mocked response"


def test_run_openai_branch(csv_file, mock_providers):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    exp = mgr.run(2)
    assert exp.status == "success"
    assert exp.state == CallerState.SUCCESS.value
    assert exp.usage_tokens == 42


def test_run_by_code(csv_file, mock_providers):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    exp = mgr.run("D01_prompt01_gpt-4o_openai-default")
    assert exp.param_provider == "openai"


def test_run_unknown_id_raises(csv_file, mock_providers):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    with pytest.raises(LLMExerException):
        mgr.run(999)


def test_run_error_state_recorded(csv_file, monkeypatch):
    import llmexer.base.llm_provider as llm_module

    class ErrorOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.ERROR

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(text="", usage_tokens=None, raw="refused")

    monkeypatch.setattr(llm_module, "OllamaProvider", ErrorOllamaProvider)

    mgr = ExperimentsManager()
    mgr.load(csv_file)
    exp = mgr.run(1)
    assert exp.state == CallerState.ERROR.value
    assert "Error" in str(exp.status)


# ---------------------------------------------------------------------------
# ExperimentsManager.stats
# ---------------------------------------------------------------------------


def test_stats_pending_before_run(csv_file):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    data = mgr.stats()
    assert data["total"] == 2
    assert data["completed"] == 0
    assert data["pending"] == 2
    assert data["providers"] == {"ollama": 1, "openai": 1}
    assert set(data["models"]) == {"llama3.3:latest", "gpt-4o"}


def test_stats_after_run(csv_file, mock_providers):
    mgr = ExperimentsManager()
    mgr.load(csv_file)
    mgr.run(1)
    mgr.run(2)
    data = mgr.stats()
    assert data["completed"] == 2
    assert data["errors"] == 0
    assert data["pending"] == 0
    assert data["total_tokens"] == 84


def test_stats_counts_errors(csv_file, monkeypatch):
    import llmexer.base.llm_provider as llm_module

    class ErrorOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.ERROR

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(text="", usage_tokens=None, raw="boom")

    monkeypatch.setattr(llm_module, "OllamaProvider", ErrorOllamaProvider)

    mgr = ExperimentsManager()
    mgr.load(csv_file)
    mgr.run(1)
    data = mgr.stats()
    assert data["errors"] == 1
    assert data["pending"] == 1


# ---------------------------------------------------------------------------
# CLI: experiment stats
# ---------------------------------------------------------------------------


def test_cli_stats_command(experiments_dir):
    eid = "stats-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    os.makedirs(exp_subdir)
    (exp_subdir / "experiment_test.csv").write_text(
        _HEADER + _ROW_OLLAMA + _ROW_OPENAI, encoding="utf-8"
    )

    result = runner.invoke(
        app,
        ["experiment", "stats", "--eid", eid, "--file", "experiment_test.csv"],
    )

    assert result.exit_code == 0, result.exception
    assert "total" in result.output
    assert "ollama" in result.output


def test_cli_run_single_id(experiments_dir, mock_providers):
    eid = "single-id-exp"
    exp_subdir = experiments_dir / eid / "experiment"
    os.makedirs(exp_subdir)
    (exp_subdir / "experiment_test.csv").write_text(
        _HEADER + _ROW_OLLAMA + _ROW_OPENAI, encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "experiment",
            "run",
            "--eid",
            eid,
            "--file",
            "experiment_test.csv",
            "--id",
            "1",
        ],
    )

    assert result.exit_code == 0, result.exception
    results_files = [
        f
        for f in os.listdir(exp_subdir)
        if f.startswith(f"experiment_{eid}_results_") and f.endswith(".csv")
    ]
    assert len(results_files) == 1
    df = pd.read_csv(exp_subdir / results_files[0], sep=";", encoding="utf-8")
    assert len(df) == 1
    assert str(df["param_provider"].iloc[0]).lower() == "ollama"
