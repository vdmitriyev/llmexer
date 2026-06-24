"""Tests for the Experiment data class and the DAO-backed ExperimentsManager."""

import json
import os

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.base.dao import ExperimentDAO
from llmexer.base.llm_manager import Experiment, ExperimentsManager
from llmexer.base.llm_provider import CallerState, ProviderResponse
from llmexer.cli import app
from llmexer.exceptions import LLMExerException
from tests.db_helpers import OLLAMA_ROW, OPENAI_ROW, read_experiment_df, seed_db

runner = CliRunner()

_DB_NAME = "experiment_20240101_01.db"


@pytest.fixture()
def db_file(tmp_path):
    """A two-provider experiment database (ollama ID 1, openai ID 2)."""
    path = tmp_path / "experiment_test.db"
    seed_db(path, {"ollama": [dict(OLLAMA_ROW)], "openai": [dict(OPENAI_ROW)]})
    return str(path)


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
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
# DAO: schema + I/O
# ---------------------------------------------------------------------------


def test_open_missing_database_raises(tmp_path):
    with pytest.raises(LLMExerException):
        ExperimentsManager().open(str(tmp_path / "nope.db"))


def test_run_without_open_raises():
    with pytest.raises(LLMExerException):
        ExperimentsManager().run(1)


def test_provider_tables_are_isolated(db_file):
    """Each provider table carries only its own parameter columns."""
    with ExperimentDAO(db_file) as dao:
        tables = dao.provider_tables()
        assert set(tables) == {"ollama", "openai"}
        ollama_cols = list(tables["ollama"].c.keys())
        openai_cols = list(tables["openai"].c.keys())
    assert "ollama_context_window" in ollama_cols
    assert "ollama_context_window" not in openai_cols
    assert "openai_seed" in openai_cols
    assert "openai_seed" not in ollama_cols


def test_fetch_rows_tags_provider(db_file):
    with ExperimentDAO(db_file) as dao:
        rows = dao.fetch_rows()
    providers = {r["_provider"] for r in rows}
    assert providers == {"ollama", "openai"}


def test_fetch_rows_by_id_and_code(db_file):
    with ExperimentDAO(db_file) as dao:
        by_id = dao.fetch_rows(id_experiment=2)
        by_code = dao.fetch_rows(id_experiment="D01_prompt01_gpt-4o_openai-default")
    assert len(by_id) == 1 and by_id[0]["param_provider"] == "openai"
    assert len(by_code) == 1 and by_code[0]["ID"] == 2


# ---------------------------------------------------------------------------
# ExperimentsManager.run
# ---------------------------------------------------------------------------


def test_run_ollama_writes_state_back(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    exp = mgr.run(1)

    assert exp.status == "success"
    assert exp.state == CallerState.SUCCESS.value
    assert exp.response_text == "mocked response"
    assert exp.call_count == 1
    # State persisted into the database row.
    row = mgr.dao.fetch_rows(id_experiment=1)[0]
    assert row["status"] == "success"
    assert row["state"] == CallerState.SUCCESS.value
    assert row["response_text"] == "mocked response"
    assert json.loads(row["response_json"])["provider"] == "ollama"


def test_run_openai_branch(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    exp = mgr.run(2)
    assert exp.status == "success"
    assert exp.state == CallerState.SUCCESS.value
    assert exp.usage_tokens == 42


def test_run_by_code(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    exp = mgr.run("D01_prompt01_gpt-4o_openai-default")
    assert exp.param_provider == "openai"


def test_run_unknown_id_raises(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    with pytest.raises(LLMExerException):
        mgr.run(999)


def test_run_error_state_recorded(db_file, monkeypatch):
    import llmexer.base.llm_provider as llm_module

    class ErrorOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.ERROR

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(text="", usage_tokens=None, raw="refused")

    monkeypatch.setattr(llm_module, "OllamaProvider", ErrorOllamaProvider)

    mgr = ExperimentsManager(db_file)
    exp = mgr.run(1)
    assert exp.state == CallerState.ERROR.value
    assert "Error" in str(exp.status)


# ---------------------------------------------------------------------------
# ExperimentsManager.stats
# ---------------------------------------------------------------------------


def test_stats_pending_before_run(db_file):
    mgr = ExperimentsManager(db_file)
    data = mgr.stats()
    assert data["total"] == 2
    assert data["completed"] == 0
    assert data["pending"] == 2
    assert data["providers"] == {"ollama": 1, "openai": 1}
    assert set(data["models"]) == {"llama3.3:latest", "gpt-4o"}


def test_stats_after_run(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    mgr.run(1)
    mgr.run(2)
    data = mgr.stats()
    assert data["completed"] == 2
    assert data["errors"] == 0
    assert data["pending"] == 0
    assert data["total_tokens"] == 84


def test_stats_counts_errors(db_file, monkeypatch):
    import llmexer.base.llm_provider as llm_module

    class ErrorOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.ERROR

        def execute(self, prompt, row):
            self.state = CallerState.ERROR
            return ProviderResponse(text="", usage_tokens=None, raw="boom")

    monkeypatch.setattr(llm_module, "OllamaProvider", ErrorOllamaProvider)

    mgr = ExperimentsManager(db_file)
    mgr.run(1)
    data = mgr.stats()
    assert data["errors"] == 1
    assert data["pending"] == 1


# ---------------------------------------------------------------------------
# CLI: experiment stats
# ---------------------------------------------------------------------------


def test_cli_stats_command(projects_dir):
    pid = "stats-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {"ollama": [dict(OLLAMA_ROW)], "openai": [dict(OPENAI_ROW)]},
    )

    result = runner.invoke(
        app, ["experiment", "stats", "--pid", pid, "--file", _DB_NAME]
    )

    assert result.exit_code == 0, result.exception
    assert "total" in result.output
    assert "ollama" in result.output


def test_cli_stats_defaults_to_single_db(projects_dir, mock_providers):
    """With no --file, stats auto-discovers the single database."""
    pid = "stats-default-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / _DB_NAME, {"ollama": [dict(OLLAMA_ROW)]})

    run_result = runner.invoke(
        app, ["experiment", "run", "--pid", pid, "--file", _DB_NAME]
    )
    assert run_result.exit_code == 0, run_result.exception

    result = runner.invoke(app, ["experiment", "stats", "--pid", pid])
    assert result.exit_code == 0, result.exception
    assert "completed" in result.output
    assert "ollama" in result.output


def test_cli_stats_missing_db_raises(projects_dir):
    """stats with no --file and no database errors, pointing to `generate`."""
    pid = "stats-no-db"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)

    result = runner.invoke(app, ["experiment", "stats", "--pid", pid])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "generate" in str(result.exception).lower()


def test_cli_stats_multiple_dbs_requires_file(projects_dir):
    """stats with no --file errors when several databases exist."""
    pid = "stats-multi"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / "experiment_20240101_01.db", {"ollama": [dict(OLLAMA_ROW)]})
    seed_db(exp_subdir / "experiment_20240101_02.db", {"openai": [dict(OPENAI_ROW)]})

    result = runner.invoke(app, ["experiment", "stats", "--pid", pid])
    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "--file" in str(result.exception)


def test_cli_run_single_id(projects_dir, mock_providers):
    pid = "single-id-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {"ollama": [dict(OLLAMA_ROW)], "openai": [dict(OPENAI_ROW)]},
    )

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _DB_NAME, "--id", "1"],
    )

    assert result.exit_code == 0, result.exception
    df = read_experiment_df(exp_subdir / _DB_NAME)
    # Full row set persisted; only the --id 1 (ollama) row was run.
    assert len(df) == 2
    by_id = df.set_index("ID")
    assert by_id.loc[1, "status"] == "success"
    assert pd.isna(by_id.loc[2, "status"])
