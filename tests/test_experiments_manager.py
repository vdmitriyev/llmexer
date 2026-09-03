"""Tests for the Experiment data class and the DAO-backed ExperimentsManager."""

import json
import os
import sqlite3

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.base.dao import ExperimentDAO
from llmexer.base.llm_manager import Experiment, ExperimentsManager
from llmexer.base.llm_provider import CallerState, ProviderResponse
from llmexer.cli import app
from llmexer.exceptions import LLMExerException, ProviderConfigException
from tests.db_helpers import (
    LITELLM_ROW,
    OLLAMA_ROW,
    OPENAI_ROW,
    read_experiment_df,
    read_params_rows,
    seed_db,
)

runner = CliRunner()

_DB_NAME = "experiment_20240101_01.db"


@pytest.fixture()
def db_file(tmp_path):
    """A three-provider database (ollama ID 1, openai ID 2, litellm ID 3)."""
    path = tmp_path / "experiment_test.db"
    seed_db(
        path,
        {
            "ollama": [dict(OLLAMA_ROW)],
            "openai": [dict(OPENAI_ROW)],
            "litellm": [dict(LITELLM_ROW)],
        },
    )
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
    """Replace the provider classes with canned fakes."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerStats

    class FakeCompletion:
        """Stands in for an SDK response: serialize_response() calls model_dump."""

        def model_dump(self, mode=None):
            return {"id": "cmpl-1", "usage": {"prompt_tokens": 10, "total_tokens": 42}}

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED
            self.stats = CallerStats(call_count=1, total_tokens=42, elapsed_seconds=0.5)

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42)

    class FakeOpenAIProvider(FakeOllamaProvider):
        """Same canned behaviour, but carries a raw response worth serialising."""

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42, raw=FakeCompletion())

    class FakeLiteLLMProvider(FakeOllamaProvider):
        """Same canned behaviour, but records that config was validated."""

        validated = False

        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            super().__init__(provider, auth=auth, base_url=base_url, **kwargs)
            self.base_url = base_url
            self.auth = auth

        def validate_config(self):
            type(self).validated = True

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked litellm response", usage_tokens=42)

    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)
    monkeypatch.setattr(llm_module, "OpenAIProvider", FakeOpenAIProvider)
    monkeypatch.setattr(llm_module, "LiteLLMProvider", FakeLiteLLMProvider)
    return FakeLiteLLMProvider


# ---------------------------------------------------------------------------
# Experiment dataclass
# ---------------------------------------------------------------------------


def test_experiment_from_row_and_to_dict_roundtrip():
    row = {
        "ID": 1,
        "code": "D01_p01",
        "prompt": "Hi",
        "provider_name": "ollama",
        "model_name": "llama3.3:latest",
        "temperature": 0.7,
    }
    exp = Experiment.from_row(row)
    assert exp.experiment_id == "D01_p01"
    assert exp.row_id == 1
    assert exp.provider_name == "ollama"

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


def test_params_tables_are_isolated(db_file):
    """Each provider's params table carries only its own parameter columns."""
    with ExperimentDAO(db_file) as dao:
        tables = dao.params_tables()
        assert set(tables) == {"ollama", "openai", "litellm"}
        ollama_cols = list(tables["ollama"].c.keys())
        openai_cols = list(tables["openai"].c.keys())
        litellm_cols = list(tables["litellm"].c.keys())
    assert "ollama_context_window" in ollama_cols
    assert "ollama_context_window" not in openai_cols
    assert "openai_seed" in openai_cols
    assert "openai_seed" not in ollama_cols
    assert {"litellm_min_p", "litellm_best_of"} <= set(litellm_cols)
    assert "vllm_min_p" not in litellm_cols
    assert "ollama_context_window" not in litellm_cols
    assert "litellm_min_p" not in ollama_cols
    # Every params table is keyed the same way.
    for cols in (ollama_cols, openai_cols, litellm_cols):
        assert cols[:2] == ["params_code", "profile_name"]


def test_provider_tables_hold_no_parameters(db_file):
    """No parameter value is stored on an experiment table any more."""
    with ExperimentDAO(db_file) as dao:
        tables = dao.provider_tables()
        assert set(tables) == {"ollama", "openai", "litellm"}
        for provider, table in tables.items():
            cols = list(table.c.keys())
            assert "params_code" in cols, provider
            assert "profile_name" in cols, provider
            for absent in (
                "temperature",
                "top_p",
                "max_tokens",
                "ollama_context_window",
                "openai_seed",
                "litellm_min_p",
            ):
                assert absent not in cols, (provider, absent)


def test_fetch_rows_join_returns_flat_params(db_file):
    """A fetched row carries its parameters inline, with no duplicated keys."""
    with ExperimentDAO(db_file) as dao:
        row = dao.fetch_rows(id_experiment=1)[0]
    assert row["params_code"] == "llama3.3:latest_ollama"
    assert row["profile_name"] == "ollama-default"
    assert row["temperature"] == 0.7
    assert row["max_tokens"] == 512
    assert row["ollama_context_window"] == 4096
    assert row["ollama_repeat_penalty"] == 1.1
    # Selecting both tables whole would produce these suffixed duplicates.
    assert "params_code_1" not in row
    assert "profile_name_1" not in row


def test_fetch_rows_keeps_a_row_whose_params_are_missing(db_file):
    """An orphaned experiment row still comes back, with NULL parameters."""
    with ExperimentDAO(db_file) as dao:
        params = dao.params_tables()["ollama"]
        with dao.engine.begin() as conn:
            conn.execute(params.delete())
        rows = dao.fetch_rows(provider="ollama")
    assert len(rows) == 1
    assert rows[0]["ID"] == 1
    assert rows[0]["temperature"] is None
    assert rows[0]["ollama_context_window"] is None


def test_insert_rows_deduplicates_parameter_sets(tmp_path):
    """Many experiment rows sharing a profile store exactly one params row."""
    path = tmp_path / "dedup.db"
    rows = [{**OLLAMA_ROW, "ID": i, "code": f"D{i:02d}"} for i in range(1, 6)]
    with ExperimentDAO(str(path), create=True) as dao:
        assert dao.insert_rows("ollama", rows) == 5

    assert len(read_params_rows(path, "ollama")) == 1


def test_insert_rows_twice_reuses_one_params_row(tmp_path):
    """A second insert of the same profile must not violate the composite PK."""
    path = tmp_path / "twice.db"
    with ExperimentDAO(str(path), create=True) as dao:
        dao.insert_rows("ollama", [dict(OLLAMA_ROW)])
        dao.insert_rows("ollama", [{**OLLAMA_ROW, "ID": 9, "code": "D09"}])

    with ExperimentDAO(str(path)) as dao:
        assert len(dao.fetch_rows(provider="ollama")) == 2
    assert len(read_params_rows(path, "ollama")) == 1


def test_insert_rows_separates_profiles_of_one_model(tmp_path):
    """Two profiles for the same model are two params rows sharing a code."""
    path = tmp_path / "profiles.db"
    rows = [
        {**OLLAMA_ROW, "ID": 1, "code": "D01", "profile_name": "hot", "temperature": 1.2},
        {**OLLAMA_ROW, "ID": 2, "code": "D02", "profile_name": "cold", "temperature": 0.1},
    ]
    with ExperimentDAO(str(path), create=True) as dao:
        dao.insert_rows("ollama", rows)

    params = read_params_rows(path, "ollama")
    assert [p["profile_name"] for p in params] == ["cold", "hot"]
    assert {p["params_code"] for p in params} == {"llama3.3:latest_ollama"}
    with ExperimentDAO(str(path)) as dao:
        fetched = {r["ID"]: r["temperature"] for r in dao.fetch_rows(provider="ollama")}
    assert fetched == {1: 1.2, 2: 0.1}


def test_open_old_layout_database_raises(tmp_path):
    """A pre-split database has no params table and must be rejected."""
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE experiment_ollama "
        "(ID INTEGER PRIMARY KEY, code VARCHAR, profile_name VARCHAR, temperature FLOAT)"
    )
    con.commit()
    con.close()

    with pytest.raises(LLMExerException) as exc:
        ExperimentDAO(str(path))
    message = str(exc.value)
    assert "params_ollama" in message
    assert "experiment generate" in message


def test_open_old_layout_database_raises_through_manager(tmp_path):
    """The same clean break surfaces through ExperimentsManager.open()."""
    path = tmp_path / "legacy_manager.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE experiment_ollama (ID INTEGER PRIMARY KEY, code VARCHAR)")
    con.commit()
    con.close()

    with pytest.raises(LLMExerException):
        ExperimentsManager().open(str(path))


def test_fetch_rows_tags_provider(db_file):
    with ExperimentDAO(db_file) as dao:
        rows = dao.fetch_rows()
    providers = {r["_provider"] for r in rows}
    assert providers == {"ollama", "openai", "litellm"}


def test_fetch_rows_by_id_and_code(db_file):
    with ExperimentDAO(db_file) as dao:
        by_id = dao.fetch_rows(id_experiment=2)
        by_code = dao.fetch_rows(id_experiment="D01_prompt01_gpt-4o_openai-default")
    assert len(by_id) == 1 and by_id[0]["provider_name"] == "openai"
    assert len(by_code) == 1 and by_code[0]["ID"] == 2


def test_fetch_rows_by_model_and_profile(db_file):
    """model_name/profile_name select in full, case-sensitively, and combine."""
    with ExperimentDAO(db_file) as dao:
        by_model = dao.fetch_rows(model_name="gpt-4o")
        by_profile = dao.fetch_rows(profile_name="ollama-default")
        combined = dao.fetch_rows(model_name="gpt-4o", profile_name="openai-default")
        wrong_case = dao.fetch_rows(model_name="GPT-4O")
        partial = dao.fetch_rows(profile_name="ollama")
        mismatched = dao.fetch_rows(model_name="gpt-4o", profile_name="ollama-default")

    assert len(by_model) == 1 and by_model[0]["_provider"] == "openai"
    assert len(by_profile) == 1 and by_profile[0]["ID"] == 1
    # The joined parameter values still come back alongside the filtered row.
    assert by_profile[0]["ollama_context_window"] == 4096
    assert len(combined) == 1 and combined[0]["ID"] == 2
    assert wrong_case == []
    assert partial == []
    assert mismatched == []


# ---------------------------------------------------------------------------
# ExperimentsManager.run
# ---------------------------------------------------------------------------


def test_run_ollama_writes_state_back(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    exp = mgr.run(1)

    assert exp.status == "success"
    assert exp.state == CallerState.FINISHED.value
    assert exp.response_text == "mocked response"
    assert exp.call_count == 1
    # State persisted into the database row.
    row = mgr.dao.fetch_rows(id_experiment=1)[0]
    assert row["status"] == "success"
    assert row["state"] == CallerState.FINISHED.value
    assert row["response_text"] == "mocked response"
    assert json.loads(row["response_json"])["provider"] == "ollama"


def test_run_openai_branch(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    exp = mgr.run(2)
    assert exp.status == "success"
    assert exp.state == CallerState.FINISHED.value
    assert exp.usage_tokens == 42
    # The full backend response is captured and persisted into response_json.
    assert exp.raw_response["usage"]["prompt_tokens"] == 10
    saved = json.loads(mgr.dao.fetch_rows(id_experiment=2)[0]["response_json"])
    assert saved["raw_response"]["usage"]["total_tokens"] == 42


def test_run_litellm_dispatches_to_provider_class(db_file, mock_providers):
    """A litellm row goes through LiteLLMProvider, not the legacy mapper."""
    mgr = ExperimentsManager(db_file)
    exp = mgr.run(3)

    assert exp.status == "success"
    assert exp.state == CallerState.FINISHED.value
    assert exp.response_text == "mocked litellm response"
    assert exp.usage_tokens == 42
    row = mgr.dao.fetch_rows(id_experiment=3)[0]
    assert row["response_text"] == "mocked litellm response"
    assert json.loads(row["response_json"])["provider"] == "litellm"


def test_run_litellm_validates_config_before_calling(db_file, mock_providers):
    """Credentials are checked up front, not swallowed into a row status."""
    mgr = ExperimentsManager(db_file)
    mgr.run(3)
    assert mock_providers.validated is True


def test_run_litellm_config_error_aborts_the_run(db_file, monkeypatch):
    """A missing token raises out of the run instead of writing an error row."""
    import llmexer.base.llm_provider as llm_module

    monkeypatch.delenv("PROVIDER_LITELLM_URL", raising=False)
    monkeypatch.delenv("PROVIDER_LITELLM_KEY", raising=False)

    called = []

    class ExplodingProvider(llm_module.LiteLLMProvider):
        def execute(self, prompt, row):  # pragma: no cover - must not run
            called.append(prompt)
            raise AssertionError("execute must not be reached")

    monkeypatch.setattr(llm_module, "LiteLLMProvider", ExplodingProvider)

    mgr = ExperimentsManager(db_file)
    with pytest.raises(ProviderConfigException):
        mgr.run(3)
    assert called == []
    # Nothing was written back for that row.
    assert mgr.dao.fetch_rows(id_experiment=3)[0]["status"] is None


def test_build_response_payload_includes_raw_response():
    from llmexer.base.llm_manager import build_response_payload

    exp = Experiment(
        model_name="gemma4:31b",
        response_text="hi",
        raw_response={"usage": {"prompt_tokens": 3}, "eval_count": 9},
    )
    payload = build_response_payload(exp, "ollama")
    assert payload["raw_response"] == {"usage": {"prompt_tokens": 3}, "eval_count": 9}


def test_run_by_code(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    exp = mgr.run("D01_prompt01_gpt-4o_openai-default")
    assert exp.provider_name == "openai"


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


def test_stats_before_run(db_file):
    mgr = ExperimentsManager(db_file)
    data = mgr.stats()
    assert data["total"] == 3
    assert data["finished"] == 0
    assert "pending" not in data
    assert data["providers"] == {"ollama": 1, "openai": 1, "litellm": 1}
    assert set(data["models"]) == {"llama3.3:latest", "gpt-4o", "gpt-oss:120b"}
    # Each model is an aggregate dict: nothing run yet -> all open, no finished.
    for agg in data["models"].values():
        assert agg["requests"] == 1
        assert agg["finished"] == 0
        assert agg["open"] == 1
        assert agg["tokens"] == 0
        assert agg["elapsed_seconds"] == 0.0
        assert agg["avg_elapsed_seconds"] == 0.0


def test_stats_after_run(db_file, mock_providers):
    mgr = ExperimentsManager(db_file)
    mgr.run(1)
    mgr.run(2)
    data = mgr.stats()
    assert data["finished"] == 2
    assert data["errors"] == 0
    assert data["total_tokens"] == 84
    # Per-model aggregates reflect the finished run (mock yields 42 tokens each).
    ollama = data["models"]["llama3.3:latest"]
    assert ollama["finished"] == 1
    assert ollama["open"] == 0
    assert ollama["tokens"] == 42
    # One finished request → average elapsed equals the total elapsed.
    assert ollama["avg_elapsed_seconds"] == ollama["elapsed_seconds"]


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
    assert data["finished"] == 0


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

    result = runner.invoke(app, ["experiment", "stats", "--pid", pid, "--file", _DB_NAME])

    assert result.exit_code == 0, result.exception
    assert "total" in result.output
    assert "ollama" in result.output
    # The Models table now carries per-model aggregate columns.
    for header in ("Model", "finished", "open", "time total", "tokens"):
        assert header in result.output
    # The average-time column header is present (single token, robust to wrapping).
    assert "average" in result.output


def test_cli_stats_defaults_to_single_db(projects_dir, mock_providers):
    """With no --file, stats auto-discovers the single database."""
    pid = "stats-default-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(exp_subdir / _DB_NAME, {"ollama": [dict(OLLAMA_ROW)]})

    run_result = runner.invoke(app, ["experiment", "run", "--pid", pid, "--file", _DB_NAME])
    assert run_result.exit_code == 0, run_result.exception

    result = runner.invoke(app, ["experiment", "stats", "--pid", pid])
    assert result.exit_code == 0, result.exception
    assert "finished" in result.output
    assert "ollama" in result.output


def test_format_hms():
    from llmexer.commands.experiment import _format_hms

    assert _format_hms(0) == "00:00:00"
    assert _format_hms(3661) == "01:01:01"
    assert _format_hms(59) == "00:00:59"
    assert _format_hms(90061) == "25:01:01"  # hours not capped at 24
    assert _format_hms(None) == "00:00:00"


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


def test_cli_run_single_code(projects_dir, mock_providers):
    pid = "single-code-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {"ollama": [dict(OLLAMA_ROW)], "openai": [dict(OPENAI_ROW)]},
    )

    result = runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _DB_NAME, "--code", "1"],
    )

    assert result.exit_code == 0, result.exception
    df = read_experiment_df(exp_subdir / _DB_NAME)
    # Full row set persisted; only the --code 1 (ollama) row was run.
    assert len(df) == 2
    by_id = df.set_index("ID")
    assert by_id.loc[1, "status"] == "success"
    assert pd.isna(by_id.loc[2, "status"])
