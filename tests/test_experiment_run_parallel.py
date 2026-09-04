"""Tests for the `--parallel-calls` option of the `experiment run` command."""

import os
import threading
import time

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import ProviderConfigException, UnexpectedCLIParamsException
from tests.db_helpers import OLLAMA_ROW, find_db, read_experiment_df, seed_db

runner = CliRunner()

# Name of the generated experiment database used throughout these tests.
_EXPERIMENT_DB_NAME = "experiment_20240101_01.db"

# Number of rows seeded into the test database.
_ROW_COUNT = 6


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


def _ollama_row(row_id):
    """Build one ollama row of the cross join, distinguishable by its ID."""
    row = dict(OLLAMA_ROW)
    row.update({"ID": row_id, "code": f"D0{row_id}_prompt01_llama3.3:latest_ollama-default"})
    return row


@pytest.fixture()
def experiment_with_rows(projects_dir):
    """Database with a single ollama table holding several pending rows."""
    pid = "parallel-run-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _EXPERIMENT_DB_NAME,
        {"ollama": [_ollama_row(row_id) for row_id in range(1, _ROW_COUNT + 1)]},
    )
    return pid, exp_subdir


class ConcurrencyProbe:
    """Records how many provider calls were in flight at the same time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    def enter(self):
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)

    def leave(self):
        with self.lock:
            self.in_flight -= 1


@pytest.fixture()
def probe_provider(monkeypatch):
    """Replace the ollama provider with a fake that measures concurrency."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    probe = ConcurrencyProbe()

    class ProbeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            probe.enter()
            try:
                # Long enough for the other workers to pile up behind it.
                time.sleep(0.05)
            finally:
                probe.leave()
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42)

    monkeypatch.setattr(llm_module, "OllamaProvider", ProbeOllamaProvider)
    return probe


def _run_with(pid, *options):
    """Invoke `experiment run` on the seeded database with extra options."""
    return runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME, *options],
    )


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


def test_run_parallel_calls_overlap_but_stay_under_the_cap(experiment_with_rows, probe_provider):
    """--parallel-calls 4 runs calls at once, never more than four of them."""
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", "4")

    assert result.exit_code == 0, result.output
    assert probe_provider.calls == _ROW_COUNT
    assert probe_provider.peak > 1
    assert probe_provider.peak <= 4
    df = read_experiment_df(find_db(exp_subdir))
    assert list(df["status"]) == ["success"] * _ROW_COUNT


def test_run_default_is_sequential(experiment_with_rows, probe_provider):
    """Without the option, only one call is ever in flight."""
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid)

    assert result.exit_code == 0, result.output
    assert probe_provider.peak == 1
    df = read_experiment_df(find_db(exp_subdir))
    assert list(df["status"]) == ["success"] * _ROW_COUNT


def test_run_parallel_calls_one_is_sequential(experiment_with_rows, probe_provider):
    """--parallel-calls 1 takes the sequential path, with no overlap."""
    pid, _ = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", "1")

    assert result.exit_code == 0, result.output
    assert probe_provider.peak == 1


def test_run_parallel_calls_writes_results_one_at_a_time(experiment_with_rows, probe_provider, monkeypatch):
    """Result writing is serialised even though the calls overlap."""
    pid, exp_subdir = experiment_with_rows

    from llmexer.base.dao import ExperimentDAO

    original = ExperimentDAO.update_result
    writes = ConcurrencyProbe()
    written_ids = []

    def recording_update_result(self, provider, row_id, result):
        writes.enter()
        try:
            written_ids.append(row_id)
            time.sleep(0.01)
            return original(self, provider, row_id, result)
        finally:
            writes.leave()

    monkeypatch.setattr(ExperimentDAO, "update_result", recording_update_result)

    result = _run_with(pid, "--parallel-calls", "4")

    assert result.exit_code == 0, result.output
    # No two writes were ever in flight together.
    assert writes.peak == 1
    assert sorted(written_ids) == list(range(1, _ROW_COUNT + 1))


def test_run_parallel_calls_saves_one_json_per_row(experiment_with_rows, probe_provider):
    """Every parallel call gets its own response file, with no name collision."""
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", "4")

    assert result.exit_code == 0, result.output
    json_files = list((exp_subdir / "responses").glob("*.json"))
    assert len(json_files) == _ROW_COUNT


def test_run_parallel_calls_reports_the_cap(experiment_with_rows, probe_provider):
    """The chosen cap is echoed back to the user."""
    pid, _ = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", "4")

    assert result.exit_code == 0, result.output
    assert "Parallel LLM calls" in result.output


# ---------------------------------------------------------------------------
# Validation / dry-run / failure tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-1"])
def test_run_parallel_calls_rejects_values_below_one(experiment_with_rows, probe_provider, value):
    """--parallel-calls must be at least 1."""
    pid, _ = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", value)

    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert probe_provider.calls == 0


def test_run_parallel_calls_dry_run_makes_no_calls(experiment_with_rows, probe_provider):
    """--dry-run with a cap set still calls nothing and writes nothing."""
    pid, exp_subdir = experiment_with_rows

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
            "--parallel-calls",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert probe_provider.calls == 0
    assert not (exp_subdir / "responses").exists()
    df = read_experiment_df(find_db(exp_subdir))
    assert df["status"].isna().all()


def test_run_parallel_calls_aborts_on_provider_config_error(experiment_with_rows, monkeypatch):
    """A configuration error aborts the run instead of erroring every row."""
    pid, exp_subdir = experiment_with_rows

    import llmexer.base.llm_manager as manager_module

    monkeypatch.setitem(manager_module.PROVIDER_CLASS_NAMES, "ollama", None)

    result = _run_with(pid, "--parallel-calls", "4")

    assert result.exit_code != 0
    assert isinstance(result.exception, ProviderConfigException)
    # Nothing was recorded: the run stopped rather than writing error rows.
    df = read_experiment_df(find_db(exp_subdir))
    assert df["status"].isna().all()
