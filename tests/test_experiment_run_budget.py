"""Tests for the per-run spending cap of the `experiment run` command."""

import os

import pandas as pd
import pytest
from typer.testing import CliRunner

from llmexer.base.dao import ExperimentDAO
from llmexer.cli import app
from tests.db_helpers import (
    OPENROUTER_ROW,
    find_db,
    read_cost_logs,
    read_experiment_df,
    seed_db,
)

runner = CliRunner()

# Name of the generated experiment database used throughout these tests.
_EXPERIMENT_DB_NAME = "experiment_20240101_01.db"

# Number of rows seeded into the test database.
_ROW_COUNT = 6

# What each mocked call reports as its cost, and the cap the tests run under:
# three calls exhaust it, so the run must stop with rows still pending.
_COST_PER_CALL = 1.0
_CAP = "3.0"


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


def _openrouter_row(row_id):
    """Build one openrouter row of the cross join, distinguishable by its ID."""
    row = dict(OPENROUTER_ROW)
    row.update(
        {
            "ID": row_id,
            "code": f"D0{row_id}_prompt01_google/gemini-3.8-flash_openrouter-gemini-default",
            "data_id": f"D0{row_id}",
        }
    )
    return row


@pytest.fixture()
def experiment_with_rows(projects_dir):
    """Database with a single openrouter table holding several pending rows."""
    pid = "budget-run-exp"
    exp_subdir = projects_dir / pid / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _EXPERIMENT_DB_NAME,
        {"openrouter": [_openrouter_row(row_id) for row_id in range(1, _ROW_COUNT + 1)]},
    )
    return pid, exp_subdir


@pytest.fixture()
def paid_provider(monkeypatch):
    """Replace the openrouter provider with a fake that charges a fixed amount."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    calls = []

    class PaidOpenRouterProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            calls.append(row["ID"])
            self.state = CallerState.FINISHED
            return ProviderResponse(text="paid answer", total_tokens=42, cost_usd=_COST_PER_CALL)

    monkeypatch.setattr(llm_module, "OpenRouterProvider", PaidOpenRouterProvider)
    return calls


@pytest.fixture()
def free_provider(monkeypatch):
    """Replace the openrouter provider with one that reports no cost at all."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FreeOpenRouterProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="free answer", total_tokens=42)

    monkeypatch.setattr(llm_module, "OpenRouterProvider", FreeOpenRouterProvider)


def _run_with(pid, *options):
    """Invoke `experiment run` on the seeded database with extra options."""
    return runner.invoke(
        app,
        ["experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME, *options],
    )


def _finished_ids(exp_subdir):
    """IDs whose status is 'success' in the database."""
    df = read_experiment_df(find_db(exp_subdir))
    return sorted(df[df["status"] == "success"]["ID"])


# ---------------------------------------------------------------------------
# The cap stops a run
# ---------------------------------------------------------------------------


def test_run_pauses_once_the_cap_is_reached(experiment_with_rows, paid_provider, monkeypatch):
    """Three calls of $1.00 exhaust a $3.00 cap; the rest are left for later."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid)

    assert result.exit_code == 0, result.output
    assert paid_provider == [1, 2, 3]
    assert _finished_ids(exp_subdir) == [1, 2, 3]


def test_pausing_is_graceful_not_an_abort(experiment_with_rows, paid_provider, monkeypatch):
    """No traceback and no error exit: the run stops, it does not fail."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, _ = experiment_with_rows

    result = _run_with(pid)

    assert result.exit_code == 0
    assert result.exception is None


def test_unrun_rows_stay_pending(experiment_with_rows, paid_provider, monkeypatch):
    """The rows behind the cap are untouched, so a later run still sees them."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    df = read_experiment_df(find_db(exp_subdir))
    unrun = df[df["ID"] > 3]
    assert unrun["status"].isna().all()
    assert unrun["response_text"].isna().all()


def test_the_pause_is_reported(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, _ = experiment_with_rows

    result = _run_with(pid)

    output = " ".join(result.output.split())
    assert "Budget reached" in output
    assert "$3.00 cap" in output
    assert "3 row(s) not run" in output


def test_a_sub_cent_cap_is_not_rendered_as_zero(experiment_with_rows, paid_provider, monkeypatch):
    """$0.004 shown as "$0.00" would read as "no budget at all"."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "0.004")
    pid, _ = experiment_with_rows

    result = _run_with(pid)

    output = " ".join(result.output.split())
    assert "Spending cap for this run: $0.0040" in output


def test_the_cap_is_announced_up_front(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "7.5")
    pid, _ = experiment_with_rows

    result = _run_with(pid)

    output = " ".join(result.output.split())
    assert "Spending cap for this run: $7.50" in output


def test_the_saved_count_matches_the_rows_actually_run(experiment_with_rows, paid_provider, monkeypatch):
    """The count must not include rows the cap stopped."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, _ = experiment_with_rows

    result = _run_with(pid)

    output = " ".join(result.output.split())
    assert "Saved 3 result(s)" in output


def test_a_paused_run_resumes_where_it_stopped(experiment_with_rows, paid_provider, monkeypatch):
    """The cap is per run, so the next invocation starts from the full budget."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)
    assert _finished_ids(exp_subdir) == [1, 2, 3]

    second = _run_with(pid)

    assert second.exit_code == 0, second.output
    assert _finished_ids(exp_subdir) == [1, 2, 3, 4, 5, 6]
    # Every row was called exactly once across the two runs.
    assert paid_provider == [1, 2, 3, 4, 5, 6]


def test_a_generous_cap_runs_everything(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "1000")
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid)

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == [1, 2, 3, 4, 5, 6]
    assert "Budget reached" not in result.output


def test_no_pause_notice_when_the_cap_falls_on_the_last_row(experiment_with_rows, paid_provider, monkeypatch):
    """Exhausted with nothing left to run is not a pause."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", str(_ROW_COUNT * _COST_PER_CALL))
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid)

    assert _finished_ids(exp_subdir) == [1, 2, 3, 4, 5, 6]
    assert "Budget reached" not in result.output


def test_a_provider_reporting_no_cost_never_pauses(experiment_with_rows, free_provider, monkeypatch):
    """Nothing measured means nothing booked; the run is not silently halted."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "0.001")
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid)

    assert result.exit_code == 0, result.output
    assert _finished_ids(exp_subdir) == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# cost_logs
# ---------------------------------------------------------------------------


def test_each_paid_call_is_logged(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    logs = read_cost_logs(find_db(exp_subdir))
    assert [entry["row_id"] for entry in logs] == [1, 2, 3]
    assert all(entry["cost_usd"] == _COST_PER_CALL for entry in logs)


def test_the_log_carries_the_identity_columns(experiment_with_rows, paid_provider, monkeypatch):
    """data_id / prompt_id are columns of their own, not buried inside code."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    entry = read_cost_logs(find_db(exp_subdir))[0]
    assert entry["table_name"] == "experiment_openrouter"
    assert entry["data_id"] == "D01"
    assert entry["prompt_id"] == "prompt01"
    assert entry["profile_name"] == "openrouter-gemini-default"
    assert entry["code"].startswith("D01_prompt01_")
    assert entry["created_at"].endswith("+00:00")


def test_the_spend_is_reported(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, _ = experiment_with_rows

    result = _run_with(pid)

    output = " ".join(result.output.split())
    assert "Spent $3.00 on this run" in output
    assert "cost_logs" in output


def test_stats_sums_the_logged_costs(experiment_with_rows, paid_provider, monkeypatch):
    """`experiment stats` reports total_costs as the sum of cost_logs.cost_usd."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    with ExperimentDAO(find_db(exp_subdir)) as dao:
        data = dao.stats()
    assert data["total_costs"] == pytest.approx(3 * _COST_PER_CALL)

    result = runner.invoke(app, ["experiment", "stats", "--pid", pid, "--file", _EXPERIMENT_DB_NAME])

    assert result.exit_code == 0, result.output
    # Rich draws a box, so compare on the cell values with the borders removed.
    cells = " ".join(result.output.replace("\u2502", " ").split())
    assert "total_costs $3.00" in cells


def test_a_free_provider_writes_no_cost_log(experiment_with_rows, free_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "1000")
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    assert read_cost_logs(find_db(exp_subdir)) == []


def test_stats_without_a_cost_log_reports_zero(experiment_with_rows, free_provider, monkeypatch):
    """A database with no cost_logs table sums to 0.0 instead of failing."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "1000")
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    with ExperimentDAO(find_db(exp_subdir)) as dao:
        assert dao.stats()["total_costs"] == 0.0


def test_the_cost_reaches_the_response_json(experiment_with_rows, paid_provider, monkeypatch):
    """No result column was added, so the JSON payload is where cost is stored."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid)

    df = read_experiment_df(find_db(exp_subdir))
    payload = df[df["ID"] == 1].iloc[0]["response_json"]
    assert '"cost_usd": 1.0' in payload


# ---------------------------------------------------------------------------
# Parallel path
# ---------------------------------------------------------------------------


def test_the_cap_stops_a_parallel_run_too(experiment_with_rows, paid_provider, monkeypatch):
    """Rows already in flight finish, but the queued ones are not called."""
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", "2")

    assert result.exit_code == 0, result.output
    finished = _finished_ids(exp_subdir)
    # At least the rows that exhausted the cap, never every row: the gate held.
    assert 3 <= len(finished) < _ROW_COUNT


def test_the_parallel_count_excludes_rows_the_cap_stopped(experiment_with_rows, paid_provider, monkeypatch):
    """Regression: the counter used to tick per finished future, skips included.

    The cap is below a single call here, so the first result to be booked
    exhausts it and only the calls already in flight can get through.
    """
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "0.5")
    pid, exp_subdir = experiment_with_rows

    result = _run_with(pid, "--parallel-calls", "4")

    finished = _finished_ids(exp_subdir)
    logs = read_cost_logs(find_db(exp_subdir))
    output = " ".join(result.output.split())
    # The reported count, the rows written and the calls logged must agree.
    assert f"Saved {len(finished)} result(s)" in output
    assert len(logs) == len(finished)
    # Never more than the calls that were already in flight: that is the
    # documented overshoot bound of a check-then-act gate.
    assert 1 <= len(finished) <= 4


def test_a_paused_parallel_run_resumes(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    _run_with(pid, "--parallel-calls", "2")
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", "1000")
    second = _run_with(pid, "--parallel-calls", "2")

    assert second.exit_code == 0, second.output
    assert _finished_ids(exp_subdir) == [1, 2, 3, 4, 5, 6]
    # No row was called twice: the resume skipped what had already succeeded.
    assert sorted(paid_provider) == [1, 2, 3, 4, 5, 6]


def test_dry_run_makes_no_calls_and_books_nothing(experiment_with_rows, paid_provider, monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_MAX_SPEND", _CAP)
    pid, exp_subdir = experiment_with_rows

    result = runner.invoke(
        app,
        ["--dry-run", "experiment", "run", "--pid", pid, "--file", _EXPERIMENT_DB_NAME],
    )

    assert result.exit_code == 0, result.output
    assert paid_provider == []
    assert read_cost_logs(find_db(exp_subdir)) == []
    df = read_experiment_df(find_db(exp_subdir))
    assert df["status"].isna().all()
    assert not pd.notna(df["response_text"]).any()
