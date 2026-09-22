"""Unit tests for the per-run spending budget."""

import threading
from unittest.mock import MagicMock

import pytest

from llmexer.base.budget import SessionBudget, as_cost, resolve_max_spend
from llmexer.constants import DEFAULT_OPENROUTER_MAX_SPEND_USD

_ENV_VAR = "PROVIDER_OPENROUTER_MAX_SPEND"


# ---------------------------------------------------------------------------
# resolve_max_spend
# ---------------------------------------------------------------------------


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert resolve_max_spend() == DEFAULT_OPENROUTER_MAX_SPEND_USD


def test_default_is_five_dollars():
    assert DEFAULT_OPENROUTER_MAX_SPEND_USD == 5.0


def test_env_var_overrides_the_default(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "12.50")
    assert resolve_max_spend() == 12.50


@pytest.mark.parametrize("value", ["abc", "", "0", "-1", "1,50"])
def test_unusable_values_fall_back_to_the_default(monkeypatch, value):
    """A typo must not silently turn the cap off, nor raise mid-run."""
    monkeypatch.setenv(_ENV_VAR, value)
    assert resolve_max_spend() == DEFAULT_OPENROUTER_MAX_SPEND_USD


def test_env_is_read_at_call_time_not_import_time(monkeypatch):
    """The CLI loads .env after llmexer.base is imported, so the read must be late."""
    monkeypatch.setenv(_ENV_VAR, "1.00")
    assert resolve_max_spend() == 1.00
    monkeypatch.setenv(_ENV_VAR, "2.00")
    assert resolve_max_spend() == 2.00


# ---------------------------------------------------------------------------
# as_cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 0.0, 1, 0.0031])
def test_as_cost_accepts_numbers(value):
    assert as_cost(value) == float(value)


@pytest.mark.parametrize("value", [None, "0.05", MagicMock(), object(), [], -0.5])
def test_as_cost_rejects_everything_else(value):
    """No float() coercion: float(MagicMock()) is 1.0 and would book a dollar."""
    assert as_cost(value) is None


@pytest.mark.parametrize("value", [True, False])
def test_as_cost_rejects_booleans(value):
    """bool subclasses int, so True would otherwise read as one dollar."""
    assert as_cost(value) is None


# ---------------------------------------------------------------------------
# SessionBudget
# ---------------------------------------------------------------------------


def test_a_fresh_budget_is_not_exhausted():
    assert SessionBudget(limit_usd=5.0).is_exhausted() is False


def test_add_returns_the_running_total():
    budget = SessionBudget(limit_usd=5.0)
    assert budget.add(1.5) == 1.5
    assert budget.add(2.0) == 3.5
    assert budget.spent_usd == 3.5


def test_below_the_cap_is_not_exhausted():
    budget = SessionBudget(limit_usd=5.0)
    budget.add(4.99)
    assert budget.is_exhausted() is False


def test_exactly_at_the_cap_is_exhausted():
    budget = SessionBudget(limit_usd=5.0)
    budget.add(5.0)
    assert budget.is_exhausted() is True


def test_over_the_cap_is_exhausted():
    budget = SessionBudget(limit_usd=5.0)
    budget.add(5.01)
    assert budget.is_exhausted() is True


def test_concurrent_adds_do_not_lose_updates():
    """`+=` is read-modify-write, so the parallel run path needs the lock."""
    budget = SessionBudget(limit_usd=10_000.0)
    barrier = threading.Barrier(8)

    def book():
        barrier.wait()
        for _ in range(500):
            budget.add(0.01)

    threads = [threading.Thread(target=book) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert budget.spent_usd == pytest.approx(8 * 500 * 0.01)


def test_two_budgets_are_independent():
    """One object per run is what makes the tally reset when the command exits."""
    first = SessionBudget(limit_usd=5.0)
    first.add(5.0)
    assert SessionBudget(limit_usd=5.0).is_exhausted() is False
