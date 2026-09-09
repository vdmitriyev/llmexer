"""Tests for the copied `stats` analysis module."""

import pandas as pd
import pytest

from llmexer.base.analysis import transform
from llmexer.base.analysis.stats import (
    GROUP_COLUMNS,
    IDENTITY_COLUMNS,
    ExperimentSummary,
    answers_by_model,
    failed_calls,
    response_time_stats,
    summary,
    token_stats,
    tokens_series,
    total_requests,
    total_responses,
    value_counts,
    value_counts_summary,
)
from llmexer.base.analysis.transform import load_experiment_db
from llmexer.base.dao import ExperimentDAO
from tests.db_helpers import LITELLM_ROW, OLLAMA_ROW, seed_db


def _row(base, **overrides):
    return dict(base, **overrides)


@pytest.fixture()
def frame():
    """Two providers covering every state a run can leave behind.

    The truncated row is the interesting one: `experiment run` stores a reply cut
    at `max_tokens` as ``status="success"``, so anything counting successes as
    answers overstates the response rate.
    """

    return pd.DataFrame(
        [
            dict(
                provider_name="ollama",
                model_name="m1",
                profile_name="p1",
                status="success",
                state="finished",
                elapsed_seconds=2.0,
                total_tokens=100,
                prompt_tokens=80,
                completion_tokens=20,
            ),
            dict(
                provider_name="ollama",
                model_name="m1",
                profile_name="p1",
                status="success",
                state="maxtokenreached",
                elapsed_seconds=4.0,
                total_tokens=50,
                prompt_tokens=None,
                completion_tokens=None,
            ),
            dict(
                provider_name="litellm",
                model_name="m1",
                profile_name="p1",
                status="Error: boom",
                state="error",
                elapsed_seconds=None,
                total_tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
            ),
            dict(
                provider_name="litellm",
                model_name="m2",
                profile_name="p2",
                status=None,
                state=None,
                elapsed_seconds=None,
                total_tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_total_requests_counts_every_generated_row(frame):
    assert total_requests(frame) == 4


def test_total_responses_splits_by_provider_and_model(frame):
    """A model served by two providers stays two rows, as `experiment stats` does."""

    out = total_responses(frame).set_index(["provider_name", "model_name"])

    assert len(out) == 3
    assert out.loc[("ollama", "m1"), "requests"] == 2


def test_total_responses_excludes_a_truncated_reply(frame):
    """`maxtokenreached` is stored as a success but carries no answer."""

    out = total_responses(frame).set_index(["provider_name", "model_name"])

    assert out.loc[("ollama", "m1"), "finished"] == 2
    assert out.loc[("ollama", "m1"), "truncated"] == 1
    assert out.loc[("ollama", "m1"), "responses"] == 1


def test_total_responses_counts_open_rows(frame):
    out = total_responses(frame).set_index(["provider_name", "model_name"])

    assert out.loc[("litellm", "m2"), "open"] == 1
    assert out.loc[("litellm", "m1"), "errors"] == 1


def test_total_responses_on_an_empty_frame_keeps_its_columns():
    out = total_responses(pd.DataFrame())

    assert out.empty
    assert "responses" in out.columns


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_response_time_stats_ignores_unrun_rows(frame):
    """Unrun rows have no elapsed time and would drag every mean to zero."""

    out = response_time_stats(frame)

    assert len(out) == 1
    assert out.iloc[0]["count"] == 2
    assert out.iloc[0]["mean"] == 3.0
    assert out.iloc[0]["min"] == 2.0
    assert out.iloc[0]["max"] == 4.0


def test_response_time_stats_without_grouping(frame):
    out = response_time_stats(frame, by=())

    assert len(out) == 1
    assert out.iloc[0]["total"] == 6.0


def test_response_time_stats_on_an_empty_frame():
    out = response_time_stats(pd.DataFrame())

    assert out.empty
    assert "median" in out.columns


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_series_zero_fills_a_missing_total(frame):
    """A row that was never run, or whose provider reported nothing, counts as 0."""

    assert list(tokens_series(frame)) == [100, 50, 0, 0]


def test_token_stats_totals_over_finished_rows(frame):
    out = token_stats(frame)

    assert len(out) == 1
    assert out.iloc[0]["tokens_total"] == 150


def test_token_stats_reports_coverage_of_the_prompt_completion_split(frame):
    """Only one of the two finished ollama rows carried a usage breakdown."""

    out = token_stats(frame)

    assert out.iloc[0]["coverage"] == 0.5
    assert out.iloc[0]["prompt_tokens_total"] == 80
    assert out.iloc[0]["completion_tokens_total"] == 20


def test_token_stats_omits_the_split_when_the_frame_has_none(frame):
    out = token_stats(frame.drop(columns=["prompt_tokens", "completion_tokens"]))

    assert "coverage" not in out.columns
    assert out.iloc[0]["tokens_total"] == 150


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_failed_calls_rate_is_over_attempted_rows(frame):
    """An unrun row must not make the failure rate look better than it is."""

    out = failed_calls(frame).set_index(["provider_name", "model_name"])

    assert out.loc[("litellm", "m1"), "failure_rate"] == 1.0
    assert out.loc[("litellm", "m2"), "attempted"] == 0
    assert out.loc[("litellm", "m2"), "failure_rate"] == 0.0


def test_failed_calls_appends_an_overall_row(frame):
    out = failed_calls(frame)
    overall = out[out["provider_name"] == "all"].iloc[0]

    assert overall["requests"] == 4
    assert overall["attempted"] == 3
    assert overall["failed"] == 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_aggregates_every_state(frame):
    result = summary(frame)

    assert result.requests == 4
    assert result.finished == 2
    assert result.truncated == 1
    assert result.responses == 1
    assert result.errors == 1
    assert result.open == 1
    assert result.providers == 2
    assert result.models == 2
    assert result.total_tokens == 150


def test_summary_failure_rate_excludes_unrun_rows(frame):
    assert summary(frame).failure_rate == pytest.approx(1 / 3)


def test_summary_of_an_empty_frame_is_all_zero():
    result = summary(pd.DataFrame())

    assert result == ExperimentSummary()
    assert result.requests == 0


def test_summary_to_frame_and_str(frame):
    result = summary(frame)

    assert list(result.to_frame().columns) == ["metric", "value"]
    assert "requests" in str(result)
    assert result.to_dict()["requests"] == 4


# ---------------------------------------------------------------------------
# Parity with the DAO, so the notebook and `experiment stats` cannot disagree
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_db(tmp_path):
    """A database mixing finished, truncated, errored and unrun rows."""

    path = tmp_path / "experiment_20240101_01.db"
    seed_db(
        path,
        {
            "ollama": [
                _row(OLLAMA_ROW, ID=1, status="success", state="finished", total_tokens=100, elapsed_seconds=2.0),
                _row(
                    OLLAMA_ROW,
                    ID=2,
                    code="D02_prompt01_llama3.3:latest_ollama-default",
                    status="success",
                    state="maxtokenreached",
                    total_tokens=50,
                    elapsed_seconds=4.0,
                ),
            ],
            # Every row of one provider must carry the same keys: they are
            # inserted with a single executemany, which rejects a ragged batch.
            "litellm": [
                _row(LITELLM_ROW, ID=3, status="Error: boom", state="error", total_tokens=None, elapsed_seconds=None),
                _row(
                    LITELLM_ROW,
                    ID=4,
                    code="D04_prompt01_gpt-oss:120b_litellm-default",
                    status=None,
                    state=None,
                    total_tokens=None,
                    elapsed_seconds=None,
                ),
            ],
        },
    )
    return path


def test_summary_matches_dao_stats(seeded_db):
    """The headline numbers must be identical to `llmexer experiment stats`."""

    with ExperimentDAO(str(seeded_db)) as dao:
        dao_stats = dao.stats()
    result = summary(load_experiment_db(seeded_db))

    assert result.requests == dao_stats["total"]
    assert result.finished == dao_stats["finished"]
    assert result.errors == dao_stats["errors"]
    assert result.running == dao_stats["running"]
    assert result.total_tokens == dao_stats["total_tokens"]


def test_total_responses_matches_dao_per_model_requests(seeded_db):
    with ExperimentDAO(str(seeded_db)) as dao:
        dao_models = {(m["model_name"], m["provider"]): m for m in dao.stats()["models"]}
    out = total_responses(load_experiment_db(seeded_db))

    for _, row in out.iterrows():
        key = (row["model_name"], row["provider_name"])
        assert row["requests"] == dao_models[key]["requests"]
        assert row["finished"] == dao_models[key]["finished"]


# ---------------------------------------------------------------------------
# value_counts_summary / value_counts — the parsed answers themselves
# ---------------------------------------------------------------------------


@pytest.fixture()
def answers():
    """A flattened-answer frame covering every kind the classifier knows."""

    return pd.DataFrame(
        {
            "verdict": ["yes", "no", "yes", None],
            "relevant": ["true", "false", "true", "true"],
            "binary": ["1", "0", "1", "1"],
            "rating": [1, 2, 3, 3],
            "topic": ["ml", "nlp", "ml", "cv"],
            "reason": ["a long free-text justification that reads as prose " * 2] * 4,
            "tags": [["a", "b"]] * 4,
            "blank": [None] * 4,
        }
    )


@pytest.mark.parametrize(
    "column,kind",
    [
        ("verdict", "boolean"),
        ("relevant", "boolean"),
        ("binary", "boolean"),
        ("rating", "countable"),
        ("topic", "categorical"),
        ("reason", "text"),
        ("tags", "nested"),
        ("blank", "empty"),
    ],
)
def test_value_counts_summary_classifies_each_answer_kind(answers, column, kind):
    out = value_counts_summary(answers).set_index("column")

    assert out.loc[column, "kind"] == kind


def test_only_yes_no_countable_and_small_sets_are_plottable(answers):
    """A bar chart of free text or of a nested list says nothing."""

    out = value_counts_summary(answers).set_index("column")

    assert list(out.loc[["verdict", "relevant", "binary", "rating", "topic"], "plottable"]) == [True] * 5
    assert list(out.loc[["reason", "tags", "blank"], "plottable"]) == [False] * 3


def test_value_counts_summary_counts_values_and_gaps(answers):
    out = value_counts_summary(answers).set_index("column")

    assert out.loc["verdict", "values"] == 3
    assert out.loc["verdict", "missing"] == 1
    assert out.loc["verdict", "unique"] == 2
    assert out.loc["verdict", "top"] == "yes"
    assert out.loc["verdict", "top_count"] == 2


def test_value_counts_summary_can_be_limited_to_some_columns(answers):
    out = value_counts_summary(answers, columns=["verdict", "nope"])

    assert list(out["column"]) == ["verdict"]


def test_value_counts_summary_on_an_empty_frame():
    out = value_counts_summary(pd.DataFrame())

    assert out.empty
    assert "plottable" in out.columns


def test_a_high_cardinality_label_column_is_not_plottable():
    """Beyond a dozen distinct values the chart is a barcode."""

    df = pd.DataFrame({"label": [f"v{i}" for i in range(40)]})

    assert not value_counts_summary(df).iloc[0]["plottable"]


def test_a_wide_ranging_number_is_numeric_not_countable():
    df = pd.DataFrame({"score": list(range(100))})
    out = value_counts_summary(df).set_index("column")

    assert out.loc["score", "kind"] == "numeric"
    assert not out.loc["score", "plottable"]


def test_value_counts_returns_counts_and_shares(answers):
    out = value_counts(answers, "verdict").set_index("verdict")

    assert out.loc["yes", "count"] == 2
    assert out.loc["yes", "share"] == pytest.approx(2 / 3)


def test_value_counts_on_a_missing_column(answers):
    with pytest.raises(KeyError):
        value_counts(answers, "nope")


# ---------------------------------------------------------------------------
# identity columns — mirrored across the copied modules
# ---------------------------------------------------------------------------


def test_identity_columns_match_the_transform_module():
    """The copied modules may not import each other, so the tuple is duplicated.

    This is the guard that keeps the two in step - the same arrangement as
    `strip_code_fence`.
    """

    assert IDENTITY_COLUMNS == transform.IDENTITY_COLUMNS


def test_every_grouping_column_is_an_identity_column():
    """`answers_by_model` groups on columns `flattened_only` must have exported."""

    assert set(GROUP_COLUMNS) <= set(IDENTITY_COLUMNS)


@pytest.fixture()
def identified_answers(answers):
    """Flattened answers as `flattened_only` now returns them: identity first."""

    identity = pd.DataFrame(
        {
            "code": [f"D0{i}_P1_m1_default" for i in range(len(answers))],
            "model_name": ["m1", "m1", "gpt", "gpt"],
            "provider_name": ["ollama", "ollama", "openai", "openai"],
            "profile_name": ["default"] * len(answers),
        }
    )

    return pd.concat([identity, answers], axis=1)


def test_value_counts_summary_skips_the_identity_columns(identified_answers):
    """Identity is not an answer, and `model_name` would be a plottable column.

    The notebook picks its chart column as the first plottable one, so without
    this the whole answer profile would describe the identity instead.
    """

    profiled = value_counts_summary(identified_answers)["column"].tolist()

    for name in IDENTITY_COLUMNS:
        assert name not in profiled
    assert "verdict" in profiled


def test_value_counts_summary_profiles_an_identity_column_when_asked(identified_answers):
    out = value_counts_summary(identified_answers, columns=["model_name"])

    assert out["column"].tolist() == ["model_name"]


# ---------------------------------------------------------------------------
# answers_by_model
# ---------------------------------------------------------------------------


@pytest.fixture()
def judged():
    """The same model on two providers, plus a row that was never answered."""

    return pd.DataFrame(
        {
            "provider_name": ["ollama", "ollama", "openai", "openai", "openai"],
            "model_name": ["m1", "m1", "m1", "gpt", "gpt"],
            "verdict": ["yes", "no", "yes", "yes", None],
            "rating": [1, 10, 2, 3, 3],
        }
    )


def test_answers_by_model_is_one_row_per_provider_and_model(judged):
    out = answers_by_model(judged, "verdict")

    assert list(zip(out["provider_name"], out["model_name"])) == [
        ("ollama", "m1"),
        ("openai", "gpt"),
        ("openai", "m1"),
    ]
    assert list(out.columns) == ["provider_name", "model_name", "no", "yes"]


def test_answers_by_model_keeps_one_model_on_two_providers_apart(judged):
    """Merging them would report a row belonging to neither setup."""

    out = answers_by_model(judged, "verdict").set_index(["provider_name", "model_name"])

    assert out.loc[("ollama", "m1"), "yes"] == 1
    assert out.loc[("openai", "m1"), "yes"] == 1


def test_answers_by_model_invents_no_combination_that_never_ran(judged):
    """`ollama`/`gpt` is not a setup, so it is not a row."""

    out = answers_by_model(judged, "verdict")

    assert ("ollama", "gpt") not in set(zip(out["provider_name"], out["model_name"]))


def test_answers_by_model_leaves_out_the_rows_with_no_answer(judged):
    out = answers_by_model(judged, "verdict")
    counted = int(out[["no", "yes"]].to_numpy().sum())

    assert counted == int(judged["verdict"].notna().sum()) == 4


def test_answers_by_model_orders_a_rating_by_its_own_scale(judged):
    """Read as text a 1-10 scale sorts 1, 10, 2 - which no reader expects."""

    out = answers_by_model(judged, "rating")

    assert [name for name in out.columns if name not in GROUP_COLUMNS] == ["1", "2", "3", "10"]


def test_answers_by_model_honours_a_grouping_override(judged):
    out = answers_by_model(judged, "verdict", by="model_name")

    assert list(out.columns) == ["model_name", "no", "yes"]
    assert out["model_name"].tolist() == ["gpt", "m1"]


def test_answers_by_model_renames_an_answer_spelled_like_a_group_column():
    """Otherwise moving the index back into columns refuses to insert it."""

    df = pd.DataFrame({"provider_name": ["a"], "model_name": ["m"], "verdict": ["model_name"]})
    out = answers_by_model(df, "verdict")

    assert "model_name_answer" in out.columns
    assert out.loc[0, "model_name_answer"] == 1


def test_answers_by_model_on_a_missing_column(judged):
    with pytest.raises(KeyError):
        answers_by_model(judged, "nope")


def test_answers_by_model_on_an_empty_frame():
    out = answers_by_model(pd.DataFrame({"verdict": pd.Series(dtype="object")}), "verdict")

    assert list(out.columns) == list(GROUP_COLUMNS)
    assert out.empty


def test_answers_by_model_when_nothing_was_answered():
    df = pd.DataFrame({"provider_name": ["a"], "model_name": ["m"], "verdict": [None]})
    out = answers_by_model(df, "verdict")

    assert list(out.columns) == list(GROUP_COLUMNS)
    assert out.empty
