"""Tests for the copied `agreement` module: yes/no detection and Cohen's kappa."""

import math

import pandas as pd
import pytest

from llmexer.base.analysis.agreement import (
    RESULT_COLUMNS,
    MissingFieldError,
    item_keys,
    kappa_matrix,
    pairwise_cohen_kappa,
    yes_no_fields,
)

MODEL = "llama3.3:latest"


def _rows(profile, answers, *, model=MODEL, provider="ollama", field="relevant"):
    """One row per answer, coded the way `experiment generate` codes them."""

    return [
        {
            "code": f"D{index:02d}_prompt01_{model}_{profile}",
            "model_name": model,
            "provider_name": provider,
            "profile_name": profile,
            field: value,
        }
        for index, value in enumerate(answers, start=1)
    ]


def _frame(*row_groups):
    return pd.DataFrame([row for group in row_groups for row in group])


# ------------------------------------------------------------------ item keys


def test_item_key_strips_the_rater_tail():
    df = _frame(_rows("ollama-default", ["yes"]))

    assert list(item_keys(df)) == ["D01_prompt01"]


def test_item_key_survives_an_underscore_in_the_data_id():
    """The tail is removed by name, so splitting on '_' cannot go wrong."""

    df = pd.DataFrame(
        [
            {
                "code": f"my_data_01_prompt01_{MODEL}_ollama-default",
                "model_name": MODEL,
                "provider_name": "ollama",
                "profile_name": "ollama-default",
                "relevant": "yes",
            }
        ]
    )

    assert list(item_keys(df)) == ["my_data_01_prompt01"]


def test_item_key_keeps_the_prompt_apart():
    """Two prompts about one paper are two items, not one."""

    df = pd.DataFrame(
        [
            {
                "code": f"D01_{prompt}_{MODEL}_ollama-default",
                "model_name": MODEL,
                "provider_name": "ollama",
                "profile_name": "ollama-default",
                "relevant": "yes",
            }
            for prompt in ("prompt01", "prompt02")
        ]
    )

    assert sorted(item_keys(df)) == ["D01_prompt01", "D01_prompt02"]


# -------------------------------------------------------------- yes/no fields


def test_yes_no_fields_accepts_every_spelling():
    df = pd.DataFrame(
        {
            "code": ["D01_prompt01_m_p", "D02_prompt01_m_p"],
            "model_name": ["m", "m"],
            "provider_name": ["ollama", "ollama"],
            "profile_name": ["p", "p"],
            "plain": ["yes", "no"],
            "mixed_case": ["Yes", "NO"],
            "boolean": ["TRUE", "false"],
            "short": ["y", "n"],
            "binary": ["1", "0"],
        }
    )

    assert yes_no_fields(df) == ["plain", "mixed_case", "boolean", "short", "binary"]


def test_yes_no_fields_rejects_ratings_and_prose_and_empties():
    df = pd.DataFrame(
        {
            "code": ["D01_prompt01_m_p", "D02_prompt01_m_p"],
            "model_name": ["m", "m"],
            "provider_name": ["ollama", "ollama"],
            "profile_name": ["p", "p"],
            "rating": [3, 5],
            "reason": ["because of the abstract", "off topic"],
            "never_answered": [None, None],
        }
    )

    assert yes_no_fields(df) == []


def test_yes_no_fields_skips_the_identity_columns():
    df = _frame(_rows("ollama-default", ["yes", "no"]))

    assert "profile_name" not in yes_no_fields(df)
    assert yes_no_fields(df) == ["relevant"]


# ---------------------------------------------------------------------- kappa


def test_perfect_agreement_is_one():
    df = _frame(
        _rows("ollama-default", ["yes", "no", "yes", "no"]),
        _rows("ollama-hot", ["yes", "no", "yes", "no"]),
    )

    result = pairwise_cohen_kappa(df, ["relevant"])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["items"] == 4
    assert row["counter_same_answer"] == 4
    assert row["counter_different_answer"] == 0
    assert row["cohen_kappa"] == pytest.approx(1.0)


def test_complete_disagreement_on_balanced_marginals_is_minus_one():
    df = _frame(
        _rows("ollama-default", ["yes", "no", "yes", "no"]),
        _rows("ollama-hot", ["no", "yes", "no", "yes"]),
    )

    row = pairwise_cohen_kappa(df, ["relevant"]).iloc[0]

    assert row["counter_same_answer"] == 0
    assert row["counter_different_answer"] == 4
    assert row["cohen_kappa"] == pytest.approx(-1.0)


def test_a_hand_computed_two_by_two():
    """20 items, 16 matches: p_o = 0.8, p_e = 0.5, kappa = 0.6."""

    a = ["yes"] * 10 + ["no"] * 10
    b = ["yes"] * 8 + ["no"] * 2 + ["no"] * 8 + ["yes"] * 2
    df = _frame(_rows("ollama-default", a), _rows("ollama-hot", b))

    row = pairwise_cohen_kappa(df, ["relevant"]).iloc[0]

    assert row["items"] == 20
    assert row["counter_same_answer"] == 16
    assert row["counter_different_answer"] == 4
    assert row["cohen_kappa"] == pytest.approx(0.6)


def test_items_counts_only_what_both_answered():
    df = _frame(
        _rows("ollama-default", ["yes", "no", "yes"]),
        _rows("ollama-hot", ["yes", "no"]),
    )

    row = pairwise_cohen_kappa(df, ["relevant"]).iloc[0]

    assert row["items"] == 2
    assert row["counter_same_answer"] + row["counter_different_answer"] == 2


def test_one_shared_label_leaves_kappa_undefined():
    """Both always answering 'no' is total agreement and no information."""

    df = _frame(
        _rows("ollama-default", ["no", "no", "no"]),
        _rows("ollama-hot", ["no", "no", "no"]),
    )

    row = pairwise_cohen_kappa(df, ["relevant"]).iloc[0]

    assert row["counter_same_answer"] == 3
    assert row["counter_different_answer"] == 0
    assert math.isnan(row["cohen_kappa"])


def test_no_shared_items_is_reported_rather_than_scored():
    default = _rows("ollama-default", ["yes", "no"])
    hot = _rows("ollama-hot", ["yes", "no"])
    for index, row in enumerate(hot):
        row["code"] = f"D{index + 10:02d}_prompt01_{MODEL}_ollama-hot"
    df = _frame(default, hot)

    row = pairwise_cohen_kappa(df, ["relevant"]).iloc[0]

    assert row["items"] == 0
    assert row["counter_same_answer"] == 0
    assert row["counter_different_answer"] == 0
    assert math.isnan(row["cohen_kappa"])


def test_one_configuration_gives_an_empty_table_with_the_full_columns():
    df = _frame(_rows("ollama-default", ["yes", "no"]))

    result = pairwise_cohen_kappa(df, ["relevant"])

    assert result.empty
    assert list(result.columns) == list(RESULT_COLUMNS)


def test_empty_frame_gives_an_empty_table():
    result = pairwise_cohen_kappa(pd.DataFrame(), [])

    assert result.empty
    assert list(result.columns) == list(RESULT_COLUMNS)


def test_a_model_served_by_two_providers_is_two_raters():
    """profile_name alone would merge these two; the triple keeps them apart."""

    df = _frame(
        _rows("shared-profile", ["yes", "no"], provider="ollama"),
        _rows("shared-profile", ["yes", "no"], provider="litellm"),
    )

    result = pairwise_cohen_kappa(df, ["relevant"])

    assert len(result) == 1
    assert {result.iloc[0]["provider_a"], result.iloc[0]["provider_b"]} == {"ollama", "litellm"}


def test_three_configurations_give_three_pairs_per_field():
    df = _frame(
        _rows("a", ["yes", "no"]),
        _rows("b", ["yes", "no"]),
        _rows("c", ["no", "yes"]),
    )

    assert len(pairwise_cohen_kappa(df, ["relevant"])) == 3


def test_only_the_named_fields_are_compared():
    df = _frame(
        _rows("ollama-default", ["yes", "no"]),
        _rows("ollama-hot", ["yes", "no"]),
    )
    df["second"] = ["y", "n", "y", "n"]

    assert sorted(pairwise_cohen_kappa(df, yes_no_fields(df))["field"]) == ["relevant", "second"]
    assert list(pairwise_cohen_kappa(df, ["second"])["field"]) == ["second"]


def test_an_unknown_field_is_rejected():
    """A typo must not silently produce an empty table."""

    df = _frame(
        _rows("ollama-default", ["yes", "no"]),
        _rows("ollama-hot", ["yes", "no"]),
    )

    with pytest.raises(MissingFieldError) as excinfo:
        pairwise_cohen_kappa(df, ["relevant", "nope"])

    assert "nope" in str(excinfo.value)


def test_an_unknown_field_is_catchable_as_key_error():
    df = _frame(_rows("ollama-default", ["yes"]))

    with pytest.raises(KeyError):
        pairwise_cohen_kappa(df, ["nope"])


def test_no_fields_gives_an_empty_table():
    df = _frame(
        _rows("ollama-default", ["yes", "no"]),
        _rows("ollama-hot", ["yes", "no"]),
    )

    result = pairwise_cohen_kappa(df, [])

    assert result.empty
    assert list(result.columns) == list(RESULT_COLUMNS)


def test_case_and_whitespace_do_not_count_as_disagreement():
    df = _frame(
        _rows("ollama-default", ["Yes", " no "]),
        _rows("ollama-hot", ["yes", "NO"]),
    )

    assert pairwise_cohen_kappa(df, ["relevant"]).iloc[0]["counter_different_answer"] == 0


def test_kappa_matrix_is_symmetric():
    df = _frame(
        _rows("ollama-default", ["yes", "no", "yes", "no"]),
        _rows("ollama-hot", ["yes", "no", "yes", "no"]),
    )
    result = pairwise_cohen_kappa(df, ["relevant"])

    matrix = kappa_matrix(result, "relevant")

    assert matrix.shape == (2, 2)
    left, right = matrix.index
    assert matrix.loc[left, right] == matrix.loc[right, left] == pytest.approx(1.0)


def test_kappa_matrix_of_an_unknown_field_is_empty():
    df = _frame(_rows("a", ["yes"]), _rows("b", ["yes"]))

    assert kappa_matrix(pairwise_cohen_kappa(df, ["relevant"]), "nope").empty
