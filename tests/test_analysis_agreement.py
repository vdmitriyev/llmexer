"""Tests for the copied `agreement` module: yes/no detection and Cohen's kappa."""

import math

import pandas as pd
import pytest

from llmexer.base.analysis.agreement import (
    INTERMEDIATE_VALUES_COLUMNS,
    RESULT_COLUMNS,
    AmbiguousProfileError,
    MissingColumnError,
    MissingFieldError,
    kappa_matrix,
    paired_answers,
    pairwise_cohen_kappa,
    save_name,
    yes_no_fields,
)

MODEL = "llama3.3:latest"


def _rows(profile, answers, *, model=MODEL, provider="ollama", field="relevant", prompt="prompt01", start=1):
    """One row per answer, shaped the way `experiment generate` writes them.

    `start` shifts the data IDs, which is how a test makes two profiles answer
    items that do not overlap.
    """

    return [
        {
            "code": f"D{index:02d}_{prompt}_{model}_{profile}",
            "data_id": f"D{index:02d}",
            "prompt_id": prompt,
            "model_name": model,
            "provider_name": provider,
            "profile_name": profile,
            field: value,
        }
        for index, value in enumerate(answers, start=start)
    ]


def _frame(*row_groups):
    return pd.DataFrame([row for group in row_groups for row in group])


# ------------------------------------------------------- grouping by prompt


def test_each_prompt_is_scored_on_its_own():
    """Two prompts give two rows per pair and field, not one pooled number."""
    df = _frame(
        _rows("ollama-default", ["yes", "no"], prompt="prompt01"),
        _rows("ollama-hot", ["yes", "no"], prompt="prompt01"),
        _rows("ollama-default", ["yes", "no"], prompt="prompt02"),
        _rows("ollama-hot", ["yes", "yes"], prompt="prompt02"),
    )

    result = pairwise_cohen_kappa(df, ["relevant"])

    assert list(result["prompt_id"]) == ["prompt01", "prompt02"]
    assert list(result["counter_different_answer"]) == [0, 1]


def test_items_are_matched_on_data_id_within_the_prompt():
    """The same data_id under two prompts is two items, one per prompt."""
    df = _frame(
        _rows("ollama-default", ["yes"], prompt="prompt01"),
        _rows("ollama-hot", ["yes"], prompt="prompt01"),
        _rows("ollama-default", ["no"], prompt="prompt02"),
        _rows("ollama-hot", ["no"], prompt="prompt02"),
    )

    result = pairwise_cohen_kappa(df, ["relevant"])

    assert list(result["items"]) == [1, 1]


def test_a_profile_absent_from_a_prompt_has_no_pair_there():
    """Profiles are taken per prompt, so a prompt with one profile scores nothing."""
    df = _frame(
        _rows("ollama-default", ["yes", "no"], prompt="prompt01"),
        _rows("ollama-hot", ["yes", "no"], prompt="prompt01"),
        _rows("ollama-default", ["yes", "no"], prompt="prompt02"),
    )

    result = pairwise_cohen_kappa(df, ["relevant"])

    assert list(result["prompt_id"]) == ["prompt01"]


def test_a_frame_without_the_id_columns_is_rejected():
    """A pre-0.4.6 export cannot say which item an answer is about."""
    df = _frame(_rows("ollama-default", ["yes"]), _rows("ollama-hot", ["yes"]))
    df = df.drop(columns=["data_id", "prompt_id"])

    with pytest.raises(MissingColumnError) as excinfo:
        pairwise_cohen_kappa(df, ["relevant"])

    assert "data_id" in str(excinfo.value)
    assert "prompt_id" in str(excinfo.value)


def test_the_id_columns_are_not_mistaken_for_answers():
    """A data_id of 0/1 would otherwise read as a yes/no vocabulary."""
    df = _frame(_rows("ollama-default", ["yes", "no"]))
    df["data_id"] = ["1", "0"]

    assert yes_no_fields(df) == ["relevant"]


# -------------------------------------------------------------- yes/no fields


def test_yes_no_fields_accepts_every_spelling():
    df = pd.DataFrame(
        {
            "code": ["D01_prompt01_m_p", "D02_prompt01_m_p"],
            "data_id": ["D01", "D02"],
            "prompt_id": ["prompt01", "prompt01"],
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
            "data_id": ["D01", "D02"],
            "prompt_id": ["prompt01", "prompt01"],
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
    # `start` puts the second profile on D11/D12, so the two share no data_id.
    hot = _rows("ollama-hot", ["yes", "no"], start=11)
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


def test_one_profile_on_two_models_is_rejected():
    """Pairing is on the profile, so a name covering two models has no identity."""
    df = _frame(
        _rows("shared", ["yes", "no"], model="gemma4:31b"),
        _rows("shared", ["yes", "no"], model="phi4:14b"),
    )

    with pytest.raises(AmbiguousProfileError) as excinfo:
        pairwise_cohen_kappa(df, ["relevant"])

    assert "shared" in str(excinfo.value)
    assert "gemma4:31b" in str(excinfo.value)


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

    matrix = kappa_matrix(result, "relevant", "prompt01")

    assert matrix.shape == (2, 2)
    left, right = matrix.index
    assert matrix.loc[left, right] == matrix.loc[right, left] == pytest.approx(1.0)


def test_kappa_matrix_of_an_unknown_field_is_empty():
    df = _frame(_rows("a", ["yes"]), _rows("b", ["yes"]))

    assert kappa_matrix(pairwise_cohen_kappa(df, ["relevant"]), "nope", "prompt01").empty


def test_kappa_matrix_of_an_unknown_prompt_is_empty():
    """A prompt nothing was scored under pivots to nothing, not to a stale matrix."""
    df = _frame(_rows("a", ["yes"]), _rows("b", ["yes"]))

    assert kappa_matrix(pairwise_cohen_kappa(df, ["relevant"]), "relevant", "prompt99").empty


# ------------------------------------------------- the intermediate values


def _written(directory):
    """Every intermediate-values CSV under `directory`, read back, by filename."""
    return {path.name: pd.read_csv(path, sep=";") for path in sorted(directory.glob("*.csv"))}


def test_nothing_is_written_without_a_directory(tmp_path):
    """The option is off by default; the table alone comes back."""
    df = _frame(_rows("ollama-default", ["yes", "no"]), _rows("ollama-hot", ["yes", "no"]))

    pairwise_cohen_kappa(df, ["relevant"])

    assert list(tmp_path.iterdir()) == []


def test_one_file_per_prompt_field_and_pair(tmp_path):
    """The name carries all four parts, so each comparison gets its own file."""
    df = _frame(
        _rows("ollama-default", ["yes", "no"], prompt="prompt01"),
        _rows("ollama-hot", ["yes", "no"], prompt="prompt01"),
        _rows("ollama-default", ["yes", "no"], prompt="prompt02"),
        _rows("ollama-hot", ["yes", "no"], prompt="prompt02"),
    )

    pairwise_cohen_kappa(df, ["relevant"], intermediate_values_dir=tmp_path)

    names = sorted(_written(tmp_path))
    assert names == [
        "prompt01_relevant_ollama-default_ollama-hot.csv",
        "prompt02_relevant_ollama-default_ollama-hot.csv",
    ]


def test_each_field_gets_its_own_file(tmp_path):
    """Two fields on one pair are two files, named apart by the field."""
    df = _frame(_rows("ollama-default", ["yes", "no"]), _rows("ollama-hot", ["yes", "no"]))
    df["second"] = ["y", "n", "y", "n"]

    pairwise_cohen_kappa(df, ["relevant", "second"], intermediate_values_dir=tmp_path)

    written = _written(tmp_path)
    assert sorted(written) == [
        "prompt01_relevant_ollama-default_ollama-hot.csv",
        "prompt01_second_ollama-default_ollama-hot.csv",
    ]
    for frame in written.values():
        assert list(frame.columns) == list(INTERMEDIATE_VALUES_COLUMNS)
        assert len(frame) == 2


def test_the_written_rows_are_the_rows_that_were_scored(tmp_path):
    """`same` sums back to the counts the table reports - the point of the file."""
    df = _frame(
        _rows("ollama-default", ["yes", "no", "yes"]),
        _rows("ollama-hot", ["yes", "no", "no"]),
    )

    result = pairwise_cohen_kappa(df, ["relevant"], intermediate_values_dir=tmp_path)

    written = list(_written(tmp_path).values())[0]
    row = result.iloc[0]
    assert len(written) == row["items"]
    assert int(written["same"].sum()) == row["counter_same_answer"]
    assert int((~written["same"]).sum()) == row["counter_different_answer"]
    assert list(written["data_id"]) == ["D01", "D02", "D03"]


def test_a_pair_with_no_shared_item_writes_no_file(tmp_path):
    """There is nothing to check by hand, so there is no file to open."""
    df = _frame(
        _rows("ollama-default", ["yes", "no"]),
        _rows("ollama-hot", ["yes", "no"], start=11),
    )

    pairwise_cohen_kappa(df, ["relevant"], intermediate_values_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_profile_name_with_punctuation_still_makes_a_filename(tmp_path):
    """A profile spelled with characters no filename can carry still writes."""
    df = _frame(
        _rows("gpt-oss:120b/hot", ["yes", "no"]),
        _rows("ollama-hot", ["yes", "no"]),
    )

    pairwise_cohen_kappa(df, ["relevant"], intermediate_values_dir=tmp_path)

    names = list(_written(tmp_path))
    assert len(names) == 1
    assert ":" not in names[0]
    assert "/" not in names[0]
    assert "gpt-oss-120b-hot" in names[0]


def test_save_name_keeps_the_separators_it_needs():
    """The underscore joins the parts of the name, so it has to survive."""
    assert save_name("prompt01_relevant_a_b.csv") == "prompt01_relevant_a_b.csv"
    assert save_name("prompt01_relevant_gpt-oss:120b_meta/llama.csv") == (
        "prompt01_relevant_gpt-oss-120b_meta-llama.csv"
    )


def test_the_directory_is_created(tmp_path):
    """A path that does not exist yet is made, as `export_as_csv` does."""
    target = tmp_path / "kappa_items" / "run01"
    df = _frame(_rows("ollama-default", ["yes"]), _rows("ollama-hot", ["yes"]))

    pairwise_cohen_kappa(df, ["relevant"], intermediate_values_dir=target)

    assert target.is_dir()
    assert len(list(target.glob("*.csv"))) == 1


# -------------------------------------------------------------- paired_answers


def _identities(*profiles, provider="ollama", model=MODEL):
    """`_profile_identities`-shaped map, built by hand for a direct call."""
    return {profile: (provider, model) for profile in profiles}


def test_paired_answers_joins_on_data_id():
    """The frame holds one row per item both profiles answered, in order."""
    df = _frame(_rows("a", ["yes", "no", "yes"]), _rows("b", ["yes", "no", "no"]))

    paired = paired_answers(df, "relevant", ("a", "b"), _identities("a", "b"))

    assert list(paired.columns) == list(INTERMEDIATE_VALUES_COLUMNS)
    assert list(paired["data_id"]) == ["D01", "D02", "D03"]
    assert list(paired["answer_a"]) == ["yes", "no", "yes"]
    assert list(paired["answer_b"]) == ["yes", "no", "no"]
    assert list(paired["same"]) == [True, True, False]


def test_paired_answers_keeps_only_what_both_answered():
    """An item only one profile answered is not an item."""
    df = _frame(_rows("a", ["yes", "no", "yes"]), _rows("b", ["yes", "no"]))

    paired = paired_answers(df, "relevant", ("a", "b"), _identities("a", "b"))

    assert list(paired["data_id"]) == ["D01", "D02"]


def test_paired_answers_normalises_before_comparing():
    """Case and whitespace are not disagreement, in the file as in the count."""
    df = _frame(_rows("a", ["Yes", " no "]), _rows("b", ["yes", "NO"]))

    paired = paired_answers(df, "relevant", ("a", "b"), _identities("a", "b"))

    assert list(paired["answer_a"]) == ["yes", "no"]
    assert list(paired["same"]) == [True, True]


def test_paired_answers_on_no_overlap_is_empty_with_the_full_columns():
    """Nothing shared still gives a frame the caller can count and write."""
    df = _frame(_rows("a", ["yes"]), _rows("b", ["yes"], start=11))

    paired = paired_answers(df, "relevant", ("a", "b"), _identities("a", "b"))

    assert paired.empty
    assert list(paired.columns) == list(INTERMEDIATE_VALUES_COLUMNS)
