"""Agreement between the profiles that answered an experiment.

Copied into a project's ``analysis/`` folder and imported there as a top-level
module, so it imports nothing from ``llmexer`` and nothing from its siblings.

The input is the flattened frame the experiment notebook builds: one row per
combination, carrying the identity columns and one column per key the model
answered with.

How a pair is scored:

* ``prompt_id`` is the set. Each prompt is scored on its own.
* ``data_id`` is the item. Two profiles meet on the items both answered.
* the score is Cohen's kappa, not a raw match rate, so agreement that chance
  alone would produce is discounted.
"""

import itertools
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Columns that identify a combination rather than an answer. Mirror of
# ``transform.IDENTITY_COLUMNS``; the copied modules may not import each other,
# so the tuple is repeated here. ``data_id`` is on the list because a data ID of
# ``0`` / ``1`` would otherwise be read as a yes/no answer.
IDENTITY_COLUMNS = ("code", "data_id", "prompt_id", "model_name", "provider_name", "profile_name")

# What a row needs before it can say which item an answer is about.
ITEM_COLUMNS = ("prompt_id", "data_id")

# Spellings of a yes/no answer, lower-cased. A column counts as yes/no when
# every value it carries comes from one of these sets. Kept in step with
# ``stats._BOOLEAN_VOCABULARIES``.
BOOLEAN_VOCABULARIES = (
    {"yes", "no"},
    {"true", "false"},
    {"y", "n"},
    {"1", "0"},
)

# CSV dialect, the one every other file in a project uses. Spelled out rather
# than imported, for the same reason as IDENTITY_COLUMNS.
CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8"

# One comparison: one row per item both profiles answered. This frame is what
# kappa is computed from and what an intermediate-values file holds, so the
# score and the evidence for it cannot drift apart.
INTERMEDIATE_VALUES_COLUMNS = (
    "prompt_id",
    "field",
    "provider_a",
    "model_a",
    "profile_a",
    "provider_b",
    "model_b",
    "profile_b",
    "data_id",
    "answer_a",
    "answer_b",
    "same",
)

# Columns of the table :func:`pairwise_cohen_kappa` returns.
RESULT_COLUMNS = (
    "prompt_id",
    "field",
    "provider_a",
    "model_a",
    "profile_a",
    "provider_b",
    "model_b",
    "profile_b",
    "items",
    "counter_same_answer",
    "counter_different_answer",
    "cohen_kappa",
)


class MissingFieldError(KeyError):
    """Raised when a field to compare is not a column of the frame.

    Subclasses ``KeyError`` so a tolerant caller can catch that instead, the
    arrangement ``transform.MissingColumnError`` uses.
    """


class MissingColumnError(KeyError):
    """Raised when the frame has no ``data_id`` / ``prompt_id`` to group on.

    A frame exported from a database older than 0.4.6 has neither: both values
    lived inside ``code`` and nowhere else.

    Reading them back out of ``code`` is deliberately not offered. The grouping
    would then depend on how old the export was, without saying so.
    """


class AmbiguousProfileError(KeyError):
    """Raised when one profile name covers more than one model.

    Profiles are what get paired here; provider and model are reported as
    attributes of each one. ``llm-params.csv`` is keyed on all three, so a name
    shared by two models causes two problems:

    * there is no single provider/model to report for it;
    * the join on ``data_id`` matches every item twice, inflating ``items`` and
      the kappa with it.
    """


def _labels(series: pd.Series) -> pd.Series:
    """Normalise answers to comparable labels: stripped and lower-cased."""

    return series.dropna().astype(str).str.strip().str.lower()


def yes_no_fields(df: pd.DataFrame, columns=None) -> list:
    """Return the answer columns holding yes/no values.

    A column qualifies when every value it carries comes from one vocabulary:
    ``yes``/``no``, ``true``/``false``, ``y``/``n`` or ``1``/``0``. A column
    answering only ``yes`` throughout still qualifies; an empty one does not.

    The identity columns are never answers, so they are skipped.
    """

    names = list(columns) if columns is not None else [c for c in df.columns if c not in IDENTITY_COLUMNS]

    fields = []
    for name in names:
        if name not in df.columns:
            continue
        labels = set(_labels(df[name]))
        if labels and any(labels <= vocabulary for vocabulary in BOOLEAN_VOCABULARIES):
            fields.append(name)

    return fields


def save_name(value: str) -> str:
    """A filename-safe version of an assembled file name.

    Model and profile names carry characters no filename can, such as
    ``gpt-oss:120b`` or ``meta-llama/Llama-3-8b``. Runs of anything but a
    letter, digit, dot, dash or underscore collapse to a single dash.

    The underscore survives: it is what separates the parts of the name.
    """

    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


def _kappa(left: pd.Series, right: pd.Series) -> float:
    """Cohen's kappa for two aligned label series.

    Returns ``NaN`` where kappa says nothing, rather than a number that reads
    like a result. That happens with no items at all, and when both profiles
    used one single label throughout - chance agreement is then already 1.

    The two counts reported beside it tell that story instead.
    """

    if left.empty:
        return float("nan")

    if len(set(left) | set(right)) < 2:
        return float("nan")

    return float(cohen_kappa_score(left, right))


def _profile_identities(prompt_rows: pd.DataFrame) -> dict:
    """Map each profile of one prompt to its ``(provider, model)``.

    Raises:
        AmbiguousProfileError: if a profile name covers more than one model.
    """

    identities = {}
    profiles = sorted(prompt_rows["profile_name"].astype(str).unique())

    for profile in profiles:
        selected = prompt_rows[prompt_rows["profile_name"].astype(str) == profile]

        seen = set()
        for _, row in selected.iterrows():
            seen.add((str(row["provider_name"]), str(row["model_name"])))

        if len(seen) > 1:
            named = ", ".join(f"{provider}/{model}" for provider, model in sorted(seen))
            raise AmbiguousProfileError(
                f"profile '{profile}' is used by more than one model ({named}). Pairing is done "
                "on the profile name, so give each model its own profile in llm-params.csv - "
                "e.g. 'ollama-gemma4-default'."
            )

        identities[profile] = sorted(seen)[0]

    return identities


def _one_profile(prompt_rows: pd.DataFrame, field: str, profile: str) -> pd.DataFrame:
    """One profile's answers for one field: ``data_id`` and ``answer``.

    Unanswered rows are dropped, so what is left is what the profile really
    answered. The overlap with another profile is then just an inner join.

    The frame is built from ``.values`` so pandas does not realign it on the
    original index.
    """

    selected = prompt_rows[prompt_rows["profile_name"].astype(str) == str(profile)]
    answered = selected[selected[field].notna()]

    return pd.DataFrame(
        {
            "data_id": answered["data_id"].astype(str).values,
            "answer": _labels(answered[field]).values,
        }
    )


def paired_answers(prompt_rows: pd.DataFrame, field: str, pair: tuple, identities: dict) -> pd.DataFrame:
    """The items two profiles both answered, one row each.

    ``pair`` is ``(profile_a, profile_b)``. ``identities`` maps a profile to its
    ``(provider, model)``, as :func:`_profile_identities` returns it.

    The inner join on ``data_id`` is what "the items both answered" means here.
    The frame comes back in :data:`INTERMEDIATE_VALUES_COLUMNS`, and is both
    what kappa is computed from and what ``intermediate_values_dir`` writes.
    """

    profile_a, profile_b = pair
    provider_a, model_a = identities[profile_a]
    provider_b, model_b = identities[profile_b]

    left = _one_profile(prompt_rows, field, profile_a)
    right = _one_profile(prompt_rows, field, profile_b)
    paired = left.merge(right, on="data_id", suffixes=("_a", "_b"))

    paired["prompt_id"] = str(prompt_rows["prompt_id"].iloc[0]) if not prompt_rows.empty else ""
    paired["field"] = field
    paired["provider_a"] = provider_a
    paired["model_a"] = model_a
    paired["profile_a"] = profile_a
    paired["provider_b"] = provider_b
    paired["model_b"] = model_b
    paired["profile_b"] = profile_b
    paired["same"] = paired["answer_a"] == paired["answer_b"]

    return paired.sort_values("data_id").reindex(columns=list(INTERMEDIATE_VALUES_COLUMNS)).reset_index(drop=True)


def calculate_kappa_paired_data(
    paired: pd.DataFrame, prompt_id: str, field: str, pair: tuple, identities: dict
) -> dict:
    """Score one already-paired comparison: the counts and Cohen's kappa.

    Everything is counted off ``paired``, the frame :func:`paired_answers`
    built, so the table can never disagree with the CSV written beside it.

    Returns one row of :data:`RESULT_COLUMNS`.
    """

    profile_a, profile_b = pair
    provider_a, model_a = identities[profile_a]
    provider_b, model_b = identities[profile_b]

    total_items = len(paired)
    same_count = int(paired["same"].sum()) if total_items else 0
    different_count = total_items - same_count
    kappa_value = _kappa(paired["answer_a"], paired["answer_b"])

    return {
        "prompt_id": prompt_id,
        "field": field,
        "provider_a": provider_a,
        "model_a": model_a,
        "profile_a": profile_a,
        "provider_b": provider_b,
        "model_b": model_b,
        "profile_b": profile_b,
        "items": total_items,
        "counter_same_answer": same_count,
        "counter_different_answer": different_count,
        "cohen_kappa": kappa_value,
    }


def _write_intermediate_values(
    paired: pd.DataFrame, intermediate_values_dir, prompt_id: str, field: str, pair: tuple
) -> Path:
    """Write one comparison to ``<prompt>_<field>_<profile_a>_<profile_b>.csv``."""

    profile_a, profile_b = pair
    filename = save_name(f"{prompt_id}_{field}_{profile_a}_{profile_b}.csv")
    target = Path(intermediate_values_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(target, index=False, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)

    return target


def _require_fields(df: pd.DataFrame, names: list) -> None:
    """Reject a field that is not a column of the frame.

    Raises:
        MissingFieldError: if any name is absent.
    """

    missing = [name for name in names if name not in df.columns]
    if not missing:
        return

    raise MissingFieldError(f"field(s) {missing} are not in the frame; available columns: {sorted(df.columns)}")


def _require_item_columns(df: pd.DataFrame) -> None:
    """Reject a frame that cannot say which item an answer is about.

    Raises:
        MissingColumnError: if ``data_id`` or ``prompt_id`` is absent.
    """

    missing = [name for name in ITEM_COLUMNS if name not in df.columns]
    if not missing:
        return

    raise MissingColumnError(
        f"the frame has no {missing} column(s). This analysis groups by 'prompt_id' and "
        "matches items on 'data_id'. Re-export from a database generated by 0.4.6 or later, "
        "or migrate an older one first (see migrations.md)."
    )


def pairwise_cohen_kappa(df: pd.DataFrame, fields, intermediate_values_dir=None) -> pd.DataFrame:
    """Cohen's kappa for every pair of profiles, per prompt and field.

    One row per (``prompt_id``, field, profile pair):

    * ``prompt_id`` is the set the pair is scored within. A model answering two
      prompts about one paper is two items under two prompts, not two items in
      one pool.
    * ``items`` is how many items both profiles answered. The pair is scored on
      that overlap alone.
    * ``counter_same_answer`` and ``counter_different_answer`` are the plain
      counts behind the chance-corrected ``cohen_kappa``.

    ``fields`` always comes from the caller. Which columns are answers, and
    which are worth comparing, is the notebook's decision; :func:`yes_no_fields`
    is the detection that feeds it. Every name must be a column of ``df``.

    ``intermediate_values_dir`` is off when ``None``. Given a directory, each
    comparison also writes the rows it was scored from to
    ``<prompt>_<field>_<profile_a>_<profile_b>.csv`` there, for checking a
    surprising kappa by hand. The returned table is the same either way.

    Fewer than two profiles, or no fields, gives an empty table with the same
    columns.

    Raises:
        MissingFieldError: if any name in ``fields`` is not a column of ``df``.
        MissingColumnError: if ``df`` has no ``data_id`` / ``prompt_id``.
        AmbiguousProfileError: if a profile name covers more than one model.
    """

    names = list(fields)
    _require_fields(df, names)

    empty = pd.DataFrame(columns=list(RESULT_COLUMNS))
    if not names or df.empty:
        return empty

    _require_item_columns(df)
    if any(column not in df.columns for column in IDENTITY_COLUMNS):
        return empty

    rows = []
    prompt_ids = sorted(df["prompt_id"].astype(str).unique())

    for prompt_id in prompt_ids:
        prompt_rows = df[df["prompt_id"].astype(str) == prompt_id]
        identities = _profile_identities(prompt_rows)
        pairs = list(itertools.combinations(sorted(identities), 2))

        for field in names:
            for pair in pairs:
                paired = paired_answers(prompt_rows, field, pair, identities)

                if intermediate_values_dir is not None and not paired.empty:
                    _write_intermediate_values(paired, intermediate_values_dir, prompt_id, field, pair)

                rows.append(calculate_kappa_paired_data(paired, prompt_id, field, pair, identities))

    if not rows:
        return empty

    result = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))

    return result.sort_values(by=["prompt_id", "field", "profile_a", "profile_b"]).reset_index(drop=True)


def kappa_matrix(result: pd.DataFrame, field: str, prompt_id: str) -> pd.DataFrame:
    """Pivot one field of one prompt into a profile x profile table.

    The prompt is required rather than optional. A matrix built across prompts
    would put each pair's several kappas in one cell and keep whichever was
    written last. Averaging them instead would weight a three-item prompt like a
    three-hundred-item one, so the caller picks the prompt.
    """

    matches_field = result["field"] == field
    matches_prompt = result["prompt_id"].astype(str) == str(prompt_id)
    selected = result[matches_field & matches_prompt]
    if selected.empty:
        return pd.DataFrame()

    labels = sorted(set(selected["profile_a"]) | set(selected["profile_b"]))
    matrix = pd.DataFrame(index=labels, columns=labels, dtype="float64")
    for row in selected.itertuples():
        matrix.loc[row.profile_a, row.profile_b] = row.cohen_kappa
        matrix.loc[row.profile_b, row.profile_a] = row.cohen_kappa

    return matrix
