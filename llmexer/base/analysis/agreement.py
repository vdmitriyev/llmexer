"""Pairwise agreement between the configurations that answered an experiment.

Copied into a project's ``analysis/`` folder by the CLI and imported there as a
top-level module, so it imports nothing from ``llmexer`` and nothing from its
sibling modules.

The input is the flattened CSV the experiment notebook writes: one row per
combination, carrying ``code``, ``model_name``, ``provider_name``,
``profile_name`` and one column per key the model answered with. Two
configurations are compared on the items they both answered - Cohen's kappa
rather than a raw match rate, so agreement that chance alone would produce is
discounted.
"""

import itertools

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# What makes one rater. `profile_name` alone is not enough: the same profile
# name can be served by two models, and those are two different raters.
RATER_COLUMNS = ("provider_name", "model_name", "profile_name")

# Values a model uses for a yes/no answer, lower-cased. A column whose values
# are all drawn from one of these pairs is a yes/no answer however it was
# spelled. Kept in step with ``stats._BOOLEAN_VOCABULARIES``.
BOOLEAN_VOCABULARIES = (
    {"yes", "no"},
    {"true", "false"},
    {"y", "n"},
    {"1", "0"},
)

# Columns of the table :func:`pairwise_cohen_kappa` returns.
RESULT_COLUMNS = (
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

    Subclasses ``KeyError``, so a caller that wants to be tolerant can catch
    that instead - the same arrangement ``transform.MissingColumnError`` uses.
    """


def item_keys(df: pd.DataFrame) -> pd.Series:
    """Return the item each answer is about: ``code`` without its rater tail.

    ``code`` is built as ``<data_id>_<prompt_id>_<model_name>_<profile_name>``,
    and the flattened CSV carries no separate data ID. The tail is removed by
    name rather than by splitting on ``_``, so a data ID or a model name that
    contains an underscore is still handled exactly.

    Two raters therefore meet on the same data row *and* the same prompt: a
    model answering two prompts about one paper counts as two items.
    """

    if df.empty or "code" not in df.columns:
        return pd.Series(dtype="object", index=df.index)

    def _strip(row) -> str:
        code = str(row["code"])
        tail = f"_{row.get('model_name', '')}_{row.get('profile_name', '')}"
        return code[: -len(tail)] if tail != "_" and code.endswith(tail) else code

    return df.apply(_strip, axis=1)


def _labels(series: pd.Series) -> pd.Series:
    """Normalise answers to comparable labels: stripped and lower-cased."""

    return series.dropna().astype(str).str.strip().str.lower()


def yes_no_fields(df: pd.DataFrame, columns=None) -> list:
    """Return the answer columns holding yes/no values.

    A column qualifies when every value it actually carries is drawn from one
    vocabulary - ``yes``/``no``, ``true``/``false``, ``y``/``n`` or ``1``/``0``.
    A column answering only ``yes`` throughout still qualifies; an empty one
    does not. The identity columns are never answers and are skipped.
    """

    names = list(columns) if columns is not None else [c for c in df.columns if c not in _identity_columns()]

    fields = []
    for name in names:
        if name not in df.columns:
            continue
        labels = set(_labels(df[name]))
        if labels and any(labels <= vocabulary for vocabulary in BOOLEAN_VOCABULARIES):
            fields.append(name)

    return fields


def _identity_columns() -> tuple:
    """Columns that identify the combination rather than the answer."""

    return ("code",) + RATER_COLUMNS


def _rater_label(rater: tuple) -> str:
    """Readable name of one rater, for sorting and for a pivoted table."""

    return " / ".join(str(part) for part in rater)


def _kappa(left: pd.Series, right: pd.Series) -> float:
    """Cohen's kappa for two aligned label series.

    Returns ``NaN`` where kappa is undefined rather than a number that reads
    like a result: with nothing to compare, or when both raters used one and
    the same single label, chance agreement is already 1 and the statistic
    carries no information. The two counts next to it tell that story.
    """

    if left.empty:
        return float("nan")

    if len(set(left) | set(right)) < 2:
        return float("nan")

    return float(cohen_kappa_score(left, right))


def pairwise_cohen_kappa(df: pd.DataFrame, fields) -> pd.DataFrame:
    """Cohen's kappa for every pair of configurations, per field.

    One row per (field, pair). ``items`` is how many items both configurations
    answered - the pair is scored on that overlap only - split into
    ``counter_same_answer`` and ``counter_different_answer``, the plain counts
    behind the chance-corrected ``cohen_kappa``.

    ``fields`` is always given by the caller: which columns are answers, and
    which of those are worth comparing, is a decision for the notebook - see
    :func:`yes_no_fields` for the detection that feeds it. Every name must be a
    column of ``df``.

    Fewer than two configurations, or no fields, gives an empty table with the
    same columns.

    Raises:
        MissingFieldError: if any name in ``fields`` is not a column of ``df``.
    """

    names = list(fields)
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise MissingFieldError(f"field(s) {missing} are not in the frame; available columns: {sorted(df.columns)}")

    empty = pd.DataFrame(columns=list(RESULT_COLUMNS))
    if not names or df.empty or any(column not in df.columns for column in _identity_columns()):
        return empty

    frame = df.copy()
    frame["_item"] = item_keys(frame)

    raters = sorted({tuple(str(row[c]) for c in RATER_COLUMNS) for _, row in frame.iterrows()})
    if len(raters) < 2:
        return empty

    rows = []
    for field in names:
        answers = {rater: _labels(_rater_frame(frame, rater).set_index("_item")[field]) for rater in raters}
        for left, right in itertools.combinations(raters, 2):
            shared = answers[left].index.intersection(answers[right].index)
            total_items = len(shared)
            a = answers[left].reindex(shared)
            b = answers[right].reindex(shared)
            matches = int((a == b).sum())
            rows.append(
                {
                    "field": field,
                    "provider_a": left[0],
                    "model_a": left[1],
                    "profile_a": left[2],
                    "provider_b": right[0],
                    "model_b": right[1],
                    "profile_b": right[2],
                    "items": total_items,
                    "counter_same_answer": matches,
                    "counter_different_answer": total_items - matches,
                    "cohen_kappa": _kappa(a, b),
                }
            )

    result = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))

    return result.sort_values(
        by=["field", "provider_a", "model_a", "profile_a", "provider_b", "model_b", "profile_b"]
    ).reset_index(drop=True)


def _rater_frame(frame: pd.DataFrame, rater: tuple) -> pd.DataFrame:
    """The rows one rater contributed."""

    mask = pd.Series(True, index=frame.index)
    for column, value in zip(RATER_COLUMNS, rater):
        mask &= frame[column].astype(str) == value

    return frame[mask]


def kappa_matrix(result: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot one field of :func:`pairwise_cohen_kappa` into a rater x rater table."""

    selected = result[result["field"] == field]
    if selected.empty:
        return pd.DataFrame()

    labels = sorted(
        {_rater_label((r.provider_a, r.model_a, r.profile_a)) for r in selected.itertuples()}
        | {_rater_label((r.provider_b, r.model_b, r.profile_b)) for r in selected.itertuples()}
    )
    matrix = pd.DataFrame(index=labels, columns=labels, dtype="float64")
    for row in selected.itertuples():
        left = _rater_label((row.provider_a, row.model_a, row.profile_a))
        right = _rater_label((row.provider_b, row.model_b, row.profile_b))
        matrix.loc[left, right] = row.cohen_kappa
        matrix.loc[right, left] = row.cohen_kappa

    return matrix
