"""Aggregate statistics over a loaded experiment frame.

Copied into a project's ``analysis/`` folder by ``llmexer analysis init``, so
this module imports nothing from ``llmexer`` and nothing from its sibling
analysis modules. Every function is pure: no printing, no file I/O, and a value
or a ``DataFrame`` returned so the notebook can display it and a test can assert
on it.

The semantics deliberately mirror ``ExperimentDAO.stats()`` so the notebook and
``llmexer experiment stats`` can never disagree about the same database:

* finished  -> ``status == "success"``
* errors    -> ``status`` starts with ``"Error"``
* running   -> ``state == "running"``
* open      -> ``status`` is null (generated but never run)
* tokens    -> ``total_tokens``, falling back to ``usage_tokens``, else 0

With one addition the DAO does not make. A reply cut short at ``max_tokens`` is
stored as ``status="success"`` with ``state="maxtokenreached"`` and the text
"No answer. Max token reached". Counting those as answers would overstate every
response rate, so they are reported separately as ``truncated`` and excluded
from :func:`total_responses`.
"""

from dataclasses import asdict, dataclass

import pandas as pd

# Grouping used by every per-model breakdown. A model served by two providers is
# two rows, never merged - same rule as `experiment stats` since 0.3.10.
GROUP_COLUMNS = ("provider_name", "model_name")

TRUNCATED_STATE = "maxtokenreached"
RUNNING_STATE = "running"
SUCCESS_STATUS = "success"
ERROR_PREFIX = "Error"


def _column(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    """Return a column, or a default-filled series when the frame lacks it."""

    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _is_success(df: pd.DataFrame) -> pd.Series:
    return _column(df, "status").eq(SUCCESS_STATUS).fillna(False).astype(bool)


def _is_error(df: pd.DataFrame) -> pd.Series:
    # `na=False`: under the pandas 3 string dtype `startswith` yields NA for
    # missing values, and an NA-bearing mask cannot index a frame.
    return _column(df, "status").astype("object").fillna("").astype(str).str.startswith(ERROR_PREFIX, na=False)


def _is_truncated(df: pd.DataFrame) -> pd.Series:
    return _column(df, "state").eq(TRUNCATED_STATE).fillna(False).astype(bool)


def _is_running(df: pd.DataFrame) -> pd.Series:
    return _column(df, "state").eq(RUNNING_STATE).fillna(False).astype(bool)


def _is_open(df: pd.DataFrame) -> pd.Series:
    return _column(df, "status").isna()


def tokens_series(df: pd.DataFrame) -> pd.Series:
    """Token count per row: ``total_tokens``, else ``usage_tokens``, else 0."""

    total = pd.to_numeric(_column(df, "total_tokens"), errors="coerce")
    usage = pd.to_numeric(_column(df, "usage_tokens"), errors="coerce")

    return total.fillna(usage).fillna(0).astype("int64")


def _groups(df: pd.DataFrame, by) -> list:
    """Resolve the grouping columns, keeping only those the frame actually has."""

    if by is None:
        by = GROUP_COLUMNS
    if isinstance(by, str):
        by = (by,)

    return [name for name in by if name in df.columns]


def _empty(columns: list) -> pd.DataFrame:
    """An empty frame carrying ``columns``, so callers never branch on shape."""

    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def total_requests(df: pd.DataFrame) -> int:
    """Number of generated combinations, run or not."""

    return int(len(df))


def total_responses(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Per provider+model counts of requests and what came back.

    Columns: ``requests``, ``responses``, ``finished``, ``truncated``,
    ``errors``, ``open``. ``responses`` counts genuine answers only - a reply
    truncated at ``max_tokens`` carries no content and is not one.
    """

    groups = _groups(df, by)
    columns = groups + ["requests", "responses", "finished", "truncated", "errors", "open"]
    if df.empty or not groups:
        if df.empty:
            return _empty(columns)

    work = df.copy()
    work["_finished"] = _is_success(df)
    work["_truncated"] = _is_truncated(df)
    work["_errors"] = _is_error(df)
    work["_open"] = _is_open(df)
    work["_responses"] = work["_finished"] & ~work["_truncated"]

    if not groups:
        aggregated = pd.DataFrame(
            [
                {
                    "requests": len(work),
                    "responses": int(work["_responses"].sum()),
                    "finished": int(work["_finished"].sum()),
                    "truncated": int(work["_truncated"].sum()),
                    "errors": int(work["_errors"].sum()),
                    "open": int(work["_open"].sum()),
                }
            ]
        )
        return aggregated

    aggregated = (
        work.groupby(list(groups), dropna=False)
        .agg(
            requests=("_finished", "size"),
            responses=("_responses", "sum"),
            finished=("_finished", "sum"),
            truncated=("_truncated", "sum"),
            errors=("_errors", "sum"),
            open=("_open", "sum"),
        )
        .reset_index()
        .sort_values(list(groups))
        .reset_index(drop=True)
    )
    for name in ("requests", "responses", "finished", "truncated", "errors", "open"):
        aggregated[name] = aggregated[name].astype("int64")

    return aggregated


def response_time_stats(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Elapsed-time statistics per provider+model, over finished rows only.

    Unrun rows carry a NULL or zero ``elapsed_seconds`` and would drag every
    mean towards zero, so they are excluded rather than counted as instant.
    """

    groups = _groups(df, by)
    columns = groups + ["count", "min", "max", "mean", "median", "p95", "total"]
    if df.empty:
        return _empty(columns)

    elapsed = pd.to_numeric(_column(df, "elapsed_seconds"), errors="coerce")
    finished = df[_is_success(df) & elapsed.notna()].copy()
    finished["_elapsed"] = elapsed[_is_success(df) & elapsed.notna()]
    if finished.empty:
        return _empty(columns)

    if not groups:
        series = finished["_elapsed"]
        return pd.DataFrame(
            [
                {
                    "count": int(series.count()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "p95": float(series.quantile(0.95)),
                    "total": float(series.sum()),
                }
            ]
        )

    aggregated = (
        finished.groupby(list(groups), dropna=False)["_elapsed"]
        .agg(
            count="count",
            min="min",
            max="max",
            mean="mean",
            median="median",
            p95=lambda s: s.quantile(0.95),
            total="sum",
        )
        .reset_index()
        .sort_values(list(groups))
        .reset_index(drop=True)
    )

    return aggregated


def token_stats(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Token totals and averages per provider+model.

    ``prompt_tokens`` / ``completion_tokens`` are reported when the frame
    carries them - ``transform.load_experiment_db`` parses them out of
    ``response_json``. They are nullable and never zero-filled: a zero would
    understate prompt cost, so a missing split shows up as ``<NA>`` and in the
    ``coverage`` column rather than as a plausible-looking number.
    """

    groups = _groups(df, by)
    columns = groups + ["requests", "tokens_total", "tokens_mean"]
    if df.empty:
        return _empty(columns + ["coverage"])

    work = df.copy()
    work["_tokens"] = tokens_series(df)
    finished = _is_success(df)
    work = work[finished]
    if work.empty:
        return _empty(columns + ["coverage"])

    has_split = "prompt_tokens" in df.columns or "completion_tokens" in df.columns
    if has_split:
        work["_prompt"] = pd.to_numeric(_column(work, "prompt_tokens"), errors="coerce")
        work["_completion"] = pd.to_numeric(_column(work, "completion_tokens"), errors="coerce")

    def _aggregate(frame: pd.DataFrame) -> dict:
        row = {
            "requests": int(len(frame)),
            "tokens_total": int(frame["_tokens"].sum()),
            "tokens_mean": float(frame["_tokens"].mean()),
        }
        if has_split:
            row["prompt_tokens_total"] = frame["_prompt"].sum(min_count=1)
            row["completion_tokens_total"] = frame["_completion"].sum(min_count=1)
            row["prompt_tokens_mean"] = frame["_prompt"].mean()
            row["completion_tokens_mean"] = frame["_completion"].mean()
            # Share of finished rows that actually carried a usage breakdown.
            row["coverage"] = float(frame["_completion"].notna().mean())
        return row

    if not groups:
        return pd.DataFrame([_aggregate(work)])

    rows = []
    for key, frame in work.groupby(list(groups), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), **_aggregate(frame)})

    return pd.DataFrame(rows).sort_values(list(groups)).reset_index(drop=True)


def failed_calls(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Failure counts and rates per provider+model, plus an ``all`` total row.

    The rate is failures over *attempted* rows (finished + errored); rows that
    were never run are reported in ``open`` but do not make the rate look better.
    """

    groups = _groups(df, by)
    columns = groups + ["requests", "attempted", "failed", "failure_rate", "truncated", "open"]
    if df.empty:
        return _empty(columns)

    work = df.copy()
    work["_failed"] = _is_error(df)
    work["_finished"] = _is_success(df)
    work["_truncated"] = _is_truncated(df)
    work["_open"] = _is_open(df)
    work["_attempted"] = work["_failed"] | work["_finished"]

    def _aggregate(frame: pd.DataFrame) -> dict:
        attempted = int(frame["_attempted"].sum())
        failed = int(frame["_failed"].sum())
        return {
            "requests": int(len(frame)),
            "attempted": attempted,
            "failed": failed,
            "failure_rate": (failed / attempted) if attempted else 0.0,
            "truncated": int(frame["_truncated"].sum()),
            "open": int(frame["_open"].sum()),
        }

    if not groups:
        return pd.DataFrame([_aggregate(work)])

    rows = []
    for key, frame in work.groupby(list(groups), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), **_aggregate(frame)})

    per_group = pd.DataFrame(rows).sort_values(list(groups)).reset_index(drop=True)
    overall = {**{name: "all" for name in groups}, **_aggregate(work)}

    return pd.concat([per_group, pd.DataFrame([overall])], ignore_index=True)


@dataclass(frozen=True)
class ExperimentSummary:
    """One-glance overview of an experiment, aggregating the functions above."""

    requests: int = 0
    responses: int = 0
    finished: int = 0
    truncated: int = 0
    errors: int = 0
    running: int = 0
    open: int = 0
    providers: int = 0
    models: int = 0
    profiles: int = 0
    total_tokens: int = 0
    elapsed_seconds_total: float = 0.0
    elapsed_seconds_mean: float = 0.0
    failure_rate: float = 0.0

    def to_dict(self) -> dict:
        """The summary as a plain dict."""

        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        """A two-column ``metric``/``value`` frame, for display in a notebook."""

        return pd.DataFrame({"metric": list(asdict(self)), "value": list(asdict(self).values())})

    def __str__(self) -> str:
        lines = [f"{name:<22} {value}" for name, value in asdict(self).items()]
        return "\n".join(lines)


def summary(df: pd.DataFrame) -> ExperimentSummary:
    """Aggregate an experiment frame into an :class:`ExperimentSummary`."""

    if df.empty:
        return ExperimentSummary()

    finished = _is_success(df)
    truncated = _is_truncated(df)
    errors = _is_error(df)
    elapsed = pd.to_numeric(_column(df, "elapsed_seconds"), errors="coerce")[finished]
    attempted = int(finished.sum()) + int(errors.sum())

    def _unique(name: str) -> int:
        return int(_column(df, name).nunique(dropna=True)) if name in df.columns else 0

    return ExperimentSummary(
        requests=len(df),
        responses=int((finished & ~truncated).sum()),
        finished=int(finished.sum()),
        truncated=int(truncated.sum()),
        errors=int(errors.sum()),
        running=int(_is_running(df).sum()),
        open=int(_is_open(df).sum()),
        providers=_unique("provider_name"),
        models=_unique("model_name"),
        profiles=_unique("profile_name"),
        total_tokens=int(tokens_series(df)[finished].sum()),
        elapsed_seconds_total=float(elapsed.sum()) if len(elapsed) else 0.0,
        elapsed_seconds_mean=float(elapsed.mean()) if elapsed.notna().any() else 0.0,
        failure_rate=(int(errors.sum()) / attempted) if attempted else 0.0,
    )


# ---------------------------------------------------------------- answer values
# Values a model uses for a yes/no answer, lower-cased. A column whose values are
# all drawn from one of these pairs is a boolean answer however it was spelled.
_BOOLEAN_VOCABULARIES = (
    {"yes", "no"},
    {"true", "false"},
    {"y", "n"},
    {"1", "0"},
)

# A column with at most this many distinct values is worth a bar chart; beyond it
# the chart is a barcode and the table says more.
MAX_PLOTTABLE_VALUES = 12

# Longer values read as prose rather than as a label, so they are not charted.
MAX_LABEL_LENGTH = 40


def _classify_values(series: pd.Series) -> str:
    """Classify a column of answers as boolean, countable, categorical or text."""

    present = series.dropna()
    if present.empty:
        return "empty"

    # A list or dict in a cell is a nested answer, not a label.
    if present.map(lambda value: isinstance(value, (list, dict, set))).any():
        return "nested"

    labels = {str(value).strip().lower() for value in present}
    if labels and any(labels <= vocabulary for vocabulary in _BOOLEAN_VOCABULARIES):
        return "boolean"

    numeric = pd.to_numeric(present, errors="coerce")
    if numeric.notna().all():
        # Whole numbers over a small range are ratings/counts - 1, 2, 3.
        whole = (numeric % 1 == 0).all()
        if whole and numeric.nunique() <= MAX_PLOTTABLE_VALUES:
            return "countable"
        return "numeric"

    if present.nunique() <= MAX_PLOTTABLE_VALUES and max(len(str(v)) for v in present) <= MAX_LABEL_LENGTH:
        return "categorical"

    return "text"


# Kinds worth a bar chart: a small, closed set of labels.
PLOTTABLE_KINDS = ("boolean", "countable", "categorical")


def value_counts_summary(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One row per answer column: how many values, how many distinct, what kind.

    Built for the frame :func:`transform.flattened_only` returns - the values the
    models actually answered with. ``kind`` is ``boolean`` (yes/no, true/false,
    1/0), ``countable`` (whole numbers over a small range, e.g. a 1-5 rating),
    ``categorical`` (a small closed set of labels), ``numeric``, ``text``,
    ``nested`` (lists or objects) or ``empty``; ``plottable`` marks the three that
    a bar chart says something about.

    Columns: ``column``, ``values``, ``missing``, ``unique``, ``kind``,
    ``plottable``, ``top``, ``top_count``.
    """

    names = list(columns) if columns is not None else list(df.columns)
    rows = []
    for name in names:
        if name not in df.columns:
            continue
        series = df[name]
        present = series.dropna()
        kind = _classify_values(series)
        counts = (
            present.astype(str).value_counts() if not present.empty and kind != "nested" else pd.Series(dtype="int64")
        )
        rows.append(
            {
                "column": name,
                "values": int(present.shape[0]),
                "missing": int(series.isna().sum()),
                "unique": int(present.nunique()) if kind != "nested" else pd.NA,
                "kind": kind,
                "plottable": kind in PLOTTABLE_KINDS,
                "top": counts.index[0] if not counts.empty else pd.NA,
                "top_count": int(counts.iloc[0]) if not counts.empty else pd.NA,
            }
        )

    columns_out = ["column", "values", "missing", "unique", "kind", "plottable", "top", "top_count"]
    if not rows:
        return _empty(columns_out)

    return pd.DataFrame(rows)[columns_out]


def value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Counts and shares for one answer column, most common first."""

    if column not in df.columns:
        raise KeyError(f"missing column: {column!r}; available: {sorted(df.columns)}")

    counts = df[column].dropna().astype(str).value_counts()
    out = counts.rename("count").to_frame()
    out["share"] = (out["count"] / out["count"].sum()) if out["count"].sum() else 0.0

    return out.reset_index(names=column)
