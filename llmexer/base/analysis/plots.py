"""Seaborn charts for an experiment or search frame.

Copied into a project's ``analysis/`` folder by ``llmexer analysis init``, so
this module imports nothing from ``llmexer`` and nothing from its sibling
analysis modules - the small groupings it needs are done inline rather than
borrowed from ``stats.py``.

Contract shared by every function here:

* it takes a ``DataFrame`` and returns a ``matplotlib.axes.Axes``;
* it never calls ``show()`` and never writes a file - the notebook renders
  inline and nothing is exported as PNG;
* passing ``ax`` draws into that axis, otherwise one is created;
* an empty frame draws a "No data" note and returns normally, so a notebook run
  against a half-finished project still executes top to bottom;
* a missing column raises ``KeyError``.

Importing this module does not touch ``matplotlib.rcParams``; call
:func:`setup_style` explicitly if you want the shared look.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Default figure size for an axis this module creates itself.
FIGSIZE = (9, 5)

# Grouping label used on the category axis of the per-model charts.
MODEL_LABEL = "model / provider"

SUCCESS_STATUS = "success"
ERROR_PREFIX = "Error"
TRUNCATED_STATE = "maxtokenreached"


def setup_style(*, context: str = "notebook", palette: str = "colorblind", grid: bool = True) -> None:
    """Apply the shared seaborn look. Call it from the notebook, not on import."""

    sns.set_theme(context=context, style="whitegrid" if grid else "white", palette=palette)


def _axis(ax, figsize=FIGSIZE):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _no_data(ax, message: str = "No data"):
    """Annotate an empty axis instead of raising on an empty frame."""

    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#888")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _require(df: pd.DataFrame, *columns: str) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise KeyError(f"missing column(s): {missing}; available: {sorted(df.columns)}")


def _model_label(df: pd.DataFrame) -> pd.Series:
    """``model (provider)`` labels, so one model on two providers stays two bars."""

    _require(df, "model_name")
    if "provider_name" in df.columns:
        return df["model_name"].astype(str) + " (" + df["provider_name"].astype(str) + ")"
    return df["model_name"].astype(str)


def _tokens(df: pd.DataFrame) -> pd.Series:
    _require(df, "total_tokens")

    return pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0)


def plot_responses_by_model(df: pd.DataFrame, ax=None, *, title: str = "Responses by model"):
    """Finished vs truncated vs errored vs open counts, one group per model."""

    ax = _axis(ax)
    if df.empty:
        return _no_data(ax)
    _require(df, "model_name")

    status = pd.Series("open", index=df.index, dtype="object")
    if "status" in df.columns:
        text = df["status"].astype("object").fillna("").astype(str)
        status[text.str.startswith(ERROR_PREFIX, na=False)] = "error"
        status[text.eq(SUCCESS_STATUS)] = "finished"
    if "state" in df.columns:
        status[df["state"].eq(TRUNCATED_STATE).fillna(False)] = "truncated"

    work = pd.DataFrame({MODEL_LABEL: _model_label(df), "outcome": status})
    counts = work.groupby([MODEL_LABEL, "outcome"], dropna=False).size().reset_index(name="responses")

    sns.barplot(data=counts, y=MODEL_LABEL, x="responses", hue="outcome", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("responses")

    return ax


def plot_response_time_by_model(df: pd.DataFrame, ax=None, *, title: str = "Response time by model"):
    """Distribution of ``elapsed_seconds`` per model, over finished rows only."""

    ax = _axis(ax)
    if df.empty:
        return _no_data(ax)
    _require(df, "model_name", "elapsed_seconds")

    elapsed = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    keep = elapsed.notna()
    if "status" in df.columns:
        keep &= df["status"].eq(SUCCESS_STATUS).fillna(False)
    if not keep.any():
        return _no_data(ax, "No finished calls")

    work = pd.DataFrame({MODEL_LABEL: _model_label(df)[keep], "seconds": elapsed[keep]})
    sns.boxplot(data=work, y=MODEL_LABEL, x="seconds", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("seconds per call")

    return ax


def plot_tokens_by_model(df: pd.DataFrame, ax=None, *, title: str = "Tokens by model"):
    """Total tokens consumed per model, over finished rows."""

    ax = _axis(ax)
    if df.empty:
        return _no_data(ax)
    _require(df, "model_name")

    tokens = _tokens(df)
    keep = pd.Series(True, index=df.index)
    if "status" in df.columns:
        keep = df["status"].eq(SUCCESS_STATUS).fillna(False)
    if not keep.any():
        return _no_data(ax, "No finished calls")

    work = pd.DataFrame({MODEL_LABEL: _model_label(df)[keep], "tokens": tokens[keep]})
    totals = work.groupby(MODEL_LABEL, dropna=False)["tokens"].sum().reset_index()

    sns.barplot(data=totals, y=MODEL_LABEL, x="tokens", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("tokens (sum over finished calls)")

    return ax


def plot_failures_by_model(df: pd.DataFrame, ax=None, *, title: str = "Failure rate by model"):
    """Share of attempted calls that errored, one bar per model."""

    ax = _axis(ax)
    if df.empty:
        return _no_data(ax)
    _require(df, "model_name")

    text = df["status"].astype("object").fillna("").astype(str) if "status" in df.columns else None
    if text is None:
        return _no_data(ax, "No status column")

    failed = text.str.startswith(ERROR_PREFIX, na=False)
    attempted = failed | text.eq(SUCCESS_STATUS)
    if not attempted.any():
        return _no_data(ax, "No attempted calls")

    work = pd.DataFrame({MODEL_LABEL: _model_label(df), "failed": failed, "attempted": attempted})
    grouped = work.groupby(MODEL_LABEL, dropna=False)[["failed", "attempted"]].sum().reset_index()
    grouped["failure_rate"] = grouped["failed"] / grouped["attempted"].where(grouped["attempted"] > 0)
    grouped["failure_rate"] = grouped["failure_rate"].fillna(0.0)

    sns.barplot(data=grouped, y=MODEL_LABEL, x="failure_rate", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("failed / attempted")
    ax.set_xlim(0, 1)

    return ax


def plot_results_by_search_engine(df: pd.DataFrame, ax=None, *, title: str = "Results by search engine"):
    """Publication counts per ``entry_source`` - the engine that found the row."""

    ax = _axis(ax)
    if df.empty:
        return _no_data(ax)
    _require(df, "entry_source")

    counts = df["entry_source"].astype(str).value_counts().reset_index()
    counts.columns = ["entry_source", "results"]

    sns.barplot(data=counts, y="entry_source", x="results", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("publications")
    ax.set_ylabel("search engine")

    return ax


def plot_flattened_values(df: pd.DataFrame, summary: pd.DataFrame, *, max_columns: int = 8, ncols: int = 3):
    """Bar-chart every answer column worth charting, one panel each.

    ``df`` is the flattened answers and ``summary`` is what
    ``stats.value_counts_summary`` made of them - the ``plottable`` flag there
    decides what gets a panel, so the rule lives in one place instead of being
    re-derived here.

    Unlike the other functions in this module this one is inherently multi-panel,
    so it returns the :class:`~matplotlib.figure.Figure` rather than a single
    ``Axes``. It still neither shows nor saves anything.
    """

    if summary is None or summary.empty or "plottable" not in summary.columns:
        figure, ax = plt.subplots(figsize=FIGSIZE)
        _no_data(ax, "No answer columns to chart")
        return figure

    names = [name for name in summary.loc[summary["plottable"], "column"] if name in df.columns][:max_columns]
    if not names:
        figure, ax = plt.subplots(figsize=FIGSIZE)
        _no_data(ax, "No yes/no or countable answers to chart")
        return figure

    ncols = max(1, min(ncols, len(names)))
    nrows = -(-len(names) // ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2), squeeze=False)

    for axis, name in zip(figure.axes, names):
        counts = df[name].dropna().astype(str).value_counts()
        # Ratings read in their own order; labels read most-common first.
        kind = summary.loc[summary["column"] == name, "kind"].iloc[0]
        counts = counts.sort_index() if kind == "countable" else counts
        sns.barplot(x=counts.index.astype(str), y=counts.to_numpy(), ax=axis)
        axis.set_title(name)
        axis.set_xlabel("")
        axis.set_ylabel("answers")
        if max(len(str(v)) for v in counts.index) > 8:
            axis.tick_params(axis="x", rotation=30)

    # Blank out the unused cells of the last row.
    for axis in figure.axes[len(names) :]:
        axis.set_visible(False)

    figure.tight_layout()

    return figure
