"""Load a project's data and reshape it for analysis.

Copied into a project's ``analysis/`` folder by ``llmexer analysis init`` and
imported there as a top-level module, so this file deliberately imports nothing
from ``llmexer`` and nothing from its sibling analysis modules.

Two consequences of that rule, both intentional:

* :func:`strip_code_fence` is a verbatim copy of ``llmexer.common.strip_code_fence``.
  ``tests/test_analysis_transform.py`` asserts the two stay in step.
* The exceptions raised here are local. :class:`MissingColumnError` subclasses
  ``KeyError`` so ``except KeyError`` catches it without importing anything.
"""

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

# Column added by :func:`flatten` carrying the per-row parse failure ("" when fine).
FLATTEN_ERROR_COLUMN = "flatten_error"

# Suffix given to a flattened column whose name is already taken by the frame.
COLLISION_SUFFIX = "_flat"

# Column holding a top-level JSON array, which has no sensible column mapping.
ITEMS_SUFFIX = "_items"

# `df.attrs` key under which :func:`flatten_llm_response` records the columns it
# added, so :func:`flattened_only` can hand back just the model's answer.
FLATTENED_COLUMNS_KEY = "llmexer_flattened_columns"

# `experiment generate` writes one pair of tables per provider.
EXPERIMENT_TABLE_PREFIX = "experiment_"
PARAMS_TABLE_PREFIX = "params_"

# Both halves of the pair are keyed by this pair of columns.
PARAMS_JOIN_KEY = ["params_code", "profile_name"]

# Provider names are interpolated into table names, so they are whitelisted first.
_PROVIDER_RE = re.compile(r"^[a-z0-9_]+$")

# CSV dialect used across the project.
CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8"

# Columns an empty result still carries, so a notebook run against a project with
# nothing generated (or nothing searched) yet reaches the end instead of dying on
# a KeyError three cells later.
EXPERIMENT_COLUMNS = (
    "ID",
    "code",
    "prompt",
    "original_data",
    "model_name",
    "provider_name",
    "profile_name",
    "response_text",
    "status",
    "state",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "elapsed_seconds",
    "timestamp",
    "response_json",
    "_provider",
)

SEARCH_COLUMNS = (
    "search_engine_internal_id",
    "year",
    "title",
    "authors",
    "abstract",
    "isOpenAccess",
    "doi",
    "language",
    "citationCount",
    "referenceCount",
    "entry_source",
    "pdf_filename",
    "txt_filename",
    "markdown_filename",
    "pdf_downloaded",
    "search_id",
    "source_file",
)

PAPER_COLUMNS = ("stem", "pdf", "txt", "markdown", "extracted")


class MissingColumnError(KeyError):
    """Raised when the column to flatten is not in the frame."""


def strip_code_fence(text: str) -> str:
    """Drop a Markdown code fence wrapping a value, if there is one.

    Models routinely answer with their JSON inside ```` ```json ... ``` ````.
    Left in place that prefix makes every such answer unparseable, so the fence
    is peeled off before parsing - the text itself is never modified otherwise.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # First line is the fence, optionally carrying a language tag ("```json").
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def parse_json_payload(value):
    """Parse one cell into a Python object.

    Returns ``(parsed, error)``. ``error`` is ``""`` when there was nothing to
    parse or the parse succeeded, so an empty response is not reported as a
    failure - plenty of rows simply have not been run yet.
    """

    if value is None:
        return None, ""
    # A dict/list is already parsed (a caller may have done the work upstream).
    if isinstance(value, (dict, list)):
        return value, ""
    # NaN and friends: scalar-only check, `pd.isna` on a list returns an array.
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None, ""
        except (TypeError, ValueError):
            pass
        return None, f"unsupported type: {type(value).__name__}"

    candidate = strip_code_fence(value)
    if not candidate:
        return None, ""

    try:
        return json.loads(candidate), ""
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid JSON: {exc}"


def flatten_llm_response(
    df: pd.DataFrame,
    column: str = "response_text",
    *,
    prefix: str = "",
    sep: str = ".",
    errors: str = "column",
) -> pd.DataFrame:
    """Parse ``column`` as JSON and expand it into flat columns.

    Every row of ``df`` is preserved, in order and with its original index: a
    malformed answer never raises and never drops a row, it simply gets ``NaN``
    in the flattened fields. Nested objects become dotted column names; a list
    nested inside an object stays a list in its cell; a JSON array at the top
    level has no column mapping at all, so it is kept whole in a
    ``<column>_items`` column rather than exploding the frame.

    ``errors`` is ``"column"`` (default - always add ``flatten_error``),
    ``"raise"`` (the first malformed row raises) or ``"ignore"`` (no column).

    A flattened name that collides with an existing column of ``df`` is renamed
    to ``<name>_flat`` rather than overwriting it: a model answering
    ``{"status": ...}`` must not clobber the experiment's own ``status``.
    """

    if column not in df.columns:
        raise MissingColumnError(f"column '{column}' is not in the frame; available columns: {sorted(df.columns)}")
    if errors not in {"column", "raise", "ignore"}:
        raise ValueError(f"errors must be 'column', 'raise' or 'ignore', got {errors!r}")

    records: list[dict] = []
    messages: list[str] = []
    items: list = []
    has_items = False

    for value in df[column]:
        parsed, message = parse_json_payload(value)
        if message and errors == "raise":
            raise ValueError(f"failed to parse {column!r}: {message}")
        messages.append(message)

        if isinstance(parsed, dict):
            records.append(parsed)
            items.append(None)
        elif parsed is None:
            records.append({})
            items.append(None)
        else:
            # A top-level array or scalar: keep it whole, expand nothing.
            records.append({})
            items.append(parsed)
            has_items = True

    flat = pd.json_normalize(records, sep=sep)
    if prefix:
        flat.columns = [f"{prefix}{name}" for name in flat.columns]
    # `json_normalize` returns a fresh RangeIndex; joining that onto a frame with
    # any other index silently yields all-NaN, so realign before joining.
    flat.index = df.index

    taken = set(df.columns)
    flat = flat.rename(columns={name: f"{name}{COLLISION_SUFFIX}" for name in flat.columns if name in taken})

    result = df.join(flat)

    if has_items:
        items_column = f"{prefix}{column}{ITEMS_SUFFIX}"
        result[items_column] = pd.Series(items, index=df.index, dtype="object")

    if errors == "column":
        result[FLATTEN_ERROR_COLUMN] = pd.Series(messages, index=df.index, dtype="object")

    # Remember what came out of the answer, so the values can be separated from
    # the experiment row they arrived on without re-deriving the difference.
    result.attrs[FLATTENED_COLUMNS_KEY] = [name for name in result.columns if name not in df.columns]

    return result


def flattened_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the columns :func:`flatten_llm_response` produced.

    The flattened frame keeps the experiment row next to the parsed answer, which
    is what the cross-model comparisons need. The answer on its own is a
    different thing - a table of what the models actually said - and that is what
    gets exported, without the prompt, the raw payload and the identity columns
    around it.

    ``flatten_error`` is left out: it describes the parse, not the answer.
    """

    names = [
        name for name in df.attrs.get(FLATTENED_COLUMNS_KEY, []) if name != FLATTEN_ERROR_COLUMN and name in df.columns
    ]

    return df[names].copy()


def export_as_csv(df: pd.DataFrame, path) -> Path:
    """Write ``df`` to ``path`` as CSV and return the path.

    Semicolon-separated UTF-8 without the index, the dialect every other CSV in a
    project uses, so the export opens the same way as ``data.csv`` and the search
    results next to it. Parent directories are created.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)

    return target


def empty_frame(columns) -> pd.DataFrame:
    """An empty frame carrying ``columns``, so callers never branch on shape."""

    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def _provider_of(table_name: str) -> str:
    """The provider a generated experiment table belongs to."""

    return table_name[len(EXPERIMENT_TABLE_PREFIX) :]


def _experiment_tables(connection) -> list:
    """Return ``(table, provider)`` for every generated experiment table.

    ``try_experiment_<provider>`` does not start with ``experiment_``, so the
    per-try tables stay out of the analysis, exactly as they stay out of
    ``experiment run`` and ``experiment stats``.
    """

    rows = connection.execute("select name from sqlite_master where type = 'table'").fetchall()
    names = sorted(name for (name,) in rows if name.startswith(EXPERIMENT_TABLE_PREFIX))

    return [(name, _provider_of(name)) for name in names if _PROVIDER_RE.match(_provider_of(name))]


def load_experiment_db(db_path) -> pd.DataFrame:
    """Load every generated row of an experiment database into one frame.

    Each ``experiment_<provider>`` table is LEFT-joined to its
    ``params_<provider>`` on ``(params_code, profile_name)``, so the
    hyperparameters a row ran under sit next to its result, and ``_provider``
    identifies the table the row came from.

    ``db_path`` may be ``None`` - a project with nothing generated yet yields an
    empty frame rather than an error, so the notebook still runs top to bottom. A
    path that *is* given but does not exist raises, because that is a typo.
    """

    if db_path is None:
        return empty_frame(EXPERIMENT_COLUMNS)

    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment database not found: '{path}'.")

    frames = []
    with sqlite3.connect(path) as connection:
        for table, provider in _experiment_tables(connection):
            # Table names are interpolated, so every provider is whitelisted
            # against _PROVIDER_RE in `_experiment_tables` before it gets here.
            experiment = pd.read_sql(f"select * from {table}", connection)  # nosec B608
            params_table = f"{PARAMS_TABLE_PREFIX}{provider}"
            try:
                params = pd.read_sql(f"select * from {params_table}", connection)  # nosec B608
            except (pd.errors.DatabaseError, sqlite3.DatabaseError) as exc:
                raise ValueError(
                    f"'{path.name}' has '{table}' but no '{params_table}'. It was generated by an "
                    "older llmexer - re-run `llmexer experiment generate`."
                ) from exc

            joined = experiment.merge(params, how="left", on=PARAMS_JOIN_KEY)
            joined["_provider"] = provider
            if not joined.empty:
                frames.append(joined)

    if not frames:
        return empty_frame(EXPERIMENT_COLUMNS)

    # pandas 3 warns when empty frames take part in a concat, hence the filter above.
    df = pd.concat(frames, ignore_index=True)
    if "ID" in df.columns:
        df = df.sort_values("ID").reset_index(drop=True)

    # Stored as INTEGER NULL: kept nullable so a provider that reported no
    # split stays <NA> instead of turning into a float NaN or a zero.
    for name in ("prompt_tokens", "completion_tokens"):
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")

    return df


def load_search_frame(searches_dir, search, *, kind: str = "results") -> pd.DataFrame:
    """Load the CSV of **one** search into a frame.

    ``search`` is a single entry of the notebook's ``SEARCHES`` list: the
    ``search`` YAML name plus the ``results`` / ``filter`` CSV names, either of
    which may be ``None`` when that file does not exist yet. ``kind`` picks which
    of the two to read.

    One search at a time is deliberate. Two searches are two different queries,
    so their rows are different populations - concatenating them would produce
    year histograms and open-access shares that describe no actual search. Point
    ``SEARCH_INDEX`` at another entry to analyse another one.

    Returns an empty frame carrying :data:`SEARCH_COLUMNS` when the search has no
    such CSV, so the notebook still runs to the end.
    """

    if kind not in {"results", "filter"}:
        raise ValueError(f"kind must be 'results' or 'filter', got {kind!r}")

    filename = (search or {}).get(kind)
    if not filename:
        return empty_frame(SEARCH_COLUMNS)

    path = Path(searches_dir) / filename
    if not path.is_file():
        return empty_frame(SEARCH_COLUMNS)

    # `dtype=str` + `keep_default_na=False` mirrors `search export`, so a DOI
    # spelled "NA" survives and every text column stays text.
    df = pd.read_csv(path, sep=CSV_SEPARATOR, dtype=str, keep_default_na=False)
    if "sem_scholar_paper_id" in df.columns:
        # Pre-OpenAlex CSVs name the column after Semantic Scholar alone.
        df = df.rename(columns={"sem_scholar_paper_id": "search_engine_internal_id"})
    if df.empty:
        return empty_frame(SEARCH_COLUMNS)

    df["search_id"] = Path((search or {}).get("search") or filename).stem.split("__")[0]
    df["source_file"] = path.name
    for name in ("year", "citationCount", "referenceCount"):
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")

    return df


def list_papers(papers_dir) -> pd.DataFrame:
    """Inventory a project's ``papers/`` folder, one row per document stem."""

    folder = Path(papers_dir)
    if not folder.is_dir():
        return empty_frame(PAPER_COLUMNS)

    found: dict = {}
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".txt", ".md"}:
            continue
        entry = found.setdefault(path.stem, {"stem": path.stem, "pdf": False, "txt": False, "markdown": False})
        entry["pdf"] = entry["pdf"] or suffix == ".pdf"
        entry["txt"] = entry["txt"] or suffix == ".txt"
        entry["markdown"] = entry["markdown"] or suffix == ".md"

    if not found:
        return empty_frame(PAPER_COLUMNS)

    df = pd.DataFrame(sorted(found.values(), key=lambda row: row["stem"]))
    df["extracted"] = df["txt"] | df["markdown"]

    return df
