"""Base methods and feature to be used in search CLI command.

Engine-specific API callers live in sibling modules (``search_semantic_scholar`` and
``search_openalex``); this module holds the shared plumbing: the CSV schema, dedup/merge,
sync, YAML I/O, language detection and the helpers used to combine results across engines.
"""

import os
import uuid
from pathlib import Path

import pandas as pd
import yaml

from llmexer.exceptions import LLMExerException
from llmexer.logger import get_logger

logger = get_logger()

# Default values
DEFAULT_SEARCH_YEAR_PARAM = "2020-2025"
DEFAULT_OPEN_ACCESS_PARAM = False

# Per-request HTTP timeout for search-engine calls (seconds); mirrors base/papers.py.
SEARCH_HTTP_TIMEOUT = 30

_PAPER_CSV_COLUMNS = [
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
]


def detect_publication_lang(title: str | None, abstract: str | None) -> str:
    """Detect language from title and abstract separately.
    Returns 'unclear' if they disagree or on failure or missing text."""
    from langdetect import LangDetectException, detect

    try:
        if title and str(title).strip() and abstract and str(abstract).strip():
            lang_title = detect(str(title).strip())
            lang_abstract = detect(str(abstract).strip())
            return lang_title if lang_title == lang_abstract else "unclear"
        text = " ".join(filter(None, [title, abstract])).strip()
        if not text:
            return "unclear"
        return detect(text)
    except LangDetectException:
        return "unclear"


def synf_df_runs_of_search_and_papers(
    df: pd.DataFrame, papers_path: str, add_new_rows: bool = True
) -> tuple[pd.DataFrame, int, int]:
    """Sync pdf_downloaded, txt_filename, markdown_filename for existing rows and
    append new rows for PDFs found in papers_path that are not yet in the DataFrame.

    When ``add_new_rows`` is ``False`` only the files listed in existing rows are checked;
    PDFs present in ``papers_path`` but not already listed are ignored (``added_count`` is ``0``).

    Returns:
        (updated_df, updated_count, added_count)
    """
    papers_dir = Path(papers_path)
    updated_count = 0

    # Ensure string columns stay as object dtype even when all values are NaN
    for col in ("txt_filename", "markdown_filename", "pdf_filename", "entry_source"):
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)

    known_pdf_filenames = set(
        df["pdf_filename"].dropna().astype(str).str.strip().loc[lambda s: s != ""]
    )

    for idx, row in df.iterrows():
        pdf_filename = str(row.get("pdf_filename", "") or "").strip()
        if not pdf_filename:
            continue
        stem = (
            pdf_filename[:-4] if pdf_filename.lower().endswith(".pdf") else pdf_filename
        )

        changed = False
        if (papers_dir / pdf_filename).exists():
            if not df.at[idx, "pdf_downloaded"]:
                df.at[idx, "pdf_downloaded"] = True
                changed = True

        txt_val = row.get("txt_filename", "")
        txt_empty = pd.isna(txt_val) or str(txt_val).strip() == ""
        txt_candidate = f"{stem}.txt"
        if txt_empty and (papers_dir / txt_candidate).exists():
            df.at[idx, "txt_filename"] = txt_candidate
            changed = True

        md_val = row.get("markdown_filename", "")
        md_empty = pd.isna(md_val) or str(md_val).strip() == ""
        md_candidate = f"{stem}.md"
        if md_empty and (papers_dir / md_candidate).exists():
            df.at[idx, "markdown_filename"] = md_candidate
            changed = True

        if changed:
            updated_count += 1

    new_rows = []
    if add_new_rows and papers_dir.exists():
        for pdf_path in sorted(papers_dir.glob("*.pdf")):
            pdf_name = pdf_path.name
            if pdf_name in known_pdf_filenames:
                continue
            stem = pdf_path.stem
            txt_exists = (papers_dir / f"{stem}.txt").exists()
            md_exists = (papers_dir / f"{stem}.md").exists()
            new_row = {col: "" for col in _PAPER_CSV_COLUMNS}
            new_row["pdf_filename"] = pdf_name
            new_row["pdf_downloaded"] = True
            new_row["entry_source"] = "manually added"
            new_row["txt_filename"] = f"{stem}.txt" if txt_exists else ""
            new_row["markdown_filename"] = f"{stem}.md" if md_exists else ""
            new_rows.append(new_row)

    added_count = len(new_rows)
    if new_rows:
        df = pd.concat(
            [df, pd.DataFrame(new_rows, columns=_PAPER_CSV_COLUMNS)], ignore_index=True
        )

    return df, updated_count, added_count


# Marker present in every merged output filename, used to exclude those files from re-merging.
_MERGED_MARKER = "__merged"
_RESULTS_STEM_SUFFIX = "__results"
_FILTERED_STEM_SUFFIX = "__filtered"
_DUPLICATES_COUNTER_COLUMN = "duplicates_counter"


def _normalize_text(value) -> str:
    """Lowercase, collapse internal whitespace and strip. Empty string for missing/blank."""
    if _is_missing(value):
        return ""
    return " ".join(str(value).split()).strip().lower()


def _is_missing(value) -> bool:
    """True for None, NaN or blank/whitespace-only values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == "" or str(value).strip().lower() == "nan"


def _dedup_key(row) -> str | None:
    """DOI-based dedup key when present, else a normalized-title key. None if neither exists."""
    doi = _normalize_text(row.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = _normalize_text(row.get("title"))
    if title:
        return f"title:{title}"
    return None


def _source_column_name(filename: str, stem_suffix: str) -> str:
    """Search id (the YAML stem) for a source CSV: file stem with ``stem_suffix`` stripped.

    e.g. ``20260626-name__results.csv`` -> ``20260626-name`` (its ``20260626-name.yaml``).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem_suffix and stem.endswith(stem_suffix):
        stem = stem[: -len(stem_suffix)]
    return stem


def _merge_metadata(metadata: dict, row) -> None:
    """Fill each metadata field with the first non-missing value seen across duplicates."""
    for col in _PAPER_CSV_COLUMNS:
        if not _is_missing(metadata.get(col)):
            continue
        value = row.get(col)
        if not _is_missing(value):
            metadata[col] = value


def gather_search_csvs(searches_path: str) -> tuple[list, list]:
    """Return ``(results_csvs, filtered_csvs)`` for a searches directory.

    Excludes any previously produced merged outputs (files containing ``__merged``).
    """
    if not os.path.isdir(searches_path):
        raise LLMExerException(f"No searches directory found: {searches_path}")

    def _collect(suffix):
        return sorted(
            p
            for p in Path(searches_path).glob(f"*{suffix}.csv")
            if _MERGED_MARKER not in p.name
        )

    return _collect(_RESULTS_STEM_SUFFIX), _collect(_FILTERED_STEM_SUFFIX)


def merge_search_csvs(csv_paths, stem_suffix: str) -> tuple[pd.DataFrame, list[str]]:
    """Merge the given search CSVs into one deduplicated DataFrame.

    Publications are deduplicated by DOI, falling back to a normalized title when the DOI is
    missing. Each source file becomes a binary column named after its search (the YAML stem,
    i.e. the filename with ``stem_suffix`` stripped): ``1`` if the publication appears in that
    search, else ``0``. ``duplicates_counter`` holds the number of duplicate occurrences of
    each publication, i.e. one less than the number of searches it was found in (``0`` for a
    publication found in a single search).

    Rows are sorted by ``year`` descending (newest first); rows with a missing/blank year go last.

    Returns:
        (merged_df, run_columns) where run_columns is the sorted list of per-search columns.
    """
    run_columns: list[str] = []
    merged: dict[str, dict] = {}

    for csv_path in csv_paths:
        column = _source_column_name(Path(csv_path).name, stem_suffix)
        if column not in run_columns:
            run_columns.append(column)
        df = pd.read_csv(csv_path, sep=";")
        for _, row in df.iterrows():
            key = _dedup_key(row)
            if key is None:
                # No DOI and no title: keep as a distinct entry.
                key = f"row:{Path(csv_path).name}:{len(merged)}"
            entry = merged.get(key)
            if entry is None:
                entry = {
                    "metadata": {col: None for col in _PAPER_CSV_COLUMNS},
                    "runs": set(),
                }
                merged[key] = entry
            _merge_metadata(entry["metadata"], row)
            entry["runs"].add(column)

    run_columns = sorted(run_columns)
    records = []
    for entry in merged.values():
        record = dict(entry["metadata"])
        runs = entry["runs"]
        record[_DUPLICATES_COUNTER_COLUMN] = len(runs) - 1
        for column in run_columns:
            record[column] = 1 if column in runs else 0
        records.append(record)

    columns = _PAPER_CSV_COLUMNS + [_DUPLICATES_COUNTER_COLUMN] + run_columns
    merged_df = pd.DataFrame(records, columns=columns)
    merged_df = merged_df.sort_values(
        by="year",
        key=lambda s: pd.to_numeric(s, errors="coerce"),
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    return merged_df, run_columns


def generate_search_id() -> str:
    """
    Generate a unique experiment ID formatted as 'YYYYMMDD-GUID'

    Returns:
      str: A string in the format 'YYYYMMDD-UUID'.
    """
    from datetime import datetime, timezone

    now_utc = datetime.now(timezone.utc)
    formatted_datetime = now_utc.strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8]
    return f"{formatted_datetime}-{unique_id}"


def save_search_query(
    experiment_path: str,
    query: str,
    year: str = DEFAULT_SEARCH_YEAR_PARAM,
    only_open_access: bool = DEFAULT_OPEN_ACCESS_PARAM,
) -> tuple[str, str]:
    """Save a search query to a YAML config file in the experiment's searches directory.

    Returns:
        (search_id, yaml_filename)
    """
    searches_path = os.path.join(experiment_path, "searches")
    ensure_directory_exists(searches_path)
    search_id = generate_search_id()

    yaml_filename = f"{search_id}.yaml"
    yaml_path = os.path.join(searches_path, yaml_filename)

    search_config = {"query": query, "year": year, "onlyOpenAccess": only_open_access}

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(search_config, f, default_flow_style=False, sort_keys=False)

    return search_id, yaml_filename


def write_records_to_csv(records: list[dict], csv_path: str) -> None:
    """Write flattened paper records to ``csv_path`` as a semicolon-separated UTF-8 CSV.

    Rows are ordered by ``year`` descending then ``title`` ascending, matching the ordering
    produced by the individual search engines.
    """
    df = pd.DataFrame(records, columns=_PAPER_CSV_COLUMNS)
    df.sort_values(by=["year", "title"], ascending=[False, True]).to_csv(
        csv_path, index=False, encoding="utf-8", sep=";"
    )
    logger.debug("Wrote CSV to '%s'", csv_path)


def combine_new_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """Append only the records from ``new`` not already present in ``existing``.

    Presence is determined by :func:`_dedup_key` (DOI when available, otherwise a normalized
    title). Records without any key (no DOI and no title) are always kept.
    """
    seen = {key for key in (_dedup_key(row) for row in existing) if key is not None}
    combined = list(existing)
    for row in new:
        key = _dedup_key(row)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        combined.append(row)
    return combined


def read_search_params(
    file: str, experiment_path: str, query_default: str = None
) -> tuple[str, str, str, bool]:
    """Read search parameters from a YAML config file.

    Returns:
        (search_id, query, year, only_open_access)
    """
    from llmexer.constants import SEARCHES_DIR

    file_stem = os.path.splitext(os.path.basename(file))[0]
    search_id = file_stem

    if os.path.isabs(file):
        search_file_path = file
    else:
        search_file_path = os.path.join(experiment_path, SEARCHES_DIR, file)

    if not os.path.exists(search_file_path):
        raise LLMExerException(f"Search file does not exist: '{file}' ")

    with open(search_file_path, "r", encoding="utf-8") as f:
        search_params = yaml.safe_load(f)

    query = search_params.get("query", query_default)
    year = search_params.get("year", DEFAULT_SEARCH_YEAR_PARAM)
    only_open_access = search_params.get("onlyOpenAccess", DEFAULT_OPEN_ACCESS_PARAM)

    return search_id, query, year, only_open_access
