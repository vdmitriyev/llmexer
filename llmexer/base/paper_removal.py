"""Find what `papers remove` / `search remove` deletes, before anything is changed.

The rows themselves are dropped by ``_filter_search()`` (``commands/search.py``), the same code
``search filter`` uses, so a removal is written and logged like any other exclusion.
"""

import os
from pathlib import Path

import pandas as pd

from llmexer.base.papers import scan_papers
from llmexer.base.search import (
    _RESULTS_STEM_SUFFIX,
    _is_missing,
    _normalize_text,
    _source_column_name,
    gather_search_csvs,
)
from llmexer.constants import PAPERS_DIR, SEARCHES_DIR
from llmexer.exceptions import PaperRemoveException, UnexpectedCLIParamsException


def removal_filter(dois: list[str], names: list[str]):
    """Return a ``(description, keep_mask)`` pair in the shape ``_filter_search()`` takes."""

    wanted_dois = {_normalize_text(doi) for doi in dois}
    wanted_names = set(names)
    parts = [f"doi={doi}" for doi in dois] + [f"name={name}" for name in names]
    description = ", ".join(parts)

    def keep_mask(df: pd.DataFrame) -> pd.Series:
        return ~_removal_mask(df, wanted_dois, wanted_names)

    return description, keep_mask


def _pdf_names(df: pd.DataFrame) -> pd.Series:
    return df["pdf_filename"].map(lambda value: "" if _is_missing(value) else str(value).strip())


def _removal_mask(df: pd.DataFrame, wanted_dois: set[str], wanted_names: set[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "doi" in df.columns:
        mask |= df["doi"].map(_normalize_text).isin(wanted_dois)
    if "pdf_filename" in df.columns:
        mask |= _pdf_names(df).isin(wanted_names)
    return mask


def find_removal_targets(experiment_path: str, dois: list[str], names: list[str]) -> tuple[list[Path], list[str]]:
    """Return the files to delete and the ids of the searches that lose a row. Changes nothing.

    Raises ``PaperRemoveException`` when a DOI or name matches no file and no row.
    """

    for name in names:
        if os.path.basename(name) != name or name in ("", ".", ".."):
            raise UnexpectedCLIParamsException(f"--name must be a file name in papers/, not a path: '{name}'.")

    wanted_dois = {_normalize_text(doi) for doi in dois}
    wanted_names = set(names)
    found_dois: set[str] = set()
    found_names: set[str] = set()
    pdf_names: set[str] = set()
    search_ids: list[str] = []

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    results_csvs, filtered_csvs = gather_search_csvs(searches_path) if os.path.isdir(searches_path) else ([], [])

    for csv_path in results_csvs + filtered_csvs:
        df = pd.read_csv(csv_path, sep=";")
        matched = df[_removal_mask(df, wanted_dois, wanted_names)]
        if matched.empty:
            continue
        found_dois.update(matched["doi"].map(_normalize_text))
        matched_pdfs = _pdf_names(matched)
        found_names.update(matched_pdfs)
        pdf_names.update(value for value in matched_pdfs if value)

    # A search loses a row when the file `_filter_search()` reads from has a match. Like
    # `search filter`, only searches with a YAML file are filtered.
    for results_csv in results_csvs:
        search_id = _source_column_name(results_csv, _RESULTS_STEM_SUFFIX)
        if not (Path(searches_path) / f"{search_id}.yaml").is_file():
            continue
        filtered_csv = Path(searches_path) / f"{search_id}__filtered.csv"
        source = filtered_csv if filtered_csv.is_file() else results_csv
        df = pd.read_csv(source, sep=";")
        if _removal_mask(df, wanted_dois, wanted_names).any():
            search_ids.append(search_id)

    papers_path = os.path.join(experiment_path, PAPERS_DIR)
    papers_by_stem = {entry.stem: entry for entry in scan_papers(papers_path, only_paper_types=True)}
    for name in names:
        if os.path.isfile(os.path.join(papers_path, name)):
            found_names.add(name)
            pdf_names.add(name)

    files: list[Path] = []
    for filename in sorted(pdf_names):
        entry = papers_by_stem.get(Path(filename).stem)
        if entry is None or Path(filename).suffix.lower().lstrip(".") not in entry.files:
            continue
        # A PDF takes the text extracted from it along; any other file goes alone.
        if filename.lower().endswith(".pdf"):
            files.extend(entry.files.values())
        else:
            files.append(Path(papers_path) / filename)

    unmatched = [doi for doi in dois if _normalize_text(doi) not in found_dois]
    unmatched += [name for name in names if name not in found_names]
    if unmatched:
        raise PaperRemoveException(
            f"Nothing to remove for: {', '.join(repr(value) for value in unmatched)}. "
            "No file in papers/ and no search result row matches."
        )

    return files, search_ids
