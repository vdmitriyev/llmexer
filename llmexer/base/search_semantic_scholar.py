"""Semantic Scholar search engine: the Bulk API caller for ``search run``."""

import json
from typing import Callable, Optional

import pandas as pd
from requests.adapters import HTTPAdapter, Retry

from llmexer.base.papers import get_first_author_last_name, make_structured_filename
from llmexer.base.search import (
    _PAPER_CSV_COLUMNS,
    SEARCH_HTTP_TIMEOUT,
    detect_publication_lang,
)
from llmexer.common import make_http_session
from llmexer.logger import get_logger

logger = get_logger()

ENTRY_SOURCE_SEMANTIC_SCHOLAR = "Semantic Scholar"

# Semantic Scholar API constants
_SEM_SCHOLAR_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_SEM_SCHOLAR_FIELDS = (
    "paperId,title,authors,abstract,isOpenAccess,externalIds,year,"
    "referenceCount,citationCount,fieldsOfStudy,citationStyles,publicationTypes"
)


def run_semantic_scholar_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    json_path: str,
    csv_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Call the Semantic Scholar bulk search API with pagination.

    When ``on_progress`` is provided it is called with a human-readable status message
    after each fetched page, e.g. ``"Semantic Scholar: page 2 — 200/540 fetched"``.

    Returns:
        records: a list of flattened paper dicts with PAPER_CSV_COLUMNS fields.
    """
    session = make_http_session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    params: dict = {
        "query": query,
        "fields": _SEM_SCHOLAR_FIELDS,
        "limit": min(batch_size, 1000),
    }

    if year:
        params["year"] = year
    if only_open_access:
        params["openAccessPdf"] = ""

    raw_json_results: list[dict] = []
    records: list[dict] = []
    page = 0

    while True:
        response = session.get(_SEM_SCHOLAR_BULK_URL, params=params, timeout=SEARCH_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        raw_json_results.append(data)
        page += 1
        total = data.get("total")

        papers = data.get("data", [])
        for paper in papers:
            if limit_size is not None and len(records) >= limit_size:
                break
            ext_ids = paper.get("externalIds") or {}
            pdf_filename = make_structured_filename(
                paper.get("year"),
                get_first_author_last_name(paper),
                paper.get("title"),
                ext_ids.get("DOI"),
            )
            records.append(
                {
                    "search_engine_internal_id": paper.get("paperId"),
                    "year": paper.get("year"),
                    "title": paper.get("title"),
                    "authors": "; ".join(a.get("name", "") for a in paper.get("authors", [])),
                    "abstract": paper.get("abstract"),
                    "isOpenAccess": paper.get("isOpenAccess"),
                    "doi": ext_ids.get("DOI"),
                    "language": detect_publication_lang(paper.get("title"), paper.get("abstract")),
                    "referenceCount": paper.get("referenceCount"),
                    "citationCount": paper.get("citationCount"),
                    "entry_source": ENTRY_SOURCE_SEMANTIC_SCHOLAR,
                    "pdf_filename": pdf_filename,
                    "txt_filename": "",
                    "markdown_filename": "",
                    "pdf_downloaded": False,
                }
            )

        logger.info("Retrieved %d paper(s) so far...", len(records))
        logger.debug("Page retrieved: %d papers in page, %d total", len(papers), len(records))
        if on_progress is not None:
            if total is not None:
                on_progress(f"Semantic Scholar: page {page} — {len(records)}/{total} fetched")
            else:
                on_progress(f"Semantic Scholar: page {page} — {len(records)} fetched")

        token = data.get("token")
        if not token:
            break
        params["token"] = token

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(raw_json_results, fh, ensure_ascii=False, indent=4)

        df = pd.DataFrame(records, columns=_PAPER_CSV_COLUMNS)
        df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw_json_results, fh, ensure_ascii=False, indent=4)
    logger.debug("Wrote raw response to '%s'", json_path)

    df = pd.DataFrame(records, columns=_PAPER_CSV_COLUMNS)
    df.sort_values(by=["year", "title"], ascending=[False, True]).to_csv(
        csv_path, index=False, encoding="utf-8", sep=";"
    )
    logger.debug("Wrote CSV to '%s'", csv_path)

    logger.info("Total retrieved: %d paper(s)", len(records))

    return records
