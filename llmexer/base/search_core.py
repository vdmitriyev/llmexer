"""CORE search engine: the CORE API v3 caller for ``search run``.

CORE (https://core.ac.uk) aggregates open-access research outputs from repositories and
journals. It needs a free API key, ``CORE_API_KEY``; like OpenAlex, CORE is skipped
without one. The query is searched in title and abstract.
"""

import os
from typing import Callable, Optional

from llmexer.base.search import SEARCH_HTTP_TIMEOUT
from llmexer.base.search_query import to_core_query
from llmexer.base.search_record import (
    announce_query,
    build_record,
    cap_message,
    effective_limit,
    parse_year_range,
    retrying_session,
    save_raw_pages,
)
from llmexer.constants import DEFAULT_CORE_MAX_RESPONSES
from llmexer.logger import get_logger

logger = get_logger()

ENTRY_SOURCE_CORE = "CORE"
CORE_API_KEY_ENV_VAR = "CORE_API_KEY"
CORE_MAX_RESPONSES_ENV_VAR = "CORE_MAX_RESPONSES"

_CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"
_CORE_MAX_PAGE_SIZE = 100
# Offset paging stops here (the search index's result window).
CORE_MAX_RECORDS = 10000


def build_core_query(query: str, year: str | None, custom_query: str | None = None) -> str:
    """The CORE ``q``: the terms in title or abstract, plus the publication-year range."""

    parts = []
    core_query = custom_query or (to_core_query(query) if query else "")
    if core_query:
        parts.append(f"({core_query})")
    years = parse_year_range(year)
    if years:
        parts.append(f"yearPublished>={years[0]} AND yearPublished<={years[1]}")
    return " AND ".join(parts)


def _authors(work: dict) -> list[str]:
    names = []
    for author in work.get("authors") or []:
        name = author.get("name") if isinstance(author, dict) else author
        if name and str(name).strip():
            names.append(" ".join(str(name).split()))
    return names


def _year(work: dict):
    for value in (work.get("yearPublished"), (work.get("publishedDate") or "")[:4]):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _doi(value) -> str | None:
    if not value:
        return None
    doi = str(value).strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if doi.lower().startswith(prefix):
            return doi[len(prefix) :]
    return doi


def core_record(work: dict) -> dict:
    """One CORE work as a row of the search CSV."""

    return build_record(
        engine_label=ENTRY_SOURCE_CORE,
        internal_id=work.get("id"),
        year=_year(work),
        title=work.get("title"),
        authors=_authors(work),
        abstract=work.get("abstract"),
        doi=_doi(work.get("doi")),
        is_open_access=True if work.get("downloadUrl") else None,
        citation_count=work.get("citationCount"),
    )


def run_core_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    raw_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
    custom_query: str | None = None,
) -> list[dict]:
    """Page through CORE with ``offset`` / ``limit`` until ``--limit`` (or the cap) is reached.

    With ``only_open_access``, works without a ``downloadUrl`` (full text) are dropped.
    """

    session = retrying_session()
    session.headers.update({"Authorization": f"Bearer {os.getenv(CORE_API_KEY_ENV_VAR, '')}"})
    core_query = build_core_query(query, year, custom_query)
    announce_query(ENTRY_SOURCE_CORE, core_query, bool(custom_query), on_progress)

    params = {"q": core_query, "limit": min(batch_size, _CORE_MAX_PAGE_SIZE), "offset": 0}
    limit = min(effective_limit(limit_size, CORE_MAX_RESPONSES_ENV_VAR, DEFAULT_CORE_MAX_RESPONSES), CORE_MAX_RECORDS)
    pages, records = [], []
    page = 0
    while True:
        response = session.get(_CORE_SEARCH_URL, params=dict(params), timeout=SEARCH_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        pages.append(data)
        save_raw_pages(raw_path, pages)
        page += 1
        total = data.get("totalHits")

        works = data.get("results") or []
        for work in works:
            if len(records) >= limit:
                break
            if only_open_access and not work.get("downloadUrl"):
                continue
            records.append(core_record(work))

        if on_progress is not None:
            on_progress(f"CORE: page {page} — {len(records)}/{total} fetched")
        if page == 1:
            warning = cap_message(total, limit_size, "CORE", CORE_MAX_RESPONSES_ENV_VAR, DEFAULT_CORE_MAX_RESPONSES)
            if total is not None and total > CORE_MAX_RECORDS and limit == CORE_MAX_RECORDS:
                warning = f"CORE: {total} result(s), but offset paging stops at {CORE_MAX_RECORDS}. Narrow the query."
            if warning:
                logger.warning(warning)
                if on_progress is not None:
                    on_progress(warning)

        params["offset"] += params["limit"]
        if (
            len(records) >= limit
            or not works
            or params["offset"] >= CORE_MAX_RECORDS
            or (total is not None and params["offset"] >= total)
        ):
            break

    logger.info("CORE total retrieved: %d record(s)", len(records))
    return records
