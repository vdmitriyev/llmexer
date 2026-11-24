"""Zenodo search engine: the Zenodo records API caller for ``search run``.

Zenodo (https://zenodo.org) is an open repository for research outputs of every field.
Only publications are searched, and of those not the project paperwork: data management
plans, deliverables, milestones and proposals. No key is needed; ``ZENODO_API_KEY`` (a
personal access token) allows 100 instead of 25 records per page and a higher rate limit.
"""

import os
import time
from typing import Callable, Optional

from llmexer.base.search import SEARCH_HTTP_TIMEOUT
from llmexer.base.search_query import to_zenodo_query
from llmexer.base.search_record import (
    announce_query,
    build_record,
    cap_message,
    effective_limit,
    parse_year_range,
    retrying_session,
    save_raw_pages,
    strip_html,
)
from llmexer.logger import get_logger

logger = get_logger()

ENTRY_SOURCE_ZENODO = "Zenodo"
ZENODO_API_KEY_ENV_VAR = "ZENODO_API_KEY"
# Publication subtypes left out: project paperwork rather than research results.
ZENODO_EXCLUDED_SUBTYPES = ("datamanagementplan", "deliverable", "milestone", "proposal")

_ZENODO_RECORDS_URL = "https://zenodo.org/api/records"
_PAGE_SIZE_WITHOUT_KEY = 25
_PAGE_SIZE_WITH_KEY = 100
# Zenodo allows 30 requests a minute without a token and 100 with one.
_DELAY_WITHOUT_KEY = 2.1
_DELAY_WITH_KEY = 0.7
# Page-based paging stops here (the search index's result window).
ZENODO_MAX_RECORDS = 10000


def _excluded_types_filter() -> str:
    # `metadata.resource_type.id` is the index field; `resource_type.subtype` is silently ignored.
    ids = " OR ".join(f"publication-{subtype}" for subtype in ZENODO_EXCLUDED_SUBTYPES)
    return f"NOT metadata.resource_type.id:({ids})"


def build_zenodo_query(query: str, year: str | None, only_open_access: bool, custom_query: str | None = None) -> str:
    """The Zenodo ``q``: the terms, the excluded publication types, and the year / open-access filters."""

    terms = custom_query or (to_zenodo_query(query) if query else "")
    parts = [f"({terms})"] if terms else []
    parts.append(_excluded_types_filter())
    years = parse_year_range(year)
    if years:
        parts.append(f"publication_date:[{years[0]}-01-01 TO {years[1]}-12-31]")
    if only_open_access:
        parts.append("access_right:open")
    return " AND ".join(parts)


def _author_name(name: str) -> str:
    """Zenodo writes "Last, First"; the other engines (and the PDF file name) expect "First Last"."""

    name = " ".join(str(name).split())
    if name.count(",") == 1:
        last, first = (part.strip() for part in name.split(","))
        if last and first:
            return f"{first} {last}"
    return name


def _year(metadata: dict):
    value = (metadata.get("publication_date") or "")[:4]
    return int(value) if value.isdigit() else None


def zenodo_record(hit: dict) -> dict:
    """One Zenodo record as a row of the search CSV."""

    metadata = hit.get("metadata") or {}
    authors = [_author_name(creator["name"]) for creator in metadata.get("creators") or [] if creator.get("name")]
    access_right = metadata.get("access_right")
    return build_record(
        engine_label=ENTRY_SOURCE_ZENODO,
        internal_id=hit.get("id"),
        year=_year(metadata),
        title=strip_html(metadata.get("title") or hit.get("title")),
        authors=authors,
        abstract=strip_html(metadata.get("description")),
        doi=metadata.get("doi") or hit.get("doi"),
        is_open_access=None if access_right is None else access_right == "open",
    )


def run_zenodo_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    raw_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
    custom_query: str | None = None,
) -> list[dict]:
    """Page through the Zenodo records API until ``--limit`` (or the cap) is reached."""

    session = retrying_session()
    api_key = os.getenv(ZENODO_API_KEY_ENV_VAR)
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    page_size = _PAGE_SIZE_WITH_KEY if api_key else _PAGE_SIZE_WITHOUT_KEY
    delay = _DELAY_WITH_KEY if api_key else _DELAY_WITHOUT_KEY

    zenodo_query = build_zenodo_query(query, year, only_open_access, custom_query)
    announce_query(ENTRY_SOURCE_ZENODO, f"{zenodo_query} [type=publication]", bool(custom_query), on_progress)

    params = {"q": zenodo_query, "type": "publication", "size": min(batch_size, page_size), "page": 1}
    limit = min(effective_limit(limit_size), ZENODO_MAX_RECORDS)
    pages, records = [], []
    while True:
        if params["page"] > 1:
            time.sleep(delay)
        response = session.get(_ZENODO_RECORDS_URL, params=dict(params), timeout=SEARCH_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        pages.append(data)
        save_raw_pages(raw_path, pages)
        hits_block = data.get("hits") or {}
        total = hits_block.get("total")
        if isinstance(total, dict):  # some Elasticsearch versions return {"value": n}
            total = total.get("value")

        hits = hits_block.get("hits") or []
        for hit in hits:
            if len(records) >= limit:
                break
            records.append(zenodo_record(hit))

        if on_progress is not None:
            on_progress(f"Zenodo: page {params['page']} — {len(records)}/{total} fetched")
        if params["page"] == 1:
            warning = cap_message(total, limit_size, ENTRY_SOURCE_ZENODO)
            if total is not None and total > ZENODO_MAX_RECORDS and limit == ZENODO_MAX_RECORDS:
                warning = f"Zenodo: {total} result(s), but paging stops at {ZENODO_MAX_RECORDS}. Narrow the query."
            if warning:
                logger.warning(warning)
                if on_progress is not None:
                    on_progress(warning)

        fetched = params["page"] * params["size"]
        if (
            len(records) >= limit
            or not hits
            or fetched >= ZENODO_MAX_RECORDS
            or (total is not None and fetched >= total)
        ):
            break
        params["page"] += 1

    logger.info("Zenodo total retrieved: %d record(s)", len(records))
    return records
