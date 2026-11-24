"""arXiv search engine: the arXiv API caller for ``search run``.

No key is needed, but arXiv asks for at most one request every three seconds. Every
paper has a DataCite DOI (``10.48550/arXiv.<id>``); the journal DOI an author added is
used instead when there is one. Every arXiv paper is open access.
"""

import re
import time
from typing import Callable, Optional

from defusedxml import ElementTree

from llmexer.base.search import SEARCH_HTTP_TIMEOUT
from llmexer.base.search_query import to_arxiv_query
from llmexer.base.search_record import (
    FIRST_YEAR,
    LAST_YEAR,
    announce_query,
    build_record,
    cap_message,
    effective_limit,
    parse_year_range,
    retrying_session,
    save_raw_pages,
)
from llmexer.logger import get_logger

logger = get_logger()

ENTRY_SOURCE_ARXIV = "arXiv"

_ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
_ARXIV_PAGE_SIZE = 500
# "make no more than one request every three seconds" (arXiv API terms of use)
ARXIV_DELAY_SECONDS = 3.1
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
_VERSION = re.compile(r"v\d+$")
_ARXIV_DOI_PREFIX = "10.48550/arXiv."


def build_arxiv_query(query: str, year: str | None, custom_query: str | None = None) -> str:
    """The ``search_query``: the terms in title or abstract, plus the submission-date range."""

    if custom_query:
        parts = [f"({custom_query})"]
    else:
        parts = [to_arxiv_query(query)] if query else []
    years = parse_year_range(year)
    if years:
        start = years[0] if years[0] != FIRST_YEAR else "1990"
        end = years[1] if years[1] != LAST_YEAR else "2999"
        parts.append(f"submittedDate:[{start}01010000 TO {end}12312359]")
    return " AND ".join(parts)


def _text(entry, path: str) -> str | None:
    value = entry.findtext(path, namespaces=_NS)
    if value is None:
        return None
    return " ".join(value.split()) or None


def arxiv_id(entry_id: str | None) -> str | None:
    """``http://arxiv.org/abs/2101.00001v2`` -> ``2101.00001`` (old ids keep their archive)."""

    if not entry_id:
        return None
    short = entry_id.split("/abs/", 1)[-1]
    return _VERSION.sub("", short)


def arxiv_records(xml_text: str) -> tuple[list[dict], int | None]:
    """``(records, total)`` from one Atom page."""

    root = ElementTree.fromstring(xml_text)
    total_text = root.findtext("opensearch:totalResults", namespaces=_NS)
    total = int(total_text) if total_text and total_text.strip().isdigit() else None

    records = []
    for entry in root.findall("atom:entry", _NS):
        paper_id = arxiv_id(_text(entry, "atom:id"))
        if not paper_id:
            continue
        published = _text(entry, "atom:published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            name for name in (_text(author, "atom:name") for author in entry.findall("atom:author", _NS)) if name
        ]
        doi = _text(entry, "arxiv:doi") or f"{_ARXIV_DOI_PREFIX}{paper_id.replace('/', '.')}"
        records.append(
            build_record(
                engine_label=ENTRY_SOURCE_ARXIV,
                internal_id=paper_id,
                year=year,
                title=_text(entry, "atom:title"),
                authors=authors,
                abstract=_text(entry, "atom:summary"),
                doi=doi,
                is_open_access=True,
            )
        )
    return records, total


def run_arxiv_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    raw_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
    custom_query: str | None = None,
) -> list[dict]:
    """Page through the arXiv API with ``start`` / ``max_results``, three seconds apart."""

    del only_open_access  # every arXiv paper is open access
    session = retrying_session()
    search_query = build_arxiv_query(query, year, custom_query)
    announce_query(ENTRY_SOURCE_ARXIV, search_query, bool(custom_query), on_progress)

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(batch_size, _ARXIV_PAGE_SIZE),
        "sortBy": "relevance",
    }
    limit = effective_limit(limit_size)
    pages, records = [], []
    page = 0
    while True:
        if page:
            time.sleep(ARXIV_DELAY_SECONDS)
        response = session.get(_ARXIV_QUERY_URL, params=dict(params), timeout=SEARCH_HTTP_TIMEOUT)
        response.raise_for_status()
        pages.append(response.text)
        save_raw_pages(raw_path, pages)
        page += 1
        page_records, total = arxiv_records(response.text)
        records.extend(page_records[: max(limit - len(records), 0)])

        if on_progress is not None:
            on_progress(f"arXiv: page {page} — {len(records)}/{total} fetched")
        if page == 1:
            warning = cap_message(total, limit_size, "arXiv")
            if warning:
                logger.warning(warning)
                if on_progress is not None:
                    on_progress(warning)

        params["start"] += params["max_results"]
        if len(records) >= limit or not page_records or (total is not None and params["start"] >= total):
            break

    logger.info("arXiv total retrieved: %d record(s)", len(records))
    return records
