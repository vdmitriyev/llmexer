"""Europe PMC search engine: the REST search API caller for ``search run``.

Europe PMC covers PubMed / MEDLINE, PubMed Central and preprint servers. No key is needed.
The query is searched in title and abstract (``TITLE_ABS``), like the other engines.
"""

from typing import Callable, Optional

from llmexer.base.search import SEARCH_HTTP_TIMEOUT
from llmexer.base.search_query import to_europepmc_query
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

ENTRY_SOURCE_EUROPEPMC = "Europe PMC"

_EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_EUROPEPMC_MAX_PAGE_SIZE = 1000


def build_europepmc_query(query: str, year: str | None, only_open_access: bool, custom_query: str | None = None) -> str:
    """The full Europe PMC query: the search terms plus the year and open-access filters."""

    if custom_query:
        parts = [f"({custom_query})"]
    else:
        parts = [to_europepmc_query(query)] if query else []
    years = parse_year_range(year)
    if years:
        parts.append(f"PUB_YEAR:[{years[0]} TO {years[1]}]")
    if only_open_access:
        parts.append("OPEN_ACCESS:y")
    return " AND ".join(parts)


def _authors(result: dict) -> list[str]:
    authors = ((result.get("authorList") or {}).get("author")) or []
    names = []
    for author in authors:
        name = author.get("fullName") or " ".join(
            part for part in (author.get("firstName"), author.get("lastName")) if part
        )
        if not name:
            name = author.get("collectiveName")
        if name:
            names.append(name.strip())
    if not names and result.get("authorString"):
        names = [name.strip().rstrip(".") for name in result["authorString"].split(",") if name.strip()]
    return names


def _year(result: dict):
    value = result.get("pubYear")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def europepmc_record(result: dict) -> dict:
    """One Europe PMC result as a row of the search CSV."""

    source, result_id = result.get("source"), result.get("id")
    is_open_access = result.get("isOpenAccess")
    return build_record(
        engine_label=ENTRY_SOURCE_EUROPEPMC,
        internal_id=f"{source}:{result_id}" if source and result_id else result_id,
        year=_year(result),
        title=strip_html(result.get("title")),
        authors=_authors(result),
        abstract=strip_html(result.get("abstractText")),
        doi=result.get("doi"),
        is_open_access=None if is_open_access is None else is_open_access == "Y",
        citation_count=result.get("citedByCount"),
    )


def run_europepmc_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    raw_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
    custom_query: str | None = None,
) -> list[dict]:
    """Page through Europe PMC with ``cursorMark`` until ``--limit`` (or the cap) is reached."""

    session = retrying_session()
    epmc_query = build_europepmc_query(query, year, only_open_access, custom_query)
    announce_query(ENTRY_SOURCE_EUROPEPMC, epmc_query, bool(custom_query), on_progress)

    params = {
        "query": epmc_query,
        "format": "json",
        "resultType": "core",
        "pageSize": min(batch_size, _EUROPEPMC_MAX_PAGE_SIZE),
        "cursorMark": "*",
    }
    limit = effective_limit(limit_size)
    pages, records = [], []
    page = 0
    while True:
        response = session.get(_EUROPEPMC_SEARCH_URL, params=dict(params), timeout=SEARCH_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        pages.append(data)
        save_raw_pages(raw_path, pages)
        page += 1
        total = data.get("hitCount")

        results = (data.get("resultList") or {}).get("result") or []
        for result in results:
            if len(records) >= limit:
                break
            records.append(europepmc_record(result))

        if on_progress is not None:
            on_progress(f"Europe PMC: page {page} — {len(records)}/{total} fetched")
        if page == 1:
            warning = cap_message(total, limit_size, "Europe PMC")
            if warning:
                logger.warning(warning)
                if on_progress is not None:
                    on_progress(warning)

        next_cursor = data.get("nextCursorMark")
        fetched_all = total is not None and len(records) >= total
        if (
            len(records) >= limit
            or fetched_all
            or not results
            or not next_cursor
            or next_cursor == params["cursorMark"]
        ):
            break
        params["cursorMark"] = next_cursor

    logger.info("Europe PMC total retrieved: %d record(s)", len(records))
    return records
