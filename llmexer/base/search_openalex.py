"""OpenAlex search engine: the Works API caller for ``search run``.

Used as a second source after Semantic Scholar. Only invoked when ``OPENALEX_API_KEY`` is
set; the command layer combines the results, keeping only publications not already found by
Semantic Scholar (matched by DOI, falling back to title).
"""

import json
import os
import re
from typing import Callable, Optional

from requests.adapters import HTTPAdapter, Retry

from llmexer.base.papers import make_structured_filename
from llmexer.base.search import SEARCH_HTTP_TIMEOUT, detect_publication_lang
from llmexer.common import make_http_session
from llmexer.constants import DEFAULT_MAX_OPENALEX_RESPONSES
from llmexer.logger import get_logger

logger = get_logger()

ENTRY_SOURCE_OPENALEX = "OpenAlex"


def _max_openalex_responses() -> int:
    """Resolve the OpenAlex result ceiling from ``MAX_OPEN_ALEX_RESPONSES`` (else the default).

    Read at call time (not import time) so ``.env`` values loaded by the CLI are honored.
    Falls back to the default on an unset, non-positive, or non-integer value.
    """
    raw = os.getenv("MAX_OPEN_ALEX_RESPONSES")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning(
                "Invalid MAX_OPEN_ALEX_RESPONSES=%r; using default %d",
                raw,
                DEFAULT_MAX_OPENALEX_RESPONSES,
            )
    return DEFAULT_MAX_OPENALEX_RESPONSES


# OpenAlex API constants
_OPENALEX_WORKS_URL = "https://api.openalex.org/works"
# OpenAlex caps per-page results at 200; 100 mirrors the batching used elsewhere.
_OPENALEX_MAX_PER_PAGE = 100
_DEFAULT_MAILTO = "llmexer.openalex@local.local"
# Query is applied as a title+abstract filter to mirror Semantic Scholar's scope. The
# top-level ``search=`` param would instead match the full text of every work, which
# returns roughly an order of magnitude more (and far less relevant) results.
_OPENALEX_SEARCH_FIELD = "title_and_abstract.search"


def _strip_doi_prefix(doi: str | None) -> str | None:
    """Reduce an OpenAlex DOI (``https://doi.org/10.x``) to the bare form (``10.x``).

    This matches the DOI format stored by Semantic Scholar so cross-engine dedup works.
    """
    if not doi:
        return doi
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if doi.lower().startswith(prefix):
            return doi[len(prefix) :]
    return doi


def _strip_openalex_id_prefix(openalex_id: str | None) -> str | None:
    """Reduce an OpenAlex work ID URL to its bare short form.

    e.g. ``https://openalex.org/W7155217518`` -> ``W7155217518``.
    """
    if not openalex_id:
        return openalex_id
    for prefix in ("https://openalex.org/", "http://openalex.org/", "openalex.org/"):
        if openalex_id.lower().startswith(prefix):
            return openalex_id[len(prefix) :]
    return openalex_id


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild abstract text from an OpenAlex ``abstract_inverted_index`` (word -> positions)."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    if not positions:
        return None
    positions.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positions)


def _author_names(work: dict) -> list[str]:
    """Ordered list of author display names from a work's ``authorships``."""
    names = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = (author.get("display_name") or "").strip()
        if name:
            names.append(name)
    return names


def _year_filter(year: str | None) -> str | None:
    """Translate a ``"2020-2025"`` / ``"2023"`` year param into an OpenAlex date filter."""
    if not year:
        return None
    parts = str(year).split("-")
    start = parts[0].strip()
    end = parts[1].strip() if len(parts) > 1 and parts[1].strip() else start
    if not start:
        return None
    return f"from_publication_date:{start}-01-01,to_publication_date:{end}-12-31"


def _convert_query_notation(query: str) -> str:
    """Translate a Semantic Scholar query into OpenAlex search notation.

    ``+`` -> ``AND``, ``|`` -> ``OR``, prefix ``-`` -> ``NOT``. Phrases ("...") and
    grouping ((...)) are identical in both notations and pass through unchanged.

    A literal ``+``/``|`` inside a quoted phrase (e.g. ``"c++"``) would also be translated;
    such queries are rare and not special-cased.
    """
    if not query:
        return query
    # Prefix negation only: "-term" / " -term" / "(-term" -> "NOT term".
    # Leaves intra-word hyphens (e.g. "state-of-the-art") untouched.
    converted = re.sub(r"(^|[\s(])-(?=[^\s)])", r"\1NOT ", query)
    converted = converted.replace("+", " AND ").replace("|", " OR ")
    return re.sub(r"\s+", " ", converted).strip()


def run_openalex_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    json_path: str,
    api_key: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Call the OpenAlex Works API with cursor pagination.

    When ``on_progress`` is provided it is called with a human-readable status message
    after each fetched page, e.g. ``"OpenAlex: page 2 — 200/3402 fetched"``.

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
        "api_key": api_key,
        "mailto": os.getenv("UNPAYWALL_EMAIL") or _DEFAULT_MAILTO,
        "per_page": min(batch_size, _OPENALEX_MAX_PER_PAGE),
        "cursor": "*",
    }

    oa_query = _convert_query_notation(query)

    filters = []
    # NOTE: a query containing a literal comma would be misread as a filter separator
    # (OpenAlex has no comma escaping in filters); such queries are rare and out of scope.
    if oa_query:
        filters.append(f"{_OPENALEX_SEARCH_FIELD}:{oa_query}")
    year_filter = _year_filter(year)
    if year_filter:
        filters.append(year_filter)
    if only_open_access:
        filters.append("is_oa:true")
    if filters:
        params["filter"] = ",".join(filters)

    if on_progress is not None:
        on_progress(f"OpenAlex query: {oa_query}")

    # Hard ceiling on processed works: the configurable cap, tightened by an explicit --limit.
    max_responses = _max_openalex_responses()
    effective_limit = (
        max_responses if limit_size is None else min(limit_size, max_responses)
    )

    raw_json_results: list[dict] = []
    records: list[dict] = []
    page = 0
    capped_notified = False
    while True:
        response = session.get(
            _OPENALEX_WORKS_URL, params=params, timeout=SEARCH_HTTP_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        raw_json_results.append(data)
        page += 1
        total = (data.get("meta") or {}).get("count")

        works = data.get("results", [])
        for work in works:
            if len(records) >= effective_limit:
                break
            title = work.get("display_name")
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            doi = _strip_doi_prefix(work.get("doi"))
            author_names = _author_names(work)
            first_author_last_name = (
                author_names[0].split()[-1] if author_names else None
            )
            pdf_filename = make_structured_filename(
                work.get("publication_year"),
                first_author_last_name,
                title,
                doi,
            )
            open_access = work.get("open_access") or {}
            referenced_works = work.get("referenced_works") or []
            records.append(
                {
                    "search_engine_internal_id": _strip_openalex_id_prefix(
                        work.get("id")
                    ),
                    "year": work.get("publication_year"),
                    "title": title,
                    "authors": "; ".join(author_names),
                    "abstract": abstract,
                    "isOpenAccess": open_access.get("is_oa"),
                    "doi": doi,
                    "language": detect_publication_lang(title, abstract),
                    "referenceCount": len(referenced_works),
                    "citationCount": work.get("cited_by_count"),
                    "entry_source": ENTRY_SOURCE_OPENALEX,
                    "pdf_filename": pdf_filename,
                    "txt_filename": "",
                    "markdown_filename": "",
                    "pdf_downloaded": False,
                }
            )

        logger.info("OpenAlex: retrieved %d work(s) so far...", len(records))
        logger.debug(
            "OpenAlex page retrieved: %d works in page, %d total",
            len(works),
            len(records),
        )
        if on_progress is not None:
            if total is not None:
                on_progress(f"OpenAlex: page {page} — {len(records)}/{total} fetched")
            else:
                on_progress(f"OpenAlex: page {page} — {len(records)} fetched")

        # Warn once when the configurable cap (not an explicit --limit) truncates the output.
        if (
            total is not None
            and effective_limit == max_responses
            and total > max_responses
            and not capped_notified
        ):
            capped_notified = True
            message = (
                f"The max output processed is capped by configs to {max_responses}. "
                "Change the search criteria or set a bigger upper limit via "
                "environment variable MAX_OPEN_ALEX_RESPONSES."
            )
            logger.warning(message)
            if on_progress is not None:
                on_progress(message)

        if len(records) >= effective_limit:
            break
        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor or not works:
            break
        params["cursor"] = next_cursor

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(raw_json_results, fh, ensure_ascii=False, indent=4)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw_json_results, fh, ensure_ascii=False, indent=4)
    logger.debug("OpenAlex: wrote raw response to '%s'", json_path)

    logger.info("OpenAlex total retrieved: %d work(s)", len(records))

    return records
