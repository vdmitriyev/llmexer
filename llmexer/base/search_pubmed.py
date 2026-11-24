"""PubMed search engine: the NCBI E-utilities caller for ``search run``.

``esearch`` finds the PMIDs and keeps them on NCBI's history server; ``efetch`` then reads
the records in batches as XML. No key is needed. ``NCBI_API_KEY`` raises the limit from
3 to 10 requests per second. The query is searched in title and abstract (``field=tiab``).
"""

import os
import re
import time
from typing import Callable, Optional

from defusedxml import ElementTree

from llmexer.base.search import SEARCH_HTTP_TIMEOUT
from llmexer.base.search_query import to_openalex_query
from llmexer.base.search_record import (
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

ENTRY_SOURCE_PUBMED = "PubMed"
NCBI_API_KEY_ENV_VAR = "NCBI_API_KEY"

_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH_URL = f"{_EUTILS_URL}/esearch.fcgi"
_EFETCH_URL = f"{_EUTILS_URL}/efetch.fcgi"
_TOOL = "llmexer"
_EFETCH_MAX_BATCH = 500
# esearch / efetch do not page past this many records of one query.
PUBMED_MAX_RECORDS = 10000
# NCBI allows 3 requests per second without a key and 10 with one.
_DELAY_WITHOUT_KEY = 0.34
_DELAY_WITH_KEY = 0.11
_YEAR = re.compile(r"\d{4}")


def build_pubmed_term(query: str, only_open_access: bool, custom_query: str | None = None) -> str:
    """The esearch ``term``: the query in ``AND OR NOT`` notation, plus the free-full-text filter."""

    term = custom_query or to_openalex_query(query) or ""
    if only_open_access:
        term = f"({term}) AND free full text[sb]" if term else "free full text[sb]"
    return term


def _common_params() -> dict:
    params = {"db": "pubmed", "tool": _TOOL}
    email = os.getenv("UNPAYWALL_EMAIL")
    if email:
        params["email"] = email
    api_key = os.getenv(NCBI_API_KEY_ENV_VAR)
    if api_key:
        params["api_key"] = api_key
    return params


def _text(element) -> str | None:
    """All text of an element, inline markup (``<i>``, ``<sup>``) included."""

    if element is None:
        return None
    text = " ".join("".join(element.itertext()).split())
    return text or None


def _abstract(article) -> str | None:
    parts = []
    for element in article.findall("Abstract/AbstractText"):
        text = _text(element)
        if not text:
            continue
        label = element.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts) or None


def _authors(article) -> list[str]:
    names = []
    for author in article.findall("AuthorList/Author"):
        collective = _text(author.find("CollectiveName"))
        if collective:
            names.append(collective)
            continue
        parts = [_text(author.find("ForeName")), _text(author.find("LastName"))]
        name = " ".join(part for part in parts if part)
        if name:
            names.append(name)
    return names


def _year(article):
    pub_date = article.find("Journal/JournalIssue/PubDate")
    candidates = []
    if pub_date is not None:
        candidates += [pub_date.findtext("Year"), pub_date.findtext("MedlineDate")]
    candidates.append(article.findtext("ArticleDate/Year"))
    for candidate in candidates:
        match = _YEAR.search(candidate or "")
        if match:
            return int(match.group(0))
    return None


def _doi(pubmed_article, article) -> str | None:
    # Only the paper's own ids: the reference list carries ArticleIds of the cited papers.
    for element in pubmed_article.findall("PubmedData/ArticleIdList/ArticleId"):
        if element.get("IdType") == "doi" and element.text:
            return element.text.strip()
    for element in article.findall("ELocationID"):
        if element.get("EIdType") == "doi" and element.text:
            return element.text.strip()
    return None


def pubmed_records(xml_text: str, only_open_access: bool = False) -> tuple[list[dict], int]:
    """``(records, skipped)`` from one efetch page; book chapters (no journal article) are skipped."""

    root = ElementTree.fromstring(xml_text)
    records = []
    for pubmed_article in root.findall("PubmedArticle"):
        citation = pubmed_article.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        if article is None:
            continue
        title = _text(article.find("ArticleTitle"))
        records.append(
            build_record(
                engine_label=ENTRY_SOURCE_PUBMED,
                internal_id=citation.findtext("PMID"),
                year=_year(article),
                title=title,
                authors=_authors(article),
                abstract=_abstract(article),
                doi=_doi(pubmed_article, article),
                is_open_access=True if only_open_access else None,
            )
        )
    skipped = len(root.findall("PubmedBookArticle"))
    return records, skipped


def run_pubmed_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    raw_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
    custom_query: str | None = None,
) -> list[dict]:
    """Run esearch once, then efetch the records in batches until ``--limit`` (or the cap)."""

    session = retrying_session()
    delay = _DELAY_WITH_KEY if os.getenv(NCBI_API_KEY_ENV_VAR) else _DELAY_WITHOUT_KEY
    term = build_pubmed_term(query, only_open_access, custom_query)

    search_params = {
        **_common_params(),
        "term": term,
        "usehistory": "y",
        "retmax": 0,
        "retmode": "json",
    }
    if not custom_query:
        # Every term in title/abstract; an own string sets its fields with tags like [tiab].
        search_params["field"] = "tiab"
    years = parse_year_range(year)
    if years:
        search_params.update({"datetype": "pdat", "mindate": years[0], "maxdate": years[1]})
    shown = term + (" [field=tiab]" if not custom_query else "")
    if years:
        shown += f" [publication date {years[0]}-{years[1]}]"
    announce_query(ENTRY_SOURCE_PUBMED, shown, bool(custom_query), on_progress)

    response = session.get(_ESEARCH_URL, params=search_params, timeout=SEARCH_HTTP_TIMEOUT)
    response.raise_for_status()
    found = response.json().get("esearchresult") or {}
    pages: list = [found]
    save_raw_pages(raw_path, pages)
    total = int(found.get("count") or 0)

    warning = cap_message(total, limit_size, "PubMed")
    limit = min(effective_limit(limit_size), PUBMED_MAX_RECORDS)
    if total > PUBMED_MAX_RECORDS and limit == PUBMED_MAX_RECORDS:
        warning = f"PubMed: {total} result(s), but E-utilities return at most {PUBMED_MAX_RECORDS}. Narrow the query."
    if warning:
        logger.warning(warning)
        if on_progress is not None:
            on_progress(warning)

    wanted = min(total, limit)
    fetch_params = {
        **_common_params(),
        "query_key": found.get("querykey"),
        "WebEnv": found.get("webenv"),
        "retmode": "xml",
        "retmax": min(batch_size, _EFETCH_MAX_BATCH),
    }
    records: list[dict] = []
    skipped = 0
    page = 0
    retstart = 0
    while retstart < wanted:
        time.sleep(delay)
        fetch_params["retstart"] = retstart
        response = session.get(_EFETCH_URL, params=dict(fetch_params), timeout=SEARCH_HTTP_TIMEOUT)
        response.raise_for_status()
        pages.append(response.text)
        save_raw_pages(raw_path, pages)
        page += 1
        page_records, page_skipped = pubmed_records(response.text, only_open_access)
        skipped += page_skipped
        records.extend(page_records[: max(wanted - len(records), 0)])
        if on_progress is not None:
            on_progress(f"PubMed: page {page} — {len(records)}/{total} fetched")
        if not page_records and not page_skipped:
            break
        retstart += fetch_params["retmax"]

    if skipped:
        logger.info("PubMed: skipped %d book chapter(s)", skipped)
    logger.info("PubMed total retrieved: %d record(s)", len(records))
    return records
