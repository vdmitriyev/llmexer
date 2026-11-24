"""Shared pieces of the Europe PMC, PubMed and arXiv search callers.

One result row (`build_record`), the per-query cap (`search_max_responses`, ``env_cap``) and the
raw-response file (`save_raw_pages`). The Semantic Scholar and OpenAlex callers
predate this module and build their rows themselves.
"""

import json
import os
import re

from requests.adapters import HTTPAdapter, Retry
from rich.markup import escape

from llmexer.base.papers import make_structured_filename
from llmexer.base.search import detect_publication_lang
from llmexer.common import make_http_session
from llmexer.constants import DEFAULT_SEARCH_MAX_RESPONSES
from llmexer.logger import get_logger

logger = get_logger()

SEARCH_MAX_RESPONSES_ENV_VAR = "SEARCH_MAX_RESPONSES"


def env_cap(env_var: str, default: int) -> int:
    """A ceiling on records per query from ``env_var``, else ``default``.

    Read at call time, so a value from ``.env`` is used. An unset, non-positive or
    non-integer value falls back to the default.
    """

    raw = os.getenv(env_var)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        logger.warning("Invalid %s=%r; using default %d", env_var, raw, default)
    return default


def search_max_responses(
    env_var: str = SEARCH_MAX_RESPONSES_ENV_VAR, default: int = DEFAULT_SEARCH_MAX_RESPONSES
) -> int:
    """The ceiling on records per query: ``SEARCH_MAX_RESPONSES``, or an engine's own variable."""

    return env_cap(env_var, default)


def effective_limit(
    limit_size: int | None, env_var: str = SEARCH_MAX_RESPONSES_ENV_VAR, default: int = DEFAULT_SEARCH_MAX_RESPONSES
) -> int:
    """``--limit`` when given, never above the configured cap."""

    cap = search_max_responses(env_var, default)
    return cap if limit_size is None else min(limit_size, cap)


def cap_message(
    total: int | None,
    limit_size: int | None,
    engine_label: str,
    env_var: str = SEARCH_MAX_RESPONSES_ENV_VAR,
    default: int = DEFAULT_SEARCH_MAX_RESPONSES,
) -> str | None:
    """A warning when the configured cap (not ``--limit``) cuts the results, else ``None``."""

    cap = search_max_responses(env_var, default)
    if total is None or total <= cap or (limit_size is not None and limit_size <= cap):
        return None
    return (
        f"{engine_label}: {total} result(s), but at most {cap} are processed. Narrow the query or raise " f"{env_var}."
    )


_HTML_TAG = re.compile(r"<[^>]+>")


def strip_html(text):
    """Text without HTML tags (``<p>``, ``<i>``), whitespace collapsed; ``None`` / "" unchanged."""

    if not text:
        return text
    return " ".join(_HTML_TAG.sub(" ", str(text)).split())


def announce_query(engine_label: str, final_query: str, custom: bool, on_progress=None) -> None:
    """Note the search string sent to an engine in the log, and print it.

    ``custom`` says whether it came from the search file's ``queries:`` (used as written)
    or was converted from ``query``.
    """

    source = "own string from queries:" if custom else "converted from query"
    logger.info("%s query (%s): %s", engine_label, source, final_query)
    if on_progress is not None:
        suffix = " (from queries:)" if custom else ""
        # Escaped: PubMed tags like [tiab] would otherwise be read as Rich markup and vanish.
        on_progress(f"{engine_label} query{suffix}: {escape(final_query)}")


def retrying_session():
    """An HTTP session with the llmexer User-Agent that retries rate limits and server errors."""

    session = make_http_session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def save_raw_pages(raw_path: str, pages: list) -> None:
    """Write every raw page fetched so far (JSON objects, or XML text) as one JSON list."""

    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=4)


def build_record(
    *,
    engine_label: str,
    internal_id,
    year,
    title,
    authors: list[str],
    abstract,
    doi,
    is_open_access=None,
    citation_count=None,
    reference_count=None,
) -> dict:
    """One result row with the columns of the search CSV."""

    first_author_last_name = authors[0].split()[-1] if authors and authors[0].split() else None
    return {
        "search_engine_internal_id": internal_id,
        "year": year,
        "title": title,
        "authors": "; ".join(authors),
        "abstract": abstract,
        "isOpenAccess": is_open_access,
        "doi": doi,
        "language": detect_publication_lang(title, abstract),
        "referenceCount": reference_count,
        "citationCount": citation_count,
        "entry_source": engine_label,
        "pdf_filename": make_structured_filename(year, first_author_last_name, title, doi),
        "txt_filename": "",
        "markdown_filename": "",
        "pdf_downloaded": False,
    }


# Bounds for an open year range ("2020-" or "-2020", as Semantic Scholar accepts them).
FIRST_YEAR, LAST_YEAR = "1800", "2999"


def parse_year_range(year) -> tuple[str, str] | None:
    """The year parameter of a search as ``(start, end)``, or ``None`` for no year filter.

    ``"2020-2025"`` -> ``("2020", "2025")``; ``"2023"`` -> ``("2023", "2023")``;
    ``"2020-"`` -> ``("2020", LAST_YEAR)``; ``"-2020"`` -> ``(FIRST_YEAR, "2020")``.
    """

    if year is None or not str(year).strip():
        return None
    text = str(year).strip()
    if "-" not in text:
        return text, text
    start, end = (part.strip() for part in text.split("-", 1))
    if not start and not end:
        return None
    return start or FIRST_YEAR, end or LAST_YEAR
