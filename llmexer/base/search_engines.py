"""The search engines ``search run`` can query, and which of them a search uses.

Engines run in the order given; each one adds only the papers the earlier ones did not
find. The list comes from the search file's ``engines:``, else
``LLMEXER_SEARCH_ENGINES`` / ``search_engines`` in ``llmexer.ini``, else the default
(Semantic Scholar, then OpenAlex and CORE, which run only when their API key is set).

Every runner takes ``(query, year, only_open_access, batch, limit, raw_path, csv_path,
on_progress, custom_query=None)`` and returns the rows for the search CSV. ``custom_query``
is the engine's own string from the search file's ``queries:``; it replaces the string
converted from ``query`` and is used as written.
"""

import os
from dataclasses import dataclass
from typing import Callable

import llmexer.constants as _constants
from llmexer.base.search_arxiv import ENTRY_SOURCE_ARXIV, run_arxiv_search
from llmexer.base.search_core import (
    CORE_API_KEY_ENV_VAR,
    ENTRY_SOURCE_CORE,
    run_core_search,
)
from llmexer.base.search_europepmc import ENTRY_SOURCE_EUROPEPMC, run_europepmc_search
from llmexer.base.search_openalex import ENTRY_SOURCE_OPENALEX, run_openalex_search
from llmexer.base.search_pubmed import ENTRY_SOURCE_PUBMED, run_pubmed_search
from llmexer.base.search_query import (
    to_arxiv_query,
    to_core_query,
    to_europepmc_query,
    to_openalex_query,
    to_pubmed_query,
    to_semantic_scholar_query,
    to_zenodo_query,
)
from llmexer.base.search_semantic_scholar import (
    ENTRY_SOURCE_SEMANTIC_SCHOLAR,
    run_semantic_scholar_search,
)
from llmexer.base.search_zenodo import ENTRY_SOURCE_ZENODO, run_zenodo_search
from llmexer.exceptions import UnexpectedCLIParamsException


@dataclass(frozen=True)
class SearchEngine:
    """One search engine: its key (as typed), label (``entry_source``) and caller."""

    key: str
    label: str
    raw_suffix: str
    runner: Callable[..., list[dict]]
    required_env: str | None = None


def _semantic_scholar(query, year, only_open_access, batch, limit, raw_path, csv_path, on_progress, custom_query=None):
    return run_semantic_scholar_search(
        query, year, only_open_access, batch, limit, raw_path, csv_path, on_progress, custom_query=custom_query
    )


def _openalex(query, year, only_open_access, batch, limit, raw_path, csv_path, on_progress, custom_query=None):
    del csv_path
    api_key = os.getenv("OPENALEX_API_KEY")
    return run_openalex_search(
        query, year, only_open_access, batch, limit, raw_path, api_key, on_progress, custom_query=custom_query
    )


def _without_csv(runner):
    def run(query, year, only_open_access, batch, limit, raw_path, csv_path, on_progress, custom_query=None):
        del csv_path
        return runner(query, year, only_open_access, batch, limit, raw_path, on_progress, custom_query=custom_query)

    return run


ENGINES = {
    engine.key: engine
    for engine in (
        # Semantic Scholar keeps the raw-file name it had before there were other engines.
        SearchEngine("semantic_scholar", ENTRY_SOURCE_SEMANTIC_SCHOLAR, "__results_raw.json", _semantic_scholar),
        SearchEngine(
            "openalex", ENTRY_SOURCE_OPENALEX, "__openalex_results_raw.json", _openalex, required_env="OPENALEX_API_KEY"
        ),
        SearchEngine(
            "core",
            ENTRY_SOURCE_CORE,
            "__core_results_raw.json",
            _without_csv(run_core_search),
            required_env=CORE_API_KEY_ENV_VAR,
        ),
        SearchEngine(
            "europepmc", ENTRY_SOURCE_EUROPEPMC, "__europepmc_results_raw.json", _without_csv(run_europepmc_search)
        ),
        SearchEngine("pubmed", ENTRY_SOURCE_PUBMED, "__pubmed_results_raw.json", _without_csv(run_pubmed_search)),
        SearchEngine("arxiv", ENTRY_SOURCE_ARXIV, "__arxiv_results_raw.json", _without_csv(run_arxiv_search)),
        SearchEngine("zenodo", ENTRY_SOURCE_ZENODO, "__zenodo_results_raw.json", _without_csv(run_zenodo_search)),
    )
}


def parse_engine_keys(values) -> list[str]:
    """Engine keys from a list, a comma-separated string, or both; order kept, repeats dropped.

    Raises:
        UnexpectedCLIParamsException: on a key that is not an engine.
    """

    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    keys = []
    for item in values:
        for value in str(item).split(","):
            key = value.strip().lower()
            if key and key not in keys:
                keys.append(key)
    unknown = [key for key in keys if key not in ENGINES]
    if unknown:
        raise UnexpectedCLIParamsException(
            f"Unknown search engine(s): {', '.join(unknown)}. Known: {', '.join(ENGINES)}."
        )
    return keys


def resolve_engines(file_values=None) -> list[str]:
    """The engines of a search: the search file's ``engines:``, else the configured default."""

    for values in (file_values, _constants.SEARCH_ENGINES or _constants.DEFAULT_SEARCH_ENGINES):
        keys = parse_engine_keys(values)
        if keys:
            return keys
    return parse_engine_keys(_constants.DEFAULT_SEARCH_ENGINES)


def raw_file_suffixes() -> list[str]:
    """Every raw-response suffix, for commands that move or list a search's files."""

    return [engine.raw_suffix for engine in ENGINES.values()]


def check_search_queries(queries, engines: list[str]) -> tuple[dict[str, str], list[str]]:
    """The own search strings of a search file's ``queries:``, and warnings about them.

    Returns ``({engine key: string}, warnings)``; empty strings are dropped. A string for
    an engine the search does not use is reported, not an error.

    Raises:
        UnexpectedCLIParamsException: when ``queries:`` is not a mapping, names an engine
            that does not exist, or holds a value that is not a string.
    """

    if queries is None:
        return {}, []
    if not isinstance(queries, dict):
        raise UnexpectedCLIParamsException(
            "`queries:` in the search file must map engine names to search strings, e.g. `pubmed: '...'`."
        )
    unknown = [str(key) for key in queries if str(key).strip().lower() not in ENGINES]
    if unknown:
        raise UnexpectedCLIParamsException(
            f"Unknown search engine(s) in `queries:`: {', '.join(unknown)}. Known: {', '.join(ENGINES)}."
        )
    checked, warnings = {}, []
    for key, value in queries.items():
        engine_key = str(key).strip().lower()
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            raise UnexpectedCLIParamsException(
                f"`queries: {engine_key}:` must be a search string, not {type(value).__name__}."
            )
        if engine_key not in engines:
            warnings.append(
                f"`queries: {engine_key}:` is not used: {ENGINES[engine_key].label} is not in this search's engines."
            )
            continue
        checked[engine_key] = value.strip()
    return checked, warnings


# Engine key -> `query` converted into that engine's own syntax, in the form `queries:` takes.
# Used as an own string, each gives the same request as the automatic conversion.
CONVERTERS = {
    "semantic_scholar": to_semantic_scholar_query,
    "openalex": to_openalex_query,  # OpenAlex scopes it to title and abstract with a filter
    "core": to_core_query,
    "europepmc": to_europepmc_query,
    "pubmed": to_pubmed_query,
    "arxiv": to_arxiv_query,
    "zenodo": to_zenodo_query,
}


def convert_query(query: str, engine_key: str) -> str:
    """``query`` written in the syntax of one engine, ready for the search file's ``queries:``."""

    return CONVERTERS[engine_key](query)
