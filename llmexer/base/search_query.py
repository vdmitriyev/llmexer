"""Search query notation: Semantic Scholar (``+ | -``) or OpenAlex (``AND OR NOT``).

A query is written in one notation and translated for the other engine. Phrases
(``"..."``) and grouping (``(...)``) are the same in both and are never changed;
nothing inside a quoted phrase counts as an operator. A query that uses operators
of both notations is rejected.
"""

import re
from enum import Enum

from llmexer.exceptions import SearchQueryNotationException


class QueryNotation(str, Enum):
    """The notation a search query is written in."""

    SEMANTIC_SCHOLAR = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    PLAIN = "plain"


# Prefix negation only: "-term" / " -term" / "(-term". Leaves "state-of-the-art" alone.
_S2_NEGATION = re.compile(r"(^|[\s(])-(?=[^\s)])")
_S2_AND_OR = re.compile(r"[+|]")
# OpenAlex operators must be uppercase whole words; "and" / "or" / "not" are search words.
_OA_OPERATOR = re.compile(r"\b(AND|OR|NOT)\b")
_OA_AND_NOT = re.compile(r"\s*\bAND\s+NOT\b\s*")
_OA_NOT = re.compile(r"\bNOT\b\s*")
_OA_AND = re.compile(r"\s*\bAND\b\s*")
_OA_OR = re.compile(r"\s*\bOR\b\s*")
_WHITESPACE = re.compile(r"\s+")


def _split_phrases(query: str) -> list[tuple[str, bool]]:
    """Split ``query`` into ``(text, is_quoted)`` segments; quoted text keeps its quotes.

    An unbalanced quote makes the rest of the query one quoted segment.
    """

    parts = query.split('"')
    has_unbalanced_quote = len(parts) % 2 == 0
    segments = []
    for index, part in enumerate(parts):
        is_quoted = index % 2 == 1
        if not is_quoted:
            segments.append((part, False))
            continue
        is_last = index == len(parts) - 1
        closing_quote = "" if (is_last and has_unbalanced_quote) else '"'
        segments.append((f'"{part}{closing_quote}', True))
    return segments


def _s2_operators(text: str) -> list[str]:
    found = _S2_AND_OR.findall(text)
    if _S2_NEGATION.search(text):
        found.append("-")
    return found


def detect_notation(query: str) -> QueryNotation:
    """Return the notation of ``query``.

    Raises:
        SearchQueryNotationException: when the query mixes both notations.
    """

    if not query:
        return QueryNotation.PLAIN

    s2_found = []
    oa_found = []
    for text, is_quoted in _split_phrases(query):
        if is_quoted:
            continue
        s2_found.extend(_s2_operators(text))
        oa_found.extend(_OA_OPERATOR.findall(text))

    if s2_found and oa_found:
        s2_listed = " ".join(sorted(set(s2_found)))
        oa_listed = " ".join(sorted(set(oa_found)))
        raise SearchQueryNotationException(
            f"The query mixes Semantic Scholar operators ({s2_listed}) with OpenAlex operators "
            f"({oa_listed}). Write it in one notation only: '+ | -' or 'AND OR NOT'."
        )
    if s2_found:
        return QueryNotation.SEMANTIC_SCHOLAR
    if oa_found:
        return QueryNotation.OPENALEX
    return QueryNotation.PLAIN


def describe_notation(query: str) -> str:
    """The notation name for display; never raises, so old saved queries still show."""

    try:
        return detect_notation(query).value
    except SearchQueryNotationException:
        return "mixed (not supported)"


def _s2_segment_to_openalex(text: str) -> str:
    negated = _S2_NEGATION.sub(r"\1NOT ", text)
    with_and = negated.replace("+", " AND ")
    with_or = with_and.replace("|", " OR ")
    return _WHITESPACE.sub(" ", with_or)


def _openalex_segment_to_s2(text: str) -> str:
    with_and_not = _OA_AND_NOT.sub(" -", text)
    with_not = _OA_NOT.sub("-", with_and_not)
    with_and = _OA_AND.sub(" +", with_not)
    with_or = _OA_OR.sub(" | ", with_and)
    return _WHITESPACE.sub(" ", with_or)


def _convert_unquoted(query: str, convert) -> str:
    converted_parts = []
    for text, is_quoted in _split_phrases(query):
        if is_quoted:
            converted_parts.append(text)
        else:
            converted_parts.append(convert(text))
    return "".join(converted_parts).strip()


def to_openalex_query(query: str) -> str:
    """Return ``query`` in OpenAlex notation."""

    if not query or detect_notation(query) != QueryNotation.SEMANTIC_SCHOLAR:
        return query
    return _convert_unquoted(query, _s2_segment_to_openalex)


def to_semantic_scholar_query(query: str) -> str:
    """Return ``query`` in Semantic Scholar notation."""

    if not query or detect_notation(query) != QueryNotation.OPENALEX:
        return query
    return _convert_unquoted(query, _openalex_segment_to_s2)
