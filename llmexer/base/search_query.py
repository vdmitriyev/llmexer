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


# A quoted phrase (possibly unclosed), a bracket, or a run of other characters.
_TOKEN = re.compile(r'"[^"]*"?|[()]|[^\s()"]+')


def to_europepmc_query(query: str) -> str:
    """``query`` for Europe PMC: OpenAlex notation, searched in title and abstract."""

    oa_query = to_openalex_query(query)
    return f"TITLE_ABS:({oa_query})" if oa_query else ""


def _join_parts(parts: list[str]) -> str:
    text = ""
    for part in parts:
        if not text or text.endswith("(") or part == ")":
            text += part
        else:
            text += f" {part}"
    return text


def _scope_each_term(query: str, wrap, and_not: str, lone_not: str | None, engine: str) -> str:
    """``query`` with every term or phrase wrapped by ``wrap`` (e.g. into title OR abstract).

    The operators and brackets stay at the top level, outside any field. Terms written
    side by side are joined with AND. ``NOT`` after a term, and ``AND NOT``, become
    ``and_not``; any other ``NOT`` (at the start, after ``OR``) becomes ``lone_not``.

    Raises:
        SearchQueryNotationException: for a lone ``NOT`` when ``lone_not`` is ``None``.
    """

    parts: list[str] = []
    after_operand = False
    for word in _TOKEN.findall(to_openalex_query(query) or ""):
        if word in ("AND", "OR"):
            parts.append(word)
            after_operand = False
        elif word == "NOT":
            if after_operand:
                parts.append(and_not)
            elif parts and parts[-1] == "AND":
                parts[-1] = and_not
            elif lone_not is not None:
                parts.append(lone_not)
            else:
                raise SearchQueryNotationException(
                    f"{engine} can only exclude terms after another term ('A AND NOT B'), "
                    "not at the start of a query or after OR."
                )
            after_operand = False
        elif word == "(":
            if after_operand:
                parts.append("AND")
            parts.append("(")
            after_operand = False
        elif word == ")":
            parts.append(")")
            after_operand = True
        else:
            if after_operand:
                parts.append("AND")
            parts.append(wrap(word))
            after_operand = True
    return _join_parts(parts)


def to_arxiv_query(query: str) -> str:
    """``query`` for the arXiv API: every term searched in title or abstract.

    arXiv has no field for title and abstract together, so each term or phrase
    becomes ``(ti:x OR abs:x)``. Terms written side by side are joined with AND,
    and ``NOT`` / ``AND NOT`` become ``ANDNOT``, the only negation arXiv knows.

    Raises:
        SearchQueryNotationException: for a negation arXiv cannot express
            (a query starting with NOT, or ``OR NOT``).
    """

    return _scope_each_term(query, lambda word: f"(ti:{word} OR abs:{word})", "ANDNOT", None, "arXiv")


def to_core_query(query: str) -> str:
    """``query`` for the CORE API: every term searched in title or abstract.

    Each term or phrase becomes ``(title:x OR abstract:x)``, with the brackets and
    operators of the query kept outside the fields. CORE drops the AND between
    bracketed groups nested inside one field (``title:((a OR b) AND c)`` matched
    ``a OR b`` only), so the whole query must not be wrapped in a field.
    """

    return _scope_each_term(query, lambda word: f"(title:{word} OR abstract:{word})", "AND NOT", "NOT", "CORE")


def to_pubmed_query(query: str) -> str:
    """``query`` for PubMed as a self-contained string: every term tagged ``[tiab]``.

    ``search run`` normally sends ``to_openalex_query(query)`` with ``field=tiab``; an own
    string from ``queries:`` is sent without that field, so this form carries the tags
    itself. ``NOT`` / ``AND NOT`` become PubMed's ``NOT``.

    Raises:
        SearchQueryNotationException: for a negation PubMed cannot express
            (a query starting with NOT, or ``OR NOT``).
    """

    return _scope_each_term(query, lambda word: f"{word}[tiab]", "NOT", None, "PubMed")


def to_zenodo_query(query: str) -> str:
    """``query`` for Zenodo: every term searched in title or description (the abstract)."""

    return _scope_each_term(query, lambda word: f"(title:{word} OR description:{word})", "AND NOT", "NOT", "Zenodo")
