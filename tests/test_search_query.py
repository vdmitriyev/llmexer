"""Tests for search query notation: Semantic Scholar (+ | -) and OpenAlex (AND OR NOT)."""

import pytest

from llmexer.base.search_query import (
    QueryNotation,
    describe_notation,
    detect_notation,
    to_arxiv_query,
    to_core_query,
    to_europepmc_query,
    to_openalex_query,
    to_semantic_scholar_query,
    to_zenodo_query,
)
from llmexer.exceptions import SearchQueryNotationException


@pytest.mark.parametrize(
    "s2_query, expected",
    [
        ("machine +learning", "machine AND learning"),
        (
            '"deep learning" | "neural networks"',
            '"deep learning" OR "neural networks"',
        ),
        ("kayak -river", "kayak NOT river"),
        (
            '(elmo +"sesame street") -cookie',
            '(elmo AND "sesame street") NOT cookie',
        ),
        # Intra-word hyphens must be preserved (not treated as negation).
        ("state-of-the-art", "state-of-the-art"),
        # Leading negation.
        ("-survey", "NOT survey"),
        # Phrase-only queries pass through unchanged.
        ('"large language models"', '"large language models"'),
        # Operators inside a phrase are not converted.
        ('"AND gate" +logic', '"AND gate" AND logic'),
        ('"c++" | rust', '"c++" OR rust'),
    ],
)
def test_semantic_scholar_to_openalex(s2_query, expected):
    assert to_openalex_query(s2_query) == expected


@pytest.mark.parametrize(
    "oa_query, expected",
    [
        ("machine AND learning", "machine +learning"),
        ('"a b" OR "c d"', '"a b" | "c d"'),
        ("kayak NOT river", "kayak -river"),
        ("a AND NOT b", "a -b"),
        ("NOT (x OR y)", "-(x | y)"),
        ('llm AND "abstract screening"', 'llm +"abstract screening"'),
        ('(elmo AND "sesame street") NOT cookie', '(elmo +"sesame street") -cookie'),
        ('"c++" OR rust', '"c++" | rust'),
    ],
)
def test_openalex_to_semantic_scholar(oa_query, expected):
    assert to_semantic_scholar_query(oa_query) == expected


@pytest.mark.parametrize("oa_query", ["machine AND learning", "(a OR b) NOT c", '"c++" OR rust'])
def test_openalex_query_passes_to_openalex_unchanged(oa_query):
    assert to_openalex_query(oa_query) == oa_query


@pytest.mark.parametrize("s2_query", ["machine +learning", "(a | b) -c", '"AND gate" +logic'])
def test_semantic_scholar_query_passes_to_semantic_scholar_unchanged(s2_query):
    assert to_semantic_scholar_query(s2_query) == s2_query


@pytest.mark.parametrize(
    "query, notation",
    [
        ("machine +learning", QueryNotation.SEMANTIC_SCHOLAR),
        ("-survey", QueryNotation.SEMANTIC_SCHOLAR),
        ("machine AND learning", QueryNotation.OPENALEX),
        ('"+ and |" OR x', QueryNotation.OPENALEX),
        ('"large language models"', QueryNotation.PLAIN),
        ("rock and roll", QueryNotation.PLAIN),
        ("state-of-the-art", QueryNotation.PLAIN),
        ("", QueryNotation.PLAIN),
        (None, QueryNotation.PLAIN),
    ],
)
def test_detect_notation(query, notation):
    assert detect_notation(query) == notation


@pytest.mark.parametrize("query", ["a +b OR c", "a | b NOT c", "-a AND b"])
def test_mixed_notation_is_rejected(query):
    with pytest.raises(SearchQueryNotationException, match="one notation"):
        detect_notation(query)
    with pytest.raises(SearchQueryNotationException):
        to_openalex_query(query)
    with pytest.raises(SearchQueryNotationException):
        to_semantic_scholar_query(query)


def test_describe_notation_does_not_raise_on_mixed():
    assert describe_notation("a +b OR c") == "mixed (not supported)"
    assert describe_notation("a AND b") == "OpenAlex"


def test_plain_and_empty_queries_pass_through():
    assert to_openalex_query("") == ""
    assert to_openalex_query(None) is None
    assert to_semantic_scholar_query("") == ""
    assert to_semantic_scholar_query(None) is None
    assert to_semantic_scholar_query("rock and roll") == "rock and roll"


def test_unbalanced_quote_keeps_the_rest_unconverted():
    assert to_openalex_query('a +"b | c') == 'a AND "b | c'


@pytest.mark.parametrize(
    "query, expected",
    [
        ("llm", "(ti:llm OR abs:llm)"),
        ('"air quality" + sensor', '(ti:"air quality" OR abs:"air quality") AND (ti:sensor OR abs:sensor)'),
        ("a AND NOT b", "(ti:a OR abs:a) ANDNOT (ti:b OR abs:b)"),
        ("a -b", "(ti:a OR abs:a) ANDNOT (ti:b OR abs:b)"),
        ("(a OR b) c", "((ti:a OR abs:a) OR (ti:b OR abs:b)) AND (ti:c OR abs:c)"),
        ('"a AND b" OR c', '(ti:"a AND b" OR abs:"a AND b") OR (ti:c OR abs:c)'),
    ],
)
def test_to_arxiv_query(query, expected):
    assert to_arxiv_query(query) == expected


@pytest.mark.parametrize("query", ["NOT a", "a OR NOT b"])
def test_to_arxiv_query_rejects_a_negation_arxiv_cannot_express(query):
    with pytest.raises(SearchQueryNotationException, match="arXiv"):
        to_arxiv_query(query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ('"air quality" + sensor -indoor', 'TITLE_ABS:("air quality" AND sensor NOT indoor)'),
        ("a OR b", "TITLE_ABS:(a OR b)"),
        ("", ""),
    ],
)
def test_to_europepmc_query(query, expected):
    assert to_europepmc_query(query) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("llm", "(title:llm OR abstract:llm)"),
        (
            '("air quality" | smog) + (sensor | monitoring)',
            '((title:"air quality" OR abstract:"air quality") OR (title:smog OR abstract:smog)) AND '
            "((title:sensor OR abstract:sensor) OR (title:monitoring OR abstract:monitoring))",
        ),
        ("a -b", "(title:a OR abstract:a) AND NOT (title:b OR abstract:b)"),
        ("a OR NOT b", "(title:a OR abstract:a) OR NOT (title:b OR abstract:b)"),
        ("NOT a", "NOT (title:a OR abstract:a)"),
    ],
)
def test_to_core_query_scopes_each_term(query, expected):
    """No field wraps a bracketed group: CORE dropped the AND between groups nested in one field."""
    assert to_core_query(query) == expected
    assert "title:(" not in to_core_query(query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("llm", "(title:llm OR description:llm)"),
        (
            '("air quality" OR smog) AND sensor',
            '((title:"air quality" OR description:"air quality") OR (title:smog OR description:smog)) AND '
            "(title:sensor OR description:sensor)",
        ),
        ("a -b", "(title:a OR description:a) AND NOT (title:b OR description:b)"),
    ],
)
def test_to_zenodo_query(query, expected):
    assert to_zenodo_query(query) == expected
