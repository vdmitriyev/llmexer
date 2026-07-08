"""Tests for Semantic Scholar -> OpenAlex query notation conversion."""

import pytest

from llmexer.base.search_openalex import _convert_query_notation


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
    ],
)
def test_convert_query_notation(s2_query, expected):
    assert _convert_query_notation(s2_query) == expected


def test_convert_query_notation_empty():
    assert _convert_query_notation("") == ""
    assert _convert_query_notation(None) is None
