"""Tests for the shared helpers in `llmexer.common`."""

import pytest

from llmexer.common import safe_filename_part


@pytest.mark.parametrize(
    "value, expected",
    [
        # The characters real model names carry.
        ("gemma4:31b", "gemma4-31b"),
        ("openai/gpt-4o", "openai-gpt-4o"),
        ("llama3.3:latest", "llama3.3-latest"),
        # Already safe values are left alone.
        ("ollama-default", "ollama-default"),
        ("phi_4", "phi_4"),
        # A run of unsafe characters collapses into one separator.
        ("a   b", "a-b"),
        ("a///b", "a-b"),
        # A value can never grow a `__` that reads as a name separator.
        ("a__b", "a_b"),
        # Leading and trailing punctuation is dropped.
        ("  spaced  ", "spaced"),
        ("--dashes--", "dashes"),
        (".hidden.", "hidden"),
    ],
)
def test_safe_filename_part_cleans_a_value(value, expected):
    """Unsafe characters become `-`, runs collapse, edges are trimmed."""
    assert safe_filename_part(value) == expected


@pytest.mark.parametrize("value", ["../../etc/passwd", "/absolute/path", "..", "./x"])
def test_safe_filename_part_cannot_escape_a_directory(value):
    """No result can carry a path separator or stay a bare `..`."""
    result = safe_filename_part(value)

    assert "/" not in result
    assert "\\" not in result
    assert result != ".."


@pytest.mark.parametrize("value", ["///", "   ", "", "..."])
def test_safe_filename_part_falls_back_when_nothing_is_left(value):
    """A value with no safe characters still names the file something."""
    assert safe_filename_part(value) == "unnamed"


def test_safe_filename_part_truncates_to_max_length():
    """A long value is cut, and the cut cannot leave a trailing separator."""
    assert safe_filename_part("x" * 60) == "x" * 40
    assert safe_filename_part("abcdefghij", max_length=4) == "abcd"
    assert safe_filename_part("abc-defgh", max_length=4) == "abc"


def test_safe_filename_part_accepts_a_non_string():
    """The helper is used on CLI values, which may arrive as None or a number."""
    assert safe_filename_part(42) == "42"
    assert safe_filename_part(None) == "None"
