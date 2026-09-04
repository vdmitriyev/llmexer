"""Tests for the OpenAlex result ceiling (MAX_OPEN_ALEX_RESPONSES)."""

from unittest.mock import Mock, patch

import pytest

from llmexer.base.search_openalex import run_openalex_search
from llmexer.constants import DEFAULT_MAX_OPENALEX_RESPONSES

_CAP_MESSAGE_MARKER = "capped by configs to"


def _work(index: int) -> dict:
    """A minimal OpenAlex work with just enough fields to be flattened."""
    return {
        "id": f"https://openalex.org/W{index}",
        "display_name": f"Paper {index}",
        "publication_year": 2023,
        "authorships": [],
        "doi": None,
    }


def _page(works: list[dict], total: int, next_cursor):
    """Build a mock OpenAlex page response."""
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {
        "results": works,
        "meta": {"count": total, "next_cursor": next_cursor},
    }
    return resp


@pytest.fixture()
def patch_session():
    """Patch the OpenAlex HTTP session; the returned setter installs the page responses."""
    mock_session = Mock()

    def _install(pages):
        mock_session.get.side_effect = pages
        return mock_session

    with (
        patch(
            "llmexer.base.search_openalex.make_http_session",
            return_value=mock_session,
        ),
        patch(
            "llmexer.base.search_openalex.detect_publication_lang",
            return_value="en",
        ),
    ):
        yield _install


def _run(tmp_path, on_progress=None, limit_size=None):
    return run_openalex_search(
        query="q",
        year="2020-2025",
        only_open_access=False,
        batch_size=100,
        limit_size=limit_size,
        json_path=str(tmp_path / "raw.json"),
        api_key="key",
        on_progress=on_progress,
    )


def test_cap_enforced_and_message_emitted(tmp_path, patch_session, monkeypatch):
    """With the cap at 150, a large query yields exactly 150 records and warns once."""
    monkeypatch.setenv("MAX_OPEN_ALEX_RESPONSES", "150")
    patch_session(
        [
            _page([_work(i) for i in range(100)], total=1000, next_cursor="c1"),
            _page([_work(i) for i in range(100, 200)], total=1000, next_cursor="c2"),
        ]
    )

    messages: list[str] = []
    records = _run(tmp_path, on_progress=messages.append)

    assert len(records) == 150
    assert any(_CAP_MESSAGE_MARKER in m and "150" in m for m in messages)


def test_env_override_respected(tmp_path, patch_session, monkeypatch):
    """MAX_OPEN_ALEX_RESPONSES=1 truncates the output to a single record."""
    monkeypatch.setenv("MAX_OPEN_ALEX_RESPONSES", "1")
    patch_session([_page([_work(i) for i in range(100)], total=1000, next_cursor="c1")])

    records = _run(tmp_path)

    assert len(records) == 1


def test_no_warning_when_under_cap(tmp_path, patch_session, monkeypatch):
    """A query with fewer results than the cap returns everything and does not warn."""
    monkeypatch.delenv("MAX_OPEN_ALEX_RESPONSES", raising=False)
    patch_session([_page([_work(i) for i in range(3)], total=3, next_cursor=None)])

    messages: list[str] = []
    records = _run(tmp_path, on_progress=messages.append)

    assert len(records) == 3
    assert not any(_CAP_MESSAGE_MARKER in m for m in messages)


def test_explicit_limit_smaller_than_cap_wins_without_cap_message(tmp_path, patch_session, monkeypatch):
    """An explicit --limit below the cap bounds the output and suppresses the cap warning."""
    monkeypatch.delenv("MAX_OPEN_ALEX_RESPONSES", raising=False)
    patch_session([_page([_work(i) for i in range(100)], total=1000, next_cursor="c1")])

    messages: list[str] = []
    records = _run(tmp_path, on_progress=messages.append, limit_size=5)

    assert len(records) == 5
    assert not any(_CAP_MESSAGE_MARKER in m for m in messages)


def test_invalid_env_value_falls_back_to_default(tmp_path, patch_session, monkeypatch):
    """A non-integer MAX_OPEN_ALEX_RESPONSES falls back to the default (no crash)."""
    monkeypatch.setenv("MAX_OPEN_ALEX_RESPONSES", "abc")
    assert DEFAULT_MAX_OPENALEX_RESPONSES >= 3
    patch_session([_page([_work(i) for i in range(3)], total=3, next_cursor=None)])

    records = _run(tmp_path)

    assert len(records) == 3
