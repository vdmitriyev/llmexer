"""Unit tests for OllamaProvider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llmexer.base.llm_provider import (
    CallerState,
    OllamaProvider,
    ProviderAuth,
    ProviderRequest,
    ProviderResponse,
)


def _make_completion(text="hello", total_tokens=42):
    """Build a minimal mock openai ChatCompletion object."""
    completion = MagicMock()
    completion.choices[0].message.content = text
    completion.usage.total_tokens = total_tokens
    return completion


def _mock_client(text="hello", total_tokens=42, side_effect=None):
    """Return a mock OpenAI client whose chat.completions.create is pre-configured."""
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = _make_completion(
            text, total_tokens
        )
    return client


def _row(**kwargs):
    base = {
        "param_model_name": "llama3.3:latest",
        "temperature": 0.7,
        "top_p": 1.0,
        "ollama_context_window": None,
        "max_tokens": None,
        "ollama_repeat_penalty": None,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_ollama_provider_is_concrete():
    caller = OllamaProvider(provider="ollama")
    assert caller.provider == "ollama"
    assert caller.state == CallerState.STARTED


def test_ollama_provider_default_base_url():
    caller = OllamaProvider(provider="ollama")
    assert caller.base_url == "http://localhost:11434/v1"


def test_ollama_provider_custom_base_url():
    caller = OllamaProvider(provider="ollama", base_url="http://remote:11434/v1")
    assert caller.base_url == "http://remote:11434/v1"


def test_ollama_provider_custom_auth():
    auth = ProviderAuth(api_key="sk-test")
    caller = OllamaProvider(provider="ollama", auth=auth)
    assert caller.auth.api_key == "sk-test"


# ---------------------------------------------------------------------------
# build_session
# ---------------------------------------------------------------------------


def test_build_session_creates_openai_client():
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    caller = OllamaProvider(provider="ollama")
    assert caller.session is None

    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        caller.build_session()

    assert caller.session is mock_client
    mock_openai_module.OpenAI.assert_called_once_with(
        base_url=caller.base_url, api_key=caller.auth.api_key
    )


# ---------------------------------------------------------------------------
# build_request
# ---------------------------------------------------------------------------


def test_build_request_sets_standard_params():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request("hello", _row())
    assert req.params["temperature"] == 0.7
    assert req.params["top_p"] == 1.0
    assert caller.request is req


def test_build_request_no_extra_body_when_empty():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request("hello", _row())
    assert "extra_body" not in req.params


def test_build_request_ollama_context_window():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request("hello", _row(ollama_context_window=4096))
    assert req.params["extra_body"]["num_ctx"] == 4096


def test_build_request_max_tokens():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request("hello", _row(max_tokens=512))
    assert req.params["extra_body"]["num_predict"] == 512


def test_build_request_repeat_penalty():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request("hello", _row(ollama_repeat_penalty=1.1))
    assert req.params["extra_body"]["repeat_penalty"] == 1.1


def test_build_request_all_extra_body_fields():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request(
        "hello",
        _row(ollama_context_window=4096, max_tokens=512, ollama_repeat_penalty=1.1),
    )
    assert req.params["extra_body"] == {
        "num_ctx": 4096,
        "num_predict": 512,
        "repeat_penalty": 1.1,
    }


def test_build_request_model_from_row():
    caller = OllamaProvider(provider="ollama")
    req = caller.build_request("hello", _row(param_model_name="phi4:14b"))
    assert req.model == "phi4:14b"


# ---------------------------------------------------------------------------
# execute — success path (pre-inject mock session)
# ---------------------------------------------------------------------------


def test_execute_success_state():
    caller = OllamaProvider(provider="ollama")
    caller.session = _mock_client("world", total_tokens=10)
    caller.execute("say hello", _row())
    assert caller.state == CallerState.SUCCESS


def test_execute_success_response_text():
    caller = OllamaProvider(provider="ollama")
    caller.session = _mock_client("world", total_tokens=10)
    resp = caller.execute("say hello", _row())
    assert resp.text == "world"
    assert resp.usage_tokens == 10


def test_execute_success_updates_stats():
    caller = OllamaProvider(provider="ollama")
    caller.session = _mock_client("ok", total_tokens=5)
    caller.execute("first", _row())
    caller.execute("second", _row())
    assert caller.stats.call_count == 2
    assert caller.stats.total_tokens == 10
    assert caller.stats.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# execute — error path
# ---------------------------------------------------------------------------


def test_execute_error_state():
    caller = OllamaProvider(provider="ollama")
    caller.session = _mock_client(side_effect=RuntimeError("connection refused"))
    resp = caller.execute("hello", _row())
    assert caller.state == CallerState.ERROR
    assert resp.text == ""
    assert "connection refused" in str(resp.raw)


def test_execute_error_still_increments_call_count():
    caller = OllamaProvider(provider="ollama")
    caller.session = _mock_client(side_effect=RuntimeError("oops"))
    caller.execute("hello", _row())
    assert caller.stats.call_count == 1


# ---------------------------------------------------------------------------
# execute — session reuse
# ---------------------------------------------------------------------------


def test_execute_reuses_existing_session():
    caller = OllamaProvider(provider="ollama")
    mock_client = _mock_client()
    caller.session = mock_client
    caller.execute("hello", _row())
    # session object must be the same instance — build_session was not called
    assert caller.session is mock_client
