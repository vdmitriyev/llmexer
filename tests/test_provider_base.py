"""Tests for LLMProviderBase and its supporting types."""

import pytest

from llmexer.base.llm_provider import (
    CallerState,
    CallerStats,
    LLMProviderBase,
    ProviderAuth,
    ProviderRequest,
    ProviderResponse,
)

# ---------------------------------------------------------------------------
# Minimal concrete stub for testing the abstract base
# ---------------------------------------------------------------------------


class _StubProvider(LLMProviderBase):
    def build_session(self) -> None:
        self.session = object()

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        self.request = ProviderRequest(model=row.get("model", "stub"), prompt=prompt)
        return self.request

    def execute(self, prompt: str, row: dict) -> ProviderResponse:
        self.state = CallerState.RUNNING
        try:
            self.build_session()
            self.build_request(prompt, row)
            self.response = ProviderResponse(text="ok", usage_tokens=10)
            self.stats.call_count += 1
            self.stats.total_tokens += 10
            self.state = CallerState.FINISHED
        except Exception as exc:
            self.state = CallerState.ERROR
            self.response = ProviderResponse(text=str(exc))
        return self.response


# ---------------------------------------------------------------------------
# CallerState tests
# ---------------------------------------------------------------------------


def test_caller_state_values():
    assert CallerState.STARTED == "started"
    assert CallerState.INCOMPLETE == "incomplete"
    assert CallerState.MAXTOKENREACHED == "maxtokenreached"
    assert CallerState.RUNNING == "running"
    assert CallerState.FINISHED == "finished"
    assert CallerState.ERROR == "error"
    assert set(CallerState) == {
        CallerState.STARTED,
        CallerState.INCOMPLETE,
        CallerState.MAXTOKENREACHED,
        CallerState.RUNNING,
        CallerState.FINISHED,
        CallerState.ERROR,
    }


def test_caller_state_is_str():
    assert isinstance(CallerState.FINISHED, str)


# ---------------------------------------------------------------------------
# ProviderAuth tests
# ---------------------------------------------------------------------------


def test_provider_auth_defaults():
    auth = ProviderAuth()
    assert auth.api_key == "na"
    assert auth.extra_headers == {}


def test_provider_auth_custom():
    auth = ProviderAuth(api_key="sk-test", extra_headers={"X-Org": "acme"})
    assert auth.api_key == "sk-test"
    assert auth.extra_headers == {"X-Org": "acme"}


# ---------------------------------------------------------------------------
# ProviderRequest tests
# ---------------------------------------------------------------------------


def test_provider_request_fields():
    req = ProviderRequest(model="llama3", prompt="hello")
    assert req.model == "llama3"
    assert req.prompt == "hello"
    assert req.params == {}


def test_provider_request_with_params():
    req = ProviderRequest(model="gpt-4", prompt="hi", params={"temperature": 0.7})
    assert req.params["temperature"] == 0.7


# ---------------------------------------------------------------------------
# ProviderResponse tests
# ---------------------------------------------------------------------------


def test_provider_response_defaults():
    resp = ProviderResponse()
    assert resp.text == ""
    assert resp.usage_tokens is None
    assert resp.raw is None


def test_provider_response_with_values():
    resp = ProviderResponse(text="answer", usage_tokens=42, raw={"id": "xyz"})
    assert resp.text == "answer"
    assert resp.usage_tokens == 42
    assert resp.raw == {"id": "xyz"}


# ---------------------------------------------------------------------------
# CallerStats tests
# ---------------------------------------------------------------------------


def test_caller_stats_defaults():
    stats = CallerStats()
    assert stats.call_count == 0
    assert stats.total_tokens == 0
    assert stats.elapsed_seconds == 0.0


# ---------------------------------------------------------------------------
# LLMProviderBase abstract enforcement
# ---------------------------------------------------------------------------


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        LLMProviderBase(provider="ollama")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Stub subclass behaviour tests
# ---------------------------------------------------------------------------


def test_concrete_stub_initial_state():
    stub = _StubProvider(provider="stub")
    assert stub.state == CallerState.STARTED
    assert stub.data is None
    assert stub.request is None
    assert stub.response is None
    assert stub.session is None


def test_stub_auth_defaults():
    stub = _StubProvider(provider="stub")
    assert stub.auth.api_key == "na"


def test_stub_timeout_default():
    stub = _StubProvider(provider="stub")
    assert stub.timeout is None


def test_stub_execute_updates_state():
    stub = _StubProvider(provider="stub")
    stub.execute("say hello", {"model": "stub-model"})
    assert stub.state == CallerState.FINISHED


def test_stub_execute_populates_response():
    stub = _StubProvider(provider="stub")
    resp = stub.execute("say hello", {"model": "stub-model"})
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "ok"
    assert resp.usage_tokens == 10


def test_stub_execute_updates_stats():
    stub = _StubProvider(provider="stub")
    stub.execute("first", {})
    stub.execute("second", {})
    assert stub.stats.call_count == 2
    assert stub.stats.total_tokens == 20


def test_stub_build_session_sets_session():
    stub = _StubProvider(provider="stub")
    assert stub.session is None
    stub.build_session()
    assert stub.session is not None


def test_stub_build_request_sets_request():
    stub = _StubProvider(provider="stub")
    req = stub.build_request("hi", {"model": "m"})
    assert isinstance(req, ProviderRequest)
    assert stub.request is req
    assert req.model == "m"
    assert req.prompt == "hi"


def test_stub_custom_auth():
    auth = ProviderAuth(api_key="sk-abc")
    stub = _StubProvider(provider="openai", auth=auth, timeout=30.0)
    assert stub.auth.api_key == "sk-abc"
    assert stub.timeout == 30.0
