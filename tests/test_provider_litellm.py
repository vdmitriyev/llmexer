"""Unit tests for LiteLLMProvider (vLLM models served behind a LiteLLM proxy)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llmexer.base.llm_provider import (
    URL_MAP,
    CallerState,
    LiteLLMProvider,
    ProviderAuth,
)
from llmexer.common import get_user_agent
from llmexer.exceptions import ProviderConfigException

_URL = "https://proxy.example.org/v1"
_TOKEN = "sk-litellm-test"  # gitleaks:allow


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
        client.chat.completions.create.return_value = _make_completion(text, total_tokens)
    return client


def _row(**kwargs):
    base = {
        "model_name": "gpt-oss:120b",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": None,
        "litellm_min_p": None,
        "litellm_best_of": None,
    }
    base.update(kwargs)
    return base


def _caller(**kwargs):
    """A fully configured provider (URL + token), unless overridden."""
    kwargs.setdefault("base_url", _URL)
    kwargs.setdefault("auth", ProviderAuth(api_key=_TOKEN))
    return LiteLLMProvider(provider="litellm", **kwargs)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_litellm_provider_is_concrete():
    caller = _caller()
    assert caller.provider == "litellm"
    assert caller.state == CallerState.STARTED


def test_litellm_provider_has_no_default_base_url():
    # A proxy is always remote and site-specific: nothing sensible to default to.
    assert URL_MAP["litellm"] is None
    assert LiteLLMProvider(provider="litellm").base_url is None


def test_litellm_provider_custom_base_url():
    assert _caller(base_url="http://other/v1").base_url == "http://other/v1"


def test_litellm_provider_custom_auth():
    assert _caller().auth.api_key == _TOKEN


# ---------------------------------------------------------------------------
# validate_config — the token/URL are mandatory for this provider
# ---------------------------------------------------------------------------


def test_validate_config_passes_when_fully_configured():
    assert _caller().validate_config() is None


def test_validate_config_raises_without_base_url():
    caller = _caller(base_url=None)
    with pytest.raises(ProviderConfigException) as exc:
        caller.validate_config()
    assert "PROVIDER_LITELLM_URL" in str(exc.value)


def test_validate_config_raises_when_api_key_is_placeholder():
    caller = _caller(auth=ProviderAuth())  # defaults to the "na" placeholder
    with pytest.raises(ProviderConfigException) as exc:
        caller.validate_config()
    assert "PROVIDER_LITELLM_KEY" in str(exc.value)


def test_validate_config_raises_when_api_key_is_empty():
    caller = _caller(auth=ProviderAuth(api_key=""))
    with pytest.raises(ProviderConfigException):
        caller.validate_config()


def test_validate_config_error_never_leaks_the_token():
    caller = _caller(base_url=None)
    with pytest.raises(ProviderConfigException) as exc:
        caller.validate_config()
    assert _TOKEN not in str(exc.value)


def test_api_key_is_kept_out_of_reprs():
    # Typer renders tracebacks with locals shown, so a token in a repr would
    # be printed to the terminal and the log on any error.
    caller = _caller()
    assert _TOKEN not in repr(caller)
    assert _TOKEN not in repr(caller.auth)
    assert caller.auth.api_key == _TOKEN


# ---------------------------------------------------------------------------
# build_session
# ---------------------------------------------------------------------------


def test_build_session_creates_openai_client():
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    caller = _caller()
    assert caller.session is None

    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        caller.build_session()

    assert caller.session is mock_client
    mock_openai_module.OpenAI.assert_called_once_with(
        base_url=_URL,
        api_key=_TOKEN,
        default_headers={"User-Agent": get_user_agent()},
    )


def test_build_session_validates_before_creating_client():
    mock_openai_module = MagicMock()
    caller = _caller(auth=ProviderAuth())

    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        with pytest.raises(ProviderConfigException):
            caller.build_session()

    mock_openai_module.OpenAI.assert_not_called()
    assert caller.session is None


# ---------------------------------------------------------------------------
# build_request
# ---------------------------------------------------------------------------


def test_build_request_sets_standard_params():
    caller = _caller()
    req = caller.build_request("hello", _row())
    assert req.params["temperature"] == 0.7
    assert req.params["top_p"] == 0.9
    assert caller.request is req


def test_build_request_no_extra_body_when_empty():
    req = _caller().build_request("hello", _row())
    assert "extra_body" not in req.params


def test_build_request_max_tokens_is_a_standard_param():
    # Unlike ollama's num_predict, the proxy takes max_tokens directly.
    req = _caller().build_request("hello", _row(max_tokens=512))
    assert req.params["max_tokens"] == 512
    assert "extra_body" not in req.params


def test_build_request_omits_max_tokens_when_absent():
    assert "max_tokens" not in _caller().build_request("hello", _row()).params


def test_build_request_min_p():
    req = _caller().build_request("hello", _row(litellm_min_p=0.05))
    assert req.params["extra_body"]["min_p"] == 0.05


def test_build_request_best_of():
    req = _caller().build_request("hello", _row(litellm_best_of=2))
    assert req.params["extra_body"]["best_of"] == 2


def test_build_request_all_extra_body_fields():
    req = _caller().build_request("hello", _row(litellm_min_p=0.05, litellm_best_of=2, max_tokens=512))
    assert req.params["extra_body"] == {"min_p": 0.05, "best_of": 2}
    assert req.params["max_tokens"] == 512


def test_build_request_model_from_row():
    req = _caller().build_request("hello", _row(model_name="gemma4:31b"))
    assert req.model == "gemma4:31b"


# ---------------------------------------------------------------------------
# execute — success path (pre-inject mock session)
# ---------------------------------------------------------------------------


def test_execute_success_state():
    caller = _caller()
    caller.session = _mock_client("world", total_tokens=10)
    caller.execute("say hello", _row())
    assert caller.state == CallerState.FINISHED


def test_execute_success_response_text():
    caller = _caller()
    caller.session = _mock_client("world", total_tokens=10)
    resp = caller.execute("say hello", _row())
    assert resp.text == "world"
    assert resp.usage_tokens == 10


def test_execute_success_updates_stats():
    caller = _caller()
    caller.session = _mock_client("ok", total_tokens=5)
    caller.execute("first", _row())
    caller.execute("second", _row())
    assert caller.stats.call_count == 2
    assert caller.stats.total_tokens == 10
    assert caller.stats.elapsed_seconds >= 0


def test_execute_forwards_params_to_the_proxy():
    caller = _caller()
    client = _mock_client()
    caller.session = client
    caller.execute("say hello", _row(max_tokens=512, litellm_min_p=0.05, litellm_best_of=2))
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-oss:120b"
    assert kwargs["messages"] == [{"role": "user", "content": "say hello"}]
    assert kwargs["max_tokens"] == 512
    assert kwargs["extra_body"] == {"min_p": 0.05, "best_of": 2}


def test_execute_sends_no_extra_body_when_empty():
    caller = _caller()
    client = _mock_client()
    caller.session = client
    caller.execute("say hello", _row())
    assert client.chat.completions.create.call_args.kwargs["extra_body"] is None


# ---------------------------------------------------------------------------
# execute — error path
# ---------------------------------------------------------------------------


def test_execute_error_state():
    caller = _caller()
    caller.session = _mock_client(side_effect=RuntimeError("connection refused"))
    resp = caller.execute("hello", _row())
    assert caller.state == CallerState.ERROR
    assert resp.text == ""
    assert "connection refused" in str(resp.raw)


def test_execute_error_still_increments_call_count():
    caller = _caller()
    caller.session = _mock_client(side_effect=RuntimeError("oops"))
    caller.execute("hello", _row())
    assert caller.stats.call_count == 1


# ---------------------------------------------------------------------------
# execute — session reuse / raw response capture
# ---------------------------------------------------------------------------


def test_execute_reuses_existing_session():
    caller = _caller()
    mock_client = _mock_client()
    caller.session = mock_client
    caller.execute("hello", _row())
    assert caller.session is mock_client


def test_execute_keeps_full_raw_completion():
    caller = _caller()
    client = _mock_client("world", total_tokens=10)
    caller.session = client
    resp = caller.execute("say hello", _row())
    assert resp.raw is client.chat.completions.create.return_value


# ---------------------------------------------------------------------------
# Provider-name validation
#
# A provider name that resolves to no endpoint used to fall through to the
# OpenAI SDK default host, yielding a 401 that blamed OpenAI. Regression guard.
# ---------------------------------------------------------------------------


def test_is_known_provider_accepts_built_ins():
    from llmexer.base.llm_provider import is_known_provider

    for name in ("ollama", "vllm", "litellm", "openai", "gemini"):
        assert is_known_provider(name) is True


def test_is_known_provider_rejects_a_profile_name():
    from llmexer.base.llm_provider import is_known_provider

    assert is_known_provider("litellm-default") is False
    assert is_known_provider("") is False
    assert is_known_provider(None) is False


def test_validate_provider_normalises_case_and_spacing():
    from llmexer.base.llm_provider import validate_provider

    assert validate_provider("  LiteLLM ") == "litellm"


def test_validate_provider_rejects_a_profile_name(monkeypatch):
    from llmexer.base.llm_provider import validate_provider

    monkeypatch.delenv("PROVIDER_LITELLM-DEFAULT_URL", raising=False)
    with pytest.raises(ProviderConfigException) as exc:
        validate_provider("litellm-default")
    message = str(exc.value)
    assert "litellm-default" in message
    # The message must point at the real cause, not at OpenAI.
    assert "llms-for-experiment.csv" in message
    assert "litellm" in message


def test_validate_provider_allows_a_configured_custom_endpoint(monkeypatch):
    from llmexer.base.llm_provider import validate_provider

    monkeypatch.setenv("PROVIDER_MYPROXY_URL", "http://myproxy/v1")
    assert validate_provider("myproxy") == "myproxy"


def test_run_experiment_row_rejects_an_unreachable_provider(monkeypatch):
    """The row that caused the 401: provider column held a profile name."""
    from llmexer.base.llm_manager import run_experiment_row

    monkeypatch.delenv("PROVIDER_LITELLM-DEFAULT_URL", raising=False)
    row = {
        "ID": 1,
        "code": "D01_prompt01_gpt-oss:120b_litellm-default",
        "prompt": "Hello",
        "model_name": "gpt-oss:120b",
        "provider_name": "litellm-default",
        "profile_name": "litellm-default",
    }
    with pytest.raises(ProviderConfigException):
        run_experiment_row(row)
