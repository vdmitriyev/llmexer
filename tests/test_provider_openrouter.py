"""Unit tests for OpenRouterProvider (models reached through the OpenRouter gateway)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llmexer.base.llm_manager import run_experiment_row
from llmexer.base.llm_provider import (
    URL_MAP,
    CallerState,
    OpenRouterProvider,
    ProviderAuth,
    resolve_provider_config,
)
from llmexer.common import get_user_agent
from llmexer.exceptions import ProviderConfigException

_MODEL = "google/gemini-3.8-flash"
_TOKEN = "sk-or-test"  # gitleaks:allow


def _make_completion(text="hello", total_tokens=42):
    """Build a minimal mock openai ChatCompletion object."""
    completion = MagicMock()
    completion.choices[0].message.content = text
    completion.usage.total_tokens = total_tokens
    completion.usage.prompt_tokens, completion.usage.completion_tokens = 10, 32
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
        "model_name": _MODEL,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": None,
        "openrouter_provider_order": None,
        "openrouter_reasoning_effort": None,
    }
    base.update(kwargs)
    return base


def _caller(**kwargs):
    """A configured provider (token set, URL left at its default), unless overridden."""
    kwargs.setdefault("auth", ProviderAuth(api_key=_TOKEN))
    return OpenRouterProvider(provider="openrouter", **kwargs)


# ---------------------------------------------------------------------------
# base URL
# ---------------------------------------------------------------------------


def test_default_base_url_is_the_public_gateway():
    """The gateway is one fixed host, so the URL needs no .env entry."""
    assert URL_MAP["openrouter"] == "https://openrouter.ai/api/v1"
    assert _caller().base_url == URL_MAP["openrouter"]


def test_base_url_falls_back_to_the_url_map(monkeypatch):
    monkeypatch.delenv("PROVIDER_OPENROUTER_URL", raising=False)
    base_url, _ = resolve_provider_config("openrouter")
    assert base_url == URL_MAP["openrouter"]


def test_base_url_env_var_overrides_the_default(monkeypatch):
    monkeypatch.setenv("PROVIDER_OPENROUTER_URL", "https://gateway.example.org/v1")
    base_url, _ = resolve_provider_config("openrouter")
    assert base_url == "https://gateway.example.org/v1"


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def test_validate_config_raises_when_api_key_is_placeholder():
    caller = _caller(auth=ProviderAuth())  # defaults to the "na" placeholder
    with pytest.raises(ProviderConfigException) as exc:
        caller.validate_config()
    assert "PROVIDER_OPENROUTER_KEY" in str(exc.value)


def test_validate_config_passes_without_a_url_env_var():
    """Only the token is mandatory: the default URL is enough to reach the gateway."""
    _caller().validate_config()


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
        base_url=URL_MAP["openrouter"],
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
    assert req.params["top_p"] == 1.0
    assert caller.request is req


def test_build_request_model_carries_its_vendor():
    req = _caller().build_request("hello", _row())
    assert req.model == _MODEL


def test_build_request_max_tokens_is_a_standard_param():
    req = _caller().build_request("hello", _row(max_tokens=512))
    assert req.params["max_tokens"] == 512


def test_build_request_omits_max_tokens_when_absent():
    assert "max_tokens" not in _caller().build_request("hello", _row()).params


def test_build_request_no_extra_body_when_the_provider_columns_are_blank():
    req = _caller().build_request("hello", _row())
    assert "extra_body" not in req.params


def test_build_request_nan_cells_are_dropped():
    """``experiment try`` builds its row with pandas, so a blank cell is NaN."""
    req = _caller().build_request(
        "hello",
        _row(openrouter_provider_order=float("nan"), openrouter_reasoning_effort=float("nan")),
    )
    assert "extra_body" not in req.params


def test_build_request_provider_order_becomes_a_routing_list():
    req = _caller().build_request("hello", _row(openrouter_provider_order="Google"))
    assert req.params["extra_body"]["provider"] == {"order": ["Google"]}


def test_build_request_provider_order_splits_on_commas():
    req = _caller().build_request("hello", _row(openrouter_provider_order="Google, DeepInfra"))
    assert req.params["extra_body"]["provider"] == {"order": ["Google", "DeepInfra"]}


def test_build_request_ignores_an_all_blank_provider_order():
    req = _caller().build_request("hello", _row(openrouter_provider_order=" , "))
    assert "extra_body" not in req.params


def test_build_request_reasoning_effort():
    req = _caller().build_request("hello", _row(openrouter_reasoning_effort="low"))
    assert req.params["extra_body"]["reasoning"] == {"effort": "low"}


def test_build_request_all_extra_body_fields():
    req = _caller().build_request(
        "hello",
        _row(openrouter_provider_order="Google", openrouter_reasoning_effort="high", max_tokens=512),
    )
    assert req.params["extra_body"] == {
        "provider": {"order": ["Google"]},
        "reasoning": {"effort": "high"},
    }
    assert req.params["max_tokens"] == 512


# ---------------------------------------------------------------------------
# _extract_cost
# ---------------------------------------------------------------------------


def test_cost_comes_from_usage_cost():
    """OpenRouter reports what a call cost in the response body."""
    completion = _make_completion()
    completion.usage.cost = 0.0031

    assert _caller()._extract_cost(completion) == 0.0031


def test_a_header_cost_wins_when_one_is_sent():
    """Honoured if the gateway ever grows the header; the body is the fallback."""
    completion = _make_completion()
    completion.usage.cost = 0.0031
    completion.headers = {"X-OpenRouter-Cost": 0.5}

    assert _caller()._extract_cost(completion) == 0.5


def test_cost_is_none_when_the_response_reports_none():
    """A MagicMock attribute must not be coerced: float(MagicMock()) is 1.0.

    Without the strict type check this test would report a dollar per mocked
    call and every other test in the suite would quietly spend the budget.
    """
    completion = _make_completion()  # usage.cost never set -> a MagicMock

    assert _caller()._extract_cost(completion) is None


def test_a_zero_cost_is_reported_as_zero_not_none():
    completion = _make_completion()
    completion.usage.cost = 0.0

    assert _caller()._extract_cost(completion) == 0.0


def test_other_providers_report_no_cost():
    """The hook is a no-op everywhere but the paid gateway."""
    from llmexer.base.llm_provider import OllamaProvider

    completion = _make_completion()
    completion.usage.cost = 0.0031

    assert OllamaProvider(provider="ollama")._extract_cost(completion) is None


def test_execute_puts_the_cost_on_the_response_and_the_stats():
    caller = _caller()
    completion = _make_completion(text="paid answer")
    completion.usage.cost = 0.0025
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    caller.session = client

    resp = caller.execute("say hi", _row())

    assert resp.cost_usd == 0.0025
    assert caller.stats.cost_usd == 0.0025


def test_stats_cost_accumulates_across_calls():
    caller = _caller()
    completion = _make_completion()
    completion.usage.cost = 0.002
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    caller.session = client

    caller.execute("one", _row())
    caller.execute("two", _row())

    assert caller.stats.cost_usd == pytest.approx(0.004)


def test_an_unreported_cost_leaves_the_stats_at_zero():
    caller = _caller()
    caller.session = _mock_client()  # no usage.cost on the completion

    resp = caller.execute("say hi", _row())

    assert resp.cost_usd is None
    assert caller.stats.cost_usd == 0.0


# ---------------------------------------------------------------------------
# execute / dispatch
# ---------------------------------------------------------------------------


def test_execute_sends_the_extra_body_and_returns_the_answer():
    caller = _caller()
    caller.session = _mock_client(text="from openrouter")

    resp = caller.execute("say hi", _row(openrouter_reasoning_effort="low"))

    assert resp.text == "from openrouter"
    assert caller.state == CallerState.FINISHED
    kwargs = caller.session.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == _MODEL
    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


def test_dispatch_builds_the_openrouter_class(monkeypatch):
    """``run_experiment_row`` must resolve the class registered for the provider."""
    import llmexer.base.llm_provider as llm_module

    built = {}

    class FakeOpenRouter(OpenRouterProvider):
        def build_session(self) -> None:
            built["base_url"] = self.base_url
            built["api_key"] = self.auth.api_key
            self.session = _mock_client(text="routed answer")

    monkeypatch.setattr(llm_module, "OpenRouterProvider", FakeOpenRouter)
    monkeypatch.delenv("PROVIDER_OPENROUTER_URL", raising=False)
    monkeypatch.setenv("PROVIDER_OPENROUTER_KEY", _TOKEN)

    experiment = run_experiment_row(
        {
            "ID": 1,
            "code": f"D01_prompt01_{_MODEL}_openrouter-gemini-default",
            "prompt": "Hello",
            "model_name": _MODEL,
            "provider_name": "openrouter",
            "profile_name": "openrouter-gemini-default",
        }
    )

    assert built["base_url"] == URL_MAP["openrouter"]
    assert built["api_key"] == _TOKEN
    assert experiment.response_text == "routed answer"
    assert experiment.state == CallerState.FINISHED.value


def test_dispatch_aborts_when_the_token_is_missing(monkeypatch):
    """A missing key stops the run up front instead of becoming a per-row 401."""
    import llmexer.base.llm_provider as llm_module

    monkeypatch.delenv("PROVIDER_OPENROUTER_KEY", raising=False)
    called = []

    class ExplodingProvider(llm_module.OpenRouterProvider):
        def execute(self, prompt, row):  # pragma: no cover - must not run
            called.append(prompt)
            raise AssertionError("execute must not be reached")

    monkeypatch.setattr(llm_module, "OpenRouterProvider", ExplodingProvider)

    with pytest.raises(ProviderConfigException):
        run_experiment_row(
            {
                "ID": 1,
                "code": f"D01_prompt01_{_MODEL}_openrouter-gemini-default",
                "prompt": "Hello",
                "model_name": _MODEL,
                "provider_name": "openrouter",
                "profile_name": "openrouter-gemini-default",
            }
        )
    assert called == []
