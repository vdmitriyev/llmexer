"""Unit tests for OpenAICompatibleProvider and the providers built on it.

Covers the parameter mapping of the providers that used to go through the
removed ``LLMRequestsMapper`` (openai, vllm, gemini), the shared call logic they
now inherit, and the two rules that replaced the mapper's fallback: unset values
are omitted from the payload, and a provider with no registered class is
rejected at dispatch.
"""

from unittest.mock import MagicMock

import pytest

from llmexer.base.experiment import COMMON_PARAM_COLUMNS, PROVIDER_PARAM_COLUMNS
from llmexer.base.llm_manager import PROVIDER_CLASS_NAMES, run_experiment_row
from llmexer.base.llm_provider import (
    MAX_TOKENS_TEXT,
    URL_MAP,
    CallerState,
    GeminiProvider,
    LiteLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderAuth,
    VLLMProvider,
    _is_set,
)
from llmexer.exceptions import ProviderConfigException


def _make_completion(text="hello", total_tokens=42, finish_reason="stop"):
    """Build a minimal mock openai ChatCompletion object."""
    completion = MagicMock()
    completion.choices[0].message.content = text
    completion.choices[0].finish_reason = finish_reason
    completion.usage.total_tokens = total_tokens
    return completion


def _mock_client(**kwargs):
    """Return a mock OpenAI client whose chat.completions.create is pre-configured."""
    client = MagicMock()
    client.chat.completions.create.return_value = _make_completion(**kwargs)
    return client


# ---------------------------------------------------------------------------
# _is_set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, float("nan")])
def test_is_set_rejects_unset_values(value):
    assert _is_set(value) is False


@pytest.mark.parametrize("value", [0, 0.0, "", 0.7, "high"])
def test_is_set_accepts_real_values(value):
    # 0 and "" are legitimate parameter values, not "unset".
    assert _is_set(value) is True


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider is a shared implementation, not a usable provider
# ---------------------------------------------------------------------------


def test_shared_base_cannot_be_instantiated():
    """Without a build_request there is no parameter mapping, so no provider."""
    with pytest.raises(TypeError):
        OpenAICompatibleProvider(provider="myproxy")  # type: ignore[abstract]


def test_every_registered_provider_subclasses_the_shared_base():
    import llmexer.base.llm_provider as llm_module

    for class_name in PROVIDER_CLASS_NAMES.values():
        assert issubclass(getattr(llm_module, class_name), OpenAICompatibleProvider)


# ---------------------------------------------------------------------------
# build_request — the mapping each provider replaced _map_params with
# ---------------------------------------------------------------------------


def test_openai_caps_generation_with_max_completion_tokens():
    caller = OpenAIProvider(provider="openai")
    req = caller.build_request("hi", {"model_name": "gpt-4o", "max_tokens": 512, "openai_seed": 42})
    assert req.params["max_completion_tokens"] == 512
    assert req.params["seed"] == 42
    # OpenAI rejects the legacy name on the newer models.
    assert "max_tokens" not in req.params
    assert "extra_body" not in req.params


def test_openai_omits_the_seed_when_the_profile_leaves_it_blank():
    caller = OpenAIProvider(provider="openai")
    req = caller.build_request("hi", {"model_name": "gpt-4o", "max_tokens": None, "openai_seed": None})
    assert "seed" not in req.params
    assert "max_completion_tokens" not in req.params


def test_vllm_sends_max_tokens_and_tunnels_its_own_knobs():
    caller = VLLMProvider(provider="vllm")
    req = caller.build_request(
        "hi",
        {"model_name": "m", "max_tokens": 512, "vllm_min_p": 0.05, "vllm_best_of": 2},
    )
    assert req.params["max_tokens"] == 512
    assert req.params["extra_body"] == {"min_p": 0.05, "best_of": 2}


def test_gemini_tunnels_the_thinking_level():
    caller = GeminiProvider(provider="gemini")
    req = caller.build_request(
        "hi",
        {"model_name": "gemini-2.5-flash", "max_tokens": 512, "gemini_thinking_level": "high"},
    )
    assert req.params["max_tokens"] == 512
    assert req.params["extra_body"] == {"thinking_level": "high"}


def test_no_extra_body_when_the_provider_specific_columns_are_blank():
    caller = VLLMProvider(provider="vllm")
    req = caller.build_request("hi", {"model_name": "m", "vllm_min_p": None, "vllm_best_of": None})
    assert "extra_body" not in req.params


@pytest.mark.parametrize(
    "provider_class, provider",
    [
        (OpenAIProvider, "openai"),
        (OllamaProvider, "ollama"),
        (VLLMProvider, "vllm"),
        (GeminiProvider, "gemini"),
        (LiteLLMProvider, "litellm"),
    ],
)
def test_unset_common_params_are_omitted_not_sent_as_null(provider_class, provider):
    """A blank cell means "not set": sending an explicit null is what the
    removed mapper avoided, and some backends reject it."""
    caller = provider_class(provider=provider)
    req = caller.build_request("hi", {"model_name": "m", "temperature": None, "top_p": float("nan")})
    assert "temperature" not in req.params
    assert "top_p" not in req.params


@pytest.mark.parametrize(
    "provider_class, provider",
    [
        (OpenAIProvider, "openai"),
        (OllamaProvider, "ollama"),
        (VLLMProvider, "vllm"),
        (GeminiProvider, "gemini"),
        (LiteLLMProvider, "litellm"),
    ],
)
def test_default_base_url_matches_the_url_map(provider_class, provider):
    assert provider_class(provider=provider).base_url == URL_MAP[provider]


def test_build_request_only_reads_declared_parameter_columns():
    """A typo'd column name would silently drop a hyperparameter, so pin the
    columns each provider actually consumes against the generated schema."""
    import llmexer.base.llm_provider as llm_module

    for provider, class_name in PROVIDER_CLASS_NAMES.items():
        allowed = set(COMMON_PARAM_COLUMNS) | set(PROVIDER_PARAM_COLUMNS.get(provider, []))
        caller = getattr(llm_module, class_name)(provider=provider)
        # A dict that records every key build_request looks at.
        seen: set = set()

        class _RecordingRow(dict):
            def get(self, key, default=None):
                seen.add(key)
                return super().get(key, default)

        caller.build_request("hi", _RecordingRow(model_name="m"))
        assert seen - {"model_name"} <= allowed, f"{class_name} reads unknown columns"


# ---------------------------------------------------------------------------
# execute — the shared call path
# ---------------------------------------------------------------------------


def test_execute_forwards_the_mapped_params():
    caller = VLLMProvider(provider="vllm")
    caller.session = _mock_client()
    caller.execute("say hi", {"model_name": "m", "max_tokens": 512, "vllm_min_p": 0.05})

    kwargs = caller.session.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["max_tokens"] == 512
    assert kwargs["extra_body"] == {"min_p": 0.05}
    # extra_body must not also leak in as a standard keyword.
    assert "min_p" not in kwargs


def test_execute_passes_no_extra_body_when_there_is_none():
    caller = OpenAIProvider(provider="openai")
    caller.session = _mock_client()
    caller.execute("say hi", {"model_name": "gpt-4o"})
    assert caller.session.chat.completions.create.call_args.kwargs["extra_body"] is None


def test_execute_records_state_and_stats():
    caller = GeminiProvider(provider="gemini")
    caller.session = _mock_client(text="answer", total_tokens=17)
    resp = caller.execute("say hi", {"model_name": "gemini-2.5-flash"})

    assert resp.text == "answer"
    assert resp.usage_tokens == 17
    assert caller.state == CallerState.FINISHED
    assert caller.stats.call_count == 1
    assert caller.stats.total_tokens == 17


def test_execute_records_an_error_without_raising():
    caller = OpenAIProvider(provider="openai")
    caller.session = MagicMock()
    caller.session.chat.completions.create.side_effect = RuntimeError("connection refused")

    resp = caller.execute("say hi", {"model_name": "gpt-4o"})
    assert resp.text == ""
    assert "connection refused" in str(resp.raw)
    assert caller.state == CallerState.ERROR
    assert caller.stats.call_count == 1


# ---------------------------------------------------------------------------
# Truncation — shared by every provider, not just litellm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_class, provider",
    [
        (OpenAIProvider, "openai"),
        (OllamaProvider, "ollama"),
        (VLLMProvider, "vllm"),
        (GeminiProvider, "gemini"),
        (LiteLLMProvider, "litellm"),
    ],
)
def test_a_truncated_answer_is_flagged_on_every_provider(provider_class, provider):
    caller = provider_class(provider=provider)
    caller.session = _mock_client(text="half an ans", finish_reason="length")

    resp = caller.execute("say hi", {"model_name": "m"})
    assert caller.state == CallerState.MAXTOKENREACHED
    assert resp.text == MAX_TOKENS_TEXT
    # The token count is still recorded: the call did consume budget.
    assert resp.usage_tokens == 42


def test_a_complete_answer_is_not_flagged():
    caller = OllamaProvider(provider="ollama")
    caller.session = _mock_client(text="a whole answer", finish_reason="stop")

    resp = caller.execute("say hi", {"model_name": "m"})
    assert caller.state == CallerState.FINISHED
    assert resp.text == "a whole answer"


def test_a_missing_finish_reason_is_not_treated_as_truncation():
    """Not every OpenAI-compatible backend sets finish_reason."""
    caller = OllamaProvider(provider="ollama")
    completion = _make_completion(text="an answer")
    del completion.choices[0].finish_reason
    caller.session = MagicMock()
    caller.session.chat.completions.create.return_value = completion

    resp = caller.execute("say hi", {"model_name": "m"})
    assert caller.state == CallerState.FINISHED
    assert resp.text == "an answer"


# ---------------------------------------------------------------------------
# Dispatch — named providers only
# ---------------------------------------------------------------------------


def test_run_rejects_a_custom_endpoint_without_a_provider_class(monkeypatch):
    """validate_provider() lets a configured custom URL through, but there is no
    parameter mapping for it, so the run aborts instead of guessing one."""
    monkeypatch.setenv("PROVIDER_MYPROXY_URL", "http://myproxy/v1")
    row = {
        "ID": 1,
        "code": "D01_prompt01_m_myproxy-default",
        "prompt": "Hello",
        "model_name": "m",
        "provider_name": "myproxy",
        "profile_name": "myproxy-default",
    }
    with pytest.raises(ProviderConfigException) as exc:
        run_experiment_row(row)
    assert "myproxy" in str(exc.value)


@pytest.mark.parametrize("provider", sorted(URL_MAP))
def test_every_built_in_provider_has_a_class(provider):
    """URL_MAP and PROVIDER_CLASS_NAMES must not drift apart."""
    assert provider in PROVIDER_CLASS_NAMES


def test_dispatch_builds_the_class_registered_for_the_provider(monkeypatch):
    import llmexer.base.llm_provider as llm_module

    built = {}

    class FakeGemini(GeminiProvider):
        def build_session(self) -> None:
            built["caller"] = self
            built["base_url"] = self.base_url
            built["api_key"] = self.auth.api_key
            self.session = _mock_client(text="from gemini")

    monkeypatch.setattr(llm_module, "GeminiProvider", FakeGemini)
    monkeypatch.delenv("PROVIDER_GEMINI_URL", raising=False)
    monkeypatch.setenv("PROVIDER_GEMINI_KEY", "AI-test")  # gitleaks:allow

    experiment = run_experiment_row(
        {
            "ID": 1,
            "code": "D01_prompt01_gemini-2.5-flash_gemini-default",
            "prompt": "Hello",
            "model_name": "gemini-2.5-flash",
            "provider_name": "gemini",
            "profile_name": "gemini-default",
        }
    )
    assert built["base_url"] == URL_MAP["gemini"]
    assert built["api_key"] == "AI-test"  # gitleaks:allow
    assert experiment.response_text == "from gemini"
    assert experiment.state == CallerState.FINISHED.value
    # Fields the removed mapper never filled in for these providers. Compared
    # against the caller's own stats rather than a lower bound: a mocked call can
    # legitimately measure 0.0 where the clock is coarse (~15ms on Windows).
    stats = built["caller"].stats
    assert experiment.call_count == stats.call_count == 1
    assert experiment.elapsed_seconds == stats.elapsed_seconds
    assert experiment.timestamp.endswith("+00:00")


def test_provider_auth_is_used_for_the_client(monkeypatch):
    caller = OpenAIProvider(provider="openai", auth=ProviderAuth(api_key="sk-test"))  # gitleaks:allow
    fake_openai = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    caller.build_session()
    fake_openai.OpenAI.assert_called_once_with(base_url=None, api_key="sk-test")  # gitleaks:allow


def test_timeout_is_forwarded_only_when_set(monkeypatch):
    fake_openai = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    OllamaProvider(provider="ollama", timeout=30.0).build_session()
    assert fake_openai.OpenAI.call_args.kwargs["timeout"] == 30.0

    fake_openai.reset_mock()
    OllamaProvider(provider="ollama").build_session()
    assert "timeout" not in fake_openai.OpenAI.call_args.kwargs
