"""LLM provider abstractions, configuration, and concrete provider callers."""

import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from llmexer.base.experiment import FILE_LLM_PARAMS, FILE_LLMS_FOR_EXPERIMENT
from llmexer.configs import logger
from llmexer.exceptions import ProviderConfigException

URL_MAP: Dict[str, Optional[str]] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "openai": None,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    # A LiteLLM proxy is always remote and site-specific: there is no sensible
    # default, so PROVIDER_LITELLM_URL must be set explicitly.
    "litellm": None,
}


def is_known_provider(provider: str) -> bool:
    """Whether ``provider`` is one of the built-in providers in :data:`URL_MAP`."""

    return str(provider or "").strip().lower() in URL_MAP


def validate_provider(provider: str) -> str:
    """Return the normalised provider name, raising if it cannot be reached.

    A provider is usable when it is either built in (:data:`URL_MAP`) or has an
    explicit ``PROVIDER_<UPPER>_URL`` configured. Anything else has no
    resolvable endpoint and would otherwise fall through to the OpenAI SDK's
    default host (``api.openai.com``) with the ``"na"`` API key placeholder,
    producing a confusing ``401`` that mentions OpenAI even when no OpenAI model
    is involved.

    Args:
        provider (str): provider name as written in ``llms-for-experiment.csv``.

    Returns:
        str: the lower-cased provider name.

    Raises:
        ProviderConfigException: if the provider is neither built in nor
            explicitly pointed at a base URL.
    """

    normalised = str(provider or "").strip().lower()
    if normalised in URL_MAP:
        return normalised
    if os.environ.get(f"PROVIDER_{normalised.upper()}_URL"):
        # A deliberately configured custom endpoint.
        return normalised
    raise ProviderConfigException(
        f"Unknown LLM provider '{provider}'. Supported providers: "
        f"{', '.join(sorted(URL_MAP))}. Check the 'provider' column of "
        f"{FILE_LLMS_FOR_EXPERIMENT} — it takes a provider name (e.g. 'litellm'), "
        "not a "
        f"profile name from {FILE_LLM_PARAMS} (e.g. 'litellm-default'); a profile "
        "belongs in the 'profile_name' column. To use a "
        f"custom endpoint, set PROVIDER_{normalised.upper()}_URL."
    )


def resolve_provider_config(provider: str) -> Tuple[Optional[str], str]:
    """Resolve the base URL and API key for a provider.

    Resolution order matches the experiment run command:
      * base URL: ``PROVIDER_<UPPER>_URL`` env var -> ``URL_MAP`` -> ``None``
      * api key:  ``PROVIDER_<UPPER>_KEY`` env var -> ``"na"``

    Args:
        provider (str): provider name (e.g. ``ollama``, ``openai``).

    Returns:
        tuple[str | None, str]: ``(base_url, api_key)``.
    """

    provider = provider.lower()
    provider_upper = provider.upper()
    base_url = os.environ.get(f"PROVIDER_{provider_upper}_URL") or URL_MAP.get(provider)
    api_key = os.environ.get(f"PROVIDER_{provider_upper}_KEY") or "na"
    return base_url, api_key


def serialize_response(obj: Any) -> Optional[Dict[str, Any]]:
    """Best-effort JSON-safe dict of an SDK response object.

    The OpenAI SDK response models allow extra fields (``extra="allow"``), so
    ``model_dump`` preserves provider-specific extras (e.g. ollama's
    ``eval_count`` / ``*_duration``). Returns ``None`` when ``obj`` is not a
    serializable model (e.g. an error string) or if the dump fails.
    """

    if obj is not None and hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return None
    return None


def _is_set(value: Any) -> bool:
    """Whether a row value should be sent to the provider.

    A blank cell in ``llm-params.csv`` means "this profile does not set it", and
    reaches here either as ``None`` (rows joined by the DAO) or as ``float("nan")``
    (rows assembled by ``experiment try``). Both must be dropped rather than sent:
    an explicit JSON ``null`` is rejected by some backends, and ``NaN`` is not
    serializable by the SDK at all.
    """

    if value is None:
        return False
    return not (isinstance(value, float) and math.isnan(value))


class CallerState(str, Enum):
    STARTED = "started"
    INCOMPLETE = "incomplete"
    MAXTOKENREACHED = "maxtokenreached"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class ProviderAuth:
    # ``repr=False``: Typer renders tracebacks with locals shown, so anything in
    # a dataclass repr ends up on screen and in the log. Keep the token out.
    api_key: str = field(default="na", repr=False)
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderRequest:
    model: str
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str = ""
    usage_tokens: Optional[int] = None
    raw: Optional[Any] = field(default=None, repr=False)


@dataclass
class CallerStats:
    call_count: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class LLMProviderBase(ABC):
    provider: str

    auth: ProviderAuth = field(default_factory=ProviderAuth)
    timeout: Optional[float] = None

    data: Optional[dict[str, Any]] = field(default=None, repr=False)
    request: Optional[ProviderRequest] = field(default=None, repr=False)
    response: Optional[ProviderResponse] = field(default=None, repr=False)
    session: Optional[Any] = field(default=None, repr=False)

    state: CallerState = field(default=CallerState.STARTED)
    stats: CallerStats = field(default_factory=CallerStats)

    @abstractmethod
    def build_session(self) -> None:
        """Initialise self.session (e.g. openai.OpenAI client)."""

    @abstractmethod
    def build_request(self, prompt: str, row: dict[str, Any]) -> ProviderRequest:
        """Translate prompt + CSV row into a ProviderRequest and set self.request."""

    @abstractmethod
    def execute(self, prompt: str, row: dict[str, Any]) -> ProviderResponse:
        """Run the call; update self.state, self.response, self.stats; return ProviderResponse."""


MAX_TOKENS_TEXT = "No answer. Max token reached"


@dataclass
class OpenAICompatibleProvider(LLMProviderBase):
    """Shared caller for endpoints speaking the OpenAI chat-completions protocol.

    Every supported provider is reached through the OpenAI SDK, so the session
    handling, the call itself, result extraction, error handling and the
    :class:`CallerStats` bookkeeping live here once. Subclasses implement only
    :meth:`build_request`, which is where the provider-specific knowledge sits:
    which column of ``llm-params.csv`` feeds which SDK argument, and whether it
    travels as a standard keyword or inside ``extra_body``.

    Abstract on purpose. Without a ``build_request`` there is no such thing as a
    generic provider, so a provider with no registered class fails at dispatch
    instead of silently running with ``temperature`` / ``top_p`` alone.
    """

    # Redeclared by every subclass to change nothing but its default. Dataclass
    # fields merge by name in reverse-MRO order and keep their original slot, so
    # the generated ``__init__`` signature is identical for all of them.
    base_url: Optional[str] = None

    def validate_config(self) -> None:
        """Check the configuration this provider cannot run without.

        Called at dispatch time, outside the caller's own exception handling, so
        a misconfiguration aborts the run instead of being recorded as a per-row
        error. Providers reachable on their default URL need nothing.
        """

    def build_session(self) -> None:
        from openai import OpenAI  # lazy import — openai is an optional dependency

        self.validate_config()
        kwargs: Dict[str, Any] = {"base_url": self.base_url, "api_key": self.auth.api_key}
        if self.timeout is not None:
            # Passed only when set, so providers that configure no timeout keep
            # getting the SDK's own default.
            kwargs["timeout"] = self.timeout
        self.session = OpenAI(**kwargs)

    def _base_params(self, row: dict) -> Dict[str, Any]:
        """The two parameters every provider takes, minus the ones left unset."""

        params: Dict[str, Any] = {
            "temperature": row.get("temperature", 0.7),
            "top_p": row.get("top_p", 1.0),
        }
        return {k: v for k, v in params.items() if _is_set(v)}

    def _make_request(
        self,
        prompt: str,
        row: dict,
        params: Dict[str, Any],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> ProviderRequest:
        """Store and return the request, tunnelling ``extra_body`` when non-empty."""

        if extra_body:
            params["extra_body"] = extra_body
        self.request = ProviderRequest(
            model=str(row.get("model_name", "")),
            prompt=prompt,
            params=params,
        )
        return self.request

    def execute(self, prompt: str, row: dict) -> ProviderResponse:
        self.data = row
        self.state = CallerState.RUNNING
        t0 = time.monotonic()
        try:
            if self.session is None:
                self.build_session()
            req = self.build_request(prompt, row)
            extra_body = req.params.pop("extra_body", None)
            completion = self.session.chat.completions.create(
                model=req.model,
                messages=[{"role": "user", "content": req.prompt}],
                extra_body=extra_body or None,
                **req.params,
            )
            choice = completion.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                # Generation stopped at max_tokens: what came back is a fragment,
                # not an answer, so it is flagged rather than stored as a result.
                text = MAX_TOKENS_TEXT
                self.state = CallerState.MAXTOKENREACHED
            else:
                text = choice.message.content or ""
                self.state = CallerState.FINISHED
            tokens = getattr(completion.usage, "total_tokens", None)
            self.response = ProviderResponse(text=text, usage_tokens=tokens, raw=completion)  # gitleaks:allow
        except Exception as exc:
            logger.exception(exc)
            self.response = ProviderResponse(text="", usage_tokens=None, raw=str(exc))
            self.state = CallerState.ERROR
        finally:
            elapsed = time.monotonic() - t0
            self.stats.call_count += 1
            self.stats.elapsed_seconds += elapsed
            if self.response and self.response.usage_tokens:
                self.stats.total_tokens += self.response.usage_tokens
        return self.response


@dataclass
class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI's own API, which caps generation with ``max_completion_tokens``."""

    base_url: Optional[str] = URL_MAP["openai"]  # None -> the SDK's default host

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        params = self._base_params(row)
        if _is_set(row.get("max_tokens")):
            params["max_completion_tokens"] = row["max_tokens"]
        if _is_set(row.get("openai_seed")):
            params["seed"] = row["openai_seed"]
        return self._make_request(prompt, row, params)


@dataclass
class OllamaProvider(OpenAICompatibleProvider):
    """Concrete LLM provider for Ollama using the OpenAI-compatible API."""

    base_url: Optional[str] = URL_MAP["ollama"]

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        # ollama takes no standard ``max_tokens``: its cap is ``num_predict``,
        # and all three of its knobs are options passed through ``extra_body``.
        extra_body: Dict[str, Any] = {
            k: v
            for k, v in {
                "num_ctx": row.get("ollama_context_window"),
                "num_predict": row.get("max_tokens"),
                "repeat_penalty": row.get("ollama_repeat_penalty"),
            }.items()
            if _is_set(v)
        }
        return self._make_request(prompt, row, self._base_params(row), extra_body)


@dataclass
class VLLMProvider(OpenAICompatibleProvider):
    """A vLLM server reached over its OpenAI-compatible endpoint."""

    base_url: Optional[str] = URL_MAP["vllm"]

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        params = self._base_params(row)
        if _is_set(row.get("max_tokens")):
            params["max_tokens"] = row["max_tokens"]
        extra_body: Dict[str, Any] = {
            k: v
            for k, v in {
                "min_p": row.get("vllm_min_p"),
                "best_of": row.get("vllm_best_of"),
            }.items()
            if _is_set(v)
        }
        return self._make_request(prompt, row, params, extra_body)


@dataclass
class GeminiProvider(OpenAICompatibleProvider):
    """Gemini reached over its OpenAI-compatible endpoint."""

    base_url: Optional[str] = URL_MAP["gemini"]

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        params = self._base_params(row)
        if _is_set(row.get("max_tokens")):
            params["max_tokens"] = row["max_tokens"]
        extra_body: Dict[str, Any] = {}
        if _is_set(row.get("gemini_thinking_level")):
            extra_body["thinking_level"] = row["gemini_thinking_level"]
        return self._make_request(prompt, row, params, extra_body)


@dataclass
class LiteLLMProvider(OpenAICompatibleProvider):
    """Concrete LLM provider for models served behind a LiteLLM proxy.

    The proxy exposes a single OpenAI-compatible endpoint in front of a vLLM
    backend, so the hyperparameters mirror the ``vllm`` ones (``min_p`` /
    ``best_of`` passed through ``extra_body``). Unlike the local providers, the
    proxy always requires an API token, and there is no default URL — both are
    validated up front by :meth:`validate_config`.
    """

    base_url: Optional[str] = URL_MAP["litellm"]

    def validate_config(self) -> None:
        """Ensure the proxy URL and API token are configured.

        Raises:
            ProviderConfigException: if the base URL or the API token is missing.
        """

        if not self.base_url:
            raise ProviderConfigException(
                f"Provider '{self.provider}' requires a base URL. "
                f"Set PROVIDER_{self.provider.upper()}_URL in your .env."
            )
        if not self.auth.api_key or self.auth.api_key == "na":
            raise ProviderConfigException(
                f"Provider '{self.provider}' requires an API token. "
                f"Set PROVIDER_{self.provider.upper()}_KEY in your .env."
            )

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        params = self._base_params(row)
        if _is_set(row.get("max_tokens")):
            params["max_tokens"] = row["max_tokens"]
        extra_body: Dict[str, Any] = {
            k: v
            for k, v in {
                "min_p": row.get("litellm_min_p"),
                "best_of": row.get("litellm_best_of"),
            }.items()
            if _is_set(v)
        }
        return self._make_request(prompt, row, params, extra_body)
