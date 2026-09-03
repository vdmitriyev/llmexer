"""LLM provider abstractions, configuration, and concrete provider callers."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from llmexer.base.experiment import FILE_LLM_PARAMS, FILE_LLMS_FOR_EXPERIMENT
from llmexer.base.llm_core import LLMRunResult
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


class LLMRequestsMapper:
    """Translates generic CSV row parameters to provider-specific OpenAI SDK arguments."""

    def __init__(self, provider: str, base_url: Optional[str] = None, api_key: str = "na"):
        from openai import OpenAI  # lazy import — openai is an optional dependency

        self.provider = provider.lower()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _map_params(self, row: dict) -> tuple:
        """Returns (standard_payload, extra_body) for the provider."""
        payload: Dict[str, Any] = {
            "temperature": row.get("temperature", 0.7),
            "top_p": row.get("top_p", 1.0),
        }
        extra_body: Dict[str, Any] = {}

        if self.provider == "openai":
            if row.get("max_tokens") is not None:
                payload["max_completion_tokens"] = row["max_tokens"]
            if row.get("openai_seed") is not None:
                payload["seed"] = row["openai_seed"]

        elif self.provider == "ollama":
            extra_body = {
                k: v
                for k, v in {
                    "num_ctx": row.get("ollama_context_window"),
                    "num_predict": row.get("max_tokens"),
                    "repeat_penalty": row.get("ollama_repeat_penalty"),
                }.items()
                if v is not None
            }

        elif self.provider == "vllm":
            if row.get("max_tokens") is not None:
                payload["max_tokens"] = row["max_tokens"]
            extra_body = {
                k: v
                for k, v in {
                    "min_p": row.get("vllm_min_p"),
                    "best_of": row.get("vllm_best_of"),
                }.items()
                if v is not None
            }

        elif self.provider == "litellm":
            if row.get("max_tokens") is not None:
                payload["max_tokens"] = row["max_tokens"]
            extra_body = {
                k: v
                for k, v in {
                    "min_p": row.get("litellm_min_p"),
                    "best_of": row.get("litellm_best_of"),
                }.items()
                if v is not None
            }

        elif self.provider == "gemini":
            if row.get("max_tokens") is not None:
                payload["max_tokens"] = row["max_tokens"]
            if row.get("gemini_thinking_level") is not None:
                extra_body["thinking_level"] = row["gemini_thinking_level"]

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        return payload, extra_body

    def execute(self, prompt: str, row: dict) -> LLMRunResult:
        """Execute a single LLM call and return a standardised result."""
        model = str(row.get("model_name", ""))
        profile = str(row.get("profile_name", ""))
        std_payload, extra = self._map_params(row)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra if extra else None,
                **std_payload,
            )
            return LLMRunResult(
                model=model,
                provider=self.provider,
                prompt=prompt,
                profile=profile,
                parameters=row,
                response_text=response.choices[0].message.content or "",
                usage_tokens=response.usage.total_tokens if response.usage else None,
                raw=serialize_response(response),
            )
        except Exception as e:
            logger.exception(e)
            return LLMRunResult(
                model=model,
                provider=self.provider,
                prompt=prompt,
                profile=profile,
                parameters=row,
                response_text="",
                status=f"Error: {e}",
            )


@dataclass
class OllamaProvider(LLMProviderBase):
    """Concrete LLM provider for Ollama using the OpenAI-compatible API."""

    base_url: str = URL_MAP["ollama"]

    def build_session(self) -> None:
        from openai import OpenAI

        self.session = OpenAI(base_url=self.base_url, api_key=self.auth.api_key)

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        extra_body: Dict[str, Any] = {
            k: v
            for k, v in {
                "num_ctx": row.get("ollama_context_window"),
                "num_predict": row.get("max_tokens"),
                "repeat_penalty": row.get("ollama_repeat_penalty"),
            }.items()
            if v is not None
        }
        params: Dict[str, Any] = {
            "temperature": row.get("temperature", 0.7),
            "top_p": row.get("top_p", 1.0),
        }
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
            text = completion.choices[0].message.content or ""
            tokens = getattr(completion.usage, "total_tokens", None)
            self.response = ProviderResponse(text=text, usage_tokens=tokens, raw=completion)  # gitleaks:allow
            self.state = CallerState.FINISHED
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
class LiteLLMProvider(LLMProviderBase):
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

    def build_session(self) -> None:
        from openai import OpenAI

        self.validate_config()
        self.session = OpenAI(base_url=self.base_url, api_key=self.auth.api_key)

    def build_request(self, prompt: str, row: dict) -> ProviderRequest:
        extra_body: Dict[str, Any] = {
            k: v
            for k, v in {
                "min_p": row.get("litellm_min_p"),
                "best_of": row.get("litellm_best_of"),
            }.items()
            if v is not None
        }
        params: Dict[str, Any] = {
            "temperature": row.get("temperature", 0.7),
            "top_p": row.get("top_p", 1.0),
        }
        if row.get("max_tokens") is not None:
            params["max_tokens"] = row["max_tokens"]
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
            text = ""
            if completion.choices[0].finish_reason == "length":
                self.state = CallerState.MAXTOKENREACHED
                text = "No answer. Max token reached"

            if self.state != CallerState.MAXTOKENREACHED:
                text = completion.choices[0].message.content or ""
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
