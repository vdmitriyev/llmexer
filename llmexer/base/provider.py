"""Abstract base class and supporting types for LLM provider callers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CallerState(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ProviderAuth:
    api_key: str = "na"
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
