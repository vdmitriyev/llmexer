"""Core result types for LLM request execution."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class LLMRunResult:
    model: str
    provider: str
    prompt: str
    profile: str
    parameters: Dict[str, Any]
    response_text: str
    usage_tokens: Optional[int] = None
    status: str = "success"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def model_dump(self) -> dict:
        return asdict(self)
