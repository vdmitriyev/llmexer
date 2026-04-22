"""LLM request execution logic for llmexer experiments."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

URL_MAP: Dict[str, Optional[str]] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "openai": None,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


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

    def model_dump(self) -> dict:
        return asdict(self)


class LLMRequestsMapper:
    """Translates generic CSV row parameters to provider-specific OpenAI SDK arguments."""

    def __init__(
        self, provider: str, base_url: Optional[str] = None, api_key: str = "na"
    ):
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
        model = str(row.get("param_model_name", row.get("model_name", "")))
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
            )
        except Exception as e:
            return LLMRunResult(
                model=model,
                provider=self.provider,
                prompt=prompt,
                profile=profile,
                parameters=row,
                response_text="",
                status=f"Error: {e}",
            )
