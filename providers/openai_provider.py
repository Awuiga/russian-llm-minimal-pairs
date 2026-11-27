"""OpenAI Responses API adapter."""

from __future__ import annotations

from typing import Any, Mapping

from providers.base import BaseProvider, ProviderRequest, optional_float, optional_int


OPENAI_TEMPERATURE_UNSUPPORTED_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


OPENAI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "grammar_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "has_error": {"type": "boolean"},
            "error_type": {"type": "string"},
            "correction": {"type": "string"},
        },
        "required": ["has_error", "error_type", "correction"],
    },
}


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI's Responses API."""

    name = "openai"
    api_key_env = "OPENAI_API_KEY"
    url = "https://api.openai.com/v1/responses"

    def build_request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: Mapping[str, Any],
    ) -> ProviderRequest:
        """Build a Responses API request with structured JSON output."""

        payload: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                }
            ],
            "text": {"format": OPENAI_RESPONSE_SCHEMA},
        }
        temperature = optional_float(options, "temperature")
        max_output_tokens = optional_int(options, "max_output_tokens")
        reasoning = options.get("reasoning")
        if temperature is not None and supports_temperature(model):
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if reasoning:
            payload["reasoning"] = {"effort": reasoning}

        return ProviderRequest(
            provider=self.name,
            model=model,
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key or 'DRY_RUN_API_KEY'}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )

    def extract_text(self, raw_json: dict[str, Any] | list[Any] | None) -> str:
        """Extract aggregated text from an OpenAI Responses object."""

        if not isinstance(raw_json, dict):
            return ""
        output_text = raw_json.get("output_text")
        if isinstance(output_text, str):
            return output_text
        parts: list[str] = []
        for item in raw_json.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)

    def extract_usage(self, raw_json: dict[str, Any] | list[Any] | None) -> tuple[int | None, int | None]:
        """Extract token usage from an OpenAI Responses object."""

        if not isinstance(raw_json, dict):
            return None, None
        usage = raw_json.get("usage") or {}
        if not isinstance(usage, dict):
            return None, None
        return usage.get("input_tokens"), usage.get("output_tokens")


def supports_temperature(model: str) -> bool:
    """Return whether an OpenAI model should receive a temperature parameter."""

    normalized = model.strip().casefold()
    return not normalized.startswith(OPENAI_TEMPERATURE_UNSUPPORTED_PREFIXES)
