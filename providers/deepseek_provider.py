"""DeepSeek OpenAI-compatible chat/completions adapter."""

from __future__ import annotations

from typing import Any, Mapping

from providers.base import BaseProvider, ProviderRequest, optional_float, optional_int


class DeepSeekProvider(BaseProvider):
    """Adapter for DeepSeek's OpenAI-compatible chat/completions API."""

    name = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"

    def build_request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: Mapping[str, Any],
    ) -> ProviderRequest:
        """Build a non-streaming chat completion request."""

        base_url = str(options.get("deepseek_base_url") or self.base_url).rstrip("/")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": optional_int(options, "max_output_tokens") or 200,
        }
        temperature = optional_float(options, "temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        thinking = options.get("thinking")
        if thinking:
            payload["thinking"] = {"type": thinking}
            reasoning = options.get("reasoning")
            if reasoning:
                payload["thinking"]["reasoning_effort"] = reasoning

        return ProviderRequest(
            provider=self.name,
            model=model,
            url=f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key or 'DRY_RUN_API_KEY'}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )

    def extract_text(self, raw_json: dict[str, Any] | list[Any] | None) -> str:
        """Extract assistant message text from a chat completion."""

        if not isinstance(raw_json, dict):
            return ""
        choices = raw_json.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def extract_usage(self, raw_json: dict[str, Any] | list[Any] | None) -> tuple[int | None, int | None]:
        """Extract prompt/completion token usage from a chat completion."""

        if not isinstance(raw_json, dict):
            return None, None
        usage = raw_json.get("usage") or {}
        if not isinstance(usage, dict):
            return None, None
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
