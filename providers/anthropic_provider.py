"""Anthropic Messages API adapter."""

from __future__ import annotations

import os
from typing import Any, Mapping

from providers.base import BaseProvider, ProviderRequest, optional_float, optional_int


class AnthropicProvider(BaseProvider):
    """Adapter for Anthropic's Messages API."""

    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url_env = "ANTHROPIC_BASE_URL"
    default_base_url = "https://api.anthropic.com"

    def build_request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: Mapping[str, Any],
    ) -> ProviderRequest:
        """Build a Messages API request."""

        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": optional_int(options, "max_output_tokens") or 200,
        }
        temperature = optional_float(options, "temperature")
        if temperature is not None:
            payload["temperature"] = temperature

        return ProviderRequest(
            provider=self.name,
            model=model,
            url=self.resolve_url(options),
            headers={
                "x-api-key": self.api_key or "DRY_RUN_API_KEY",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload=payload,
        )

    def extract_text(self, raw_json: dict[str, Any] | list[Any] | None) -> str:
        """Extract text blocks from an Anthropic message response."""

        if not isinstance(raw_json, dict):
            return ""
        parts: list[str] = []
        for block in raw_json.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    def extract_usage(self, raw_json: dict[str, Any] | list[Any] | None) -> tuple[int | None, int | None]:
        """Extract token usage from an Anthropic message response."""

        if not isinstance(raw_json, dict):
            return None, None
        usage = raw_json.get("usage") or {}
        if not isinstance(usage, dict):
            return None, None
        return usage.get("input_tokens"), usage.get("output_tokens")

    def resolve_url(self, options: Mapping[str, Any]) -> str:
        """Resolve Anthropic-compatible endpoint from options/env/defaults.

        `ANTHROPIC_BASE_URL` may be either a base host, a `/v1` base path, or
        the full `/v1/messages` endpoint. This makes gateway providers usable
        without changing experiment code.
        """

        configured = str(
            options.get("anthropic_base_url") or os.getenv(self.base_url_env) or self.default_base_url
        ).rstrip("/")
        if configured.endswith("/v1/messages"):
            return configured
        if configured.endswith("/v1"):
            return f"{configured}/messages"
        return f"{configured}/v1/messages"
