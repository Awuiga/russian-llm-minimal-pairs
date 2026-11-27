"""Base classes and HTTP utilities for LLM provider adapters.

Provider-specific modules are responsible for building payloads and extracting
text/usage from responses. This module handles the shared HTTP mechanics,
including retries, rate-limit handling, latency measurement, and key redaction.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ProviderRequest:
    """A fully rendered provider request."""

    provider: str
    model: str
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a serializable request with sensitive headers redacted."""

        return {
            "provider": self.provider,
            "model": self.model,
            "url": self.url,
            "headers": redact_headers(self.headers),
            "payload": self.payload,
        }


@dataclass(frozen=True)
class ProviderResponse:
    """A normalized transport-level provider response."""

    request_status: str
    status_code: int | None
    raw_json: dict[str, Any] | list[Any] | None
    raw_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    error_text: str = ""
    retry_count: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when provider configuration is missing or invalid."""


class BaseProvider:
    """Base adapter for one LLM provider."""

    name = "base"
    api_key_env = ""

    def __init__(self, *, allow_missing_api_key: bool = False) -> None:
        self.api_key = os.getenv(self.api_key_env, "")
        if not self.api_key and not allow_missing_api_key:
            raise ProviderError(f"Missing required environment variable: {self.api_key_env}")

    def build_request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: Mapping[str, Any],
    ) -> ProviderRequest:
        """Build a provider request.

        Subclasses must implement this method.
        """

        raise NotImplementedError

    def extract_text(self, raw_json: dict[str, Any] | list[Any] | None) -> str:
        """Extract assistant text from a provider response."""

        raise NotImplementedError

    def extract_usage(self, raw_json: dict[str, Any] | list[Any] | None) -> tuple[int | None, int | None]:
        """Extract input and output token counts if the provider returned them."""

        return None, None

    def execute(
        self,
        request: ProviderRequest,
        *,
        max_retries: int = 3,
        timeout: float = 60.0,
        backoff_base: float = 1.0,
    ) -> ProviderResponse:
        """Send a JSON request with retries and exponential backoff."""

        attempts = max(1, max_retries + 1)
        last_error = ""
        for attempt_index in range(attempts):
            started = time.perf_counter()
            try:
                http_request = urllib.request.Request(
                    request.url,
                    data=json.dumps(request.payload, ensure_ascii=False).encode("utf-8"),
                    headers=request.headers,
                    method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    raw_json = parse_json_body(body)
                    input_tokens, output_tokens = self.extract_usage(raw_json)
                    return ProviderResponse(
                        request_status="succeeded",
                        status_code=response.status,
                        raw_json=raw_json,
                        raw_text=self.extract_text(raw_json),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        retry_count=attempt_index,
                        response_headers=dict(response.headers.items()),
                    )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = body or str(exc)
                if exc.code in RETRYABLE_STATUS_CODES and attempt_index < attempts - 1:
                    sleep_for_retry(exc.headers, attempt_index, backoff_base)
                    continue
                return ProviderResponse(
                    request_status="failed",
                    status_code=exc.code,
                    raw_json=parse_json_body(body),
                    raw_text="",
                    latency_ms=latency_ms,
                    error_text=last_error,
                    retry_count=attempt_index,
                    response_headers=dict(exc.headers.items()),
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(exc)
                if attempt_index < attempts - 1:
                    sleep_for_retry({}, attempt_index, backoff_base)
                    continue
                return ProviderResponse(
                    request_status="failed",
                    status_code=None,
                    raw_json=None,
                    raw_text="",
                    latency_ms=latency_ms,
                    error_text=last_error,
                    retry_count=attempt_index,
                )

        return ProviderResponse(
            request_status="failed",
            status_code=None,
            raw_json=None,
            raw_text="",
            error_text=last_error or "request failed",
            retry_count=attempts - 1,
        )


def parse_json_body(body: str) -> dict[str, Any] | list[Any] | None:
    """Parse a JSON response body, preserving non-JSON bodies as raw text elsewhere."""

    if not body.strip():
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"_non_json_body": body}
    if isinstance(parsed, (dict, list)):
        return parsed
    return {"value": parsed}


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Redact secrets before persisting request metadata."""

    redacted: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.casefold()
        if lowered in {"authorization", "x-api-key"}:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def sleep_for_retry(headers: Mapping[str, str], attempt_index: int, backoff_base: float) -> None:
    """Sleep according to Retry-After or exponential backoff."""

    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    if retry_after:
        try:
            time.sleep(min(float(retry_after), 60.0))
            return
        except ValueError:
            pass
    time.sleep(min(backoff_base * (2**attempt_index), 60.0))


def optional_float(options: Mapping[str, Any], key: str) -> float | None:
    """Read an optional float from a mapping."""

    value = options.get(key)
    if value in (None, ""):
        return None
    return float(value)


def optional_int(options: Mapping[str, Any], key: str) -> int | None:
    """Read an optional int from a mapping."""

    value = options.get(key)
    if value in (None, ""):
        return None
    return int(value)
