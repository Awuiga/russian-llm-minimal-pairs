"""Parse and normalize model responses for the API experiment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


CORRECTION_RE = re.compile(
    r"(?:correction|исправлен(?:ие|ное предложение)?|исправленный вариант)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedResponse:
    """Normalized model response fields."""

    has_error_pred: bool | None
    error_type_pred: str
    correction_pred: str
    parse_status: str
    raw_json: dict[str, Any] | None


def parse_model_response(raw_text: str) -> ParsedResponse:
    """Parse a model response into the common result schema.

    The parser first tries strict JSON, then JSON extraction from surrounding
    text or Markdown fences. If that fails, it keeps the raw text and attempts a
    small correction heuristic.
    """

    text = raw_text.strip()
    parsed_json = try_parse_json_object(text)
    if parsed_json is None:
        extracted = extract_json_object(text)
        parsed_json = try_parse_json_object(extracted) if extracted else None

    if parsed_json is not None:
        has_error = normalize_bool(parsed_json.get("has_error"))
        correction = stringify(parsed_json.get("correction"))
        error_type = stringify(parsed_json.get("error_type") or "none")
        status = "ok" if has_error is not None and correction else "partial"
        return ParsedResponse(
            has_error_pred=has_error,
            error_type_pred=error_type,
            correction_pred=correction,
            parse_status=status,
            raw_json=parsed_json,
        )

    heuristic_correction = extract_correction_heuristic(text)
    return ParsedResponse(
        has_error_pred=None,
        error_type_pred="",
        correction_pred=heuristic_correction,
        parse_status="partial" if heuristic_correction else "failed",
        raw_json=None,
    )


def try_parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse text as a JSON object."""

    if not text:
        return None
    cleaned = strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def strip_json_fence(text: str) -> str:
    """Remove a Markdown code fence around JSON if present."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from arbitrary text."""

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def normalize_bool(value: Any) -> bool | None:
    """Normalize boolean-like values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().casefold()
    if lowered in {"true", "yes", "y", "1", "да", "есть", "ошибка"}:
        return True
    if lowered in {"false", "no", "n", "0", "нет", "none", "без ошибки"}:
        return False
    return None


def stringify(value: Any) -> str:
    """Return a compact string for scalar JSON values."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def extract_correction_heuristic(text: str) -> str:
    """Try to recover a correction from a non-JSON response."""

    for line in text.splitlines():
        match = CORRECTION_RE.search(line.strip())
        if match:
            return cleanup_sentence(match.group(1))
    quoted = re.findall(r"[\"«](.+?[.!?])[\"»]", text)
    if quoted:
        return cleanup_sentence(quoted[-1])
    return ""


def cleanup_sentence(text: str) -> str:
    """Remove obvious markup and trailing commentary from a heuristic correction."""

    cleaned = text.strip().strip("`*_ ")
    if "}" in cleaned and "{" not in cleaned:
        cleaned = cleaned.split("}", 1)[0]
    return cleaned.strip()
