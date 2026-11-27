"""Orchestrate reproducible API experiments over final corpus items.

The runner is deliberately provider-agnostic: prompts, request rendering,
transport, parsing, and persistence live in separate modules. This keeps the
experiment easy to audit and makes interrupted runs resumable from the
normalized result file.
"""

from __future__ import annotations

import json
import random
import time
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from parser import ParsedResponse, parse_model_response
from prompts import list_prompt_ids, render_prompt
from providers.anthropic_provider import AnthropicProvider
from providers.base import BaseProvider, ProviderResponse
from providers.deepseek_provider import DeepSeekProvider
from providers.openai_provider import OpenAIProvider
from storage import RunStorage, load_items


PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
}


@dataclass(frozen=True)
class ProviderSpec:
    """One provider/model/prompt slice of an experiment matrix."""

    name: str
    models: list[str]
    prompt_ids: list[str]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApiRunOptions:
    """Configuration for one API experiment run."""

    input_path: Path
    output_dir: Path
    provider_specs: list[ProviderSpec]
    run_id: str
    limit: int | None = None
    seed: int | None = None
    temperature: float = 0.0
    max_output_tokens: int = 200
    reasoning: str = ""
    thinking: str = ""
    dry_run: bool = False
    resume: bool = False
    max_retries: int = 3
    timeout: float = 60.0
    backoff_base: float = 1.0
    max_concurrency: int = 1
    sleep_between_requests: float = 0.0


@dataclass(frozen=True)
class ApiRunResult:
    """Compact result returned by the runner."""

    run_id: str
    output_dir: Path
    paths: list[Path]
    stats: dict[str, Any]


def default_run_id(prefix: str = "run") -> str:
    """Create a timestamped run id."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def create_provider(provider_name: str, *, allow_missing_api_key: bool = False) -> BaseProvider:
    """Instantiate a provider adapter by name."""

    try:
        provider_class = PROVIDER_CLASSES[provider_name]
    except KeyError as exc:
        known = ", ".join(sorted(PROVIDER_CLASSES))
        raise ValueError(f"Unknown provider {provider_name!r}. Known providers: {known}") from exc
    return provider_class(allow_missing_api_key=allow_missing_api_key)


def load_matrix_config(
    config_path: str | Path,
    *,
    output_dir_override: str | Path | None = None,
    resume: bool = False,
) -> ApiRunOptions:
    """Load a matrix JSON config into runner options."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    configured_run_id = str(config.get("run_id") or "")
    provisional_run_id = configured_run_id or default_run_id("matrix")
    output_dir = Path(output_dir_override or config.get("output_dir") or Path("./runs") / provisional_run_id)
    run_id = configured_run_id or (find_existing_run_id(output_dir) if resume else "") or provisional_run_id
    provider_specs = [
        ProviderSpec(
            name=str(provider["name"]),
            models=[str(model) for model in provider.get("models", [])],
            prompt_ids=[str(prompt) for prompt in provider.get("prompts", provider.get("prompt_ids", []))],
            options={key: value for key, value in provider.items() if key not in {"name", "models", "prompts", "prompt_ids"}},
        )
        for provider in config.get("providers", [])
    ]
    if not provider_specs:
        raise ValueError("Matrix config must contain at least one provider spec")

    return ApiRunOptions(
        input_path=Path(config["input_path"]),
        output_dir=output_dir,
        provider_specs=normalize_provider_specs(provider_specs),
        run_id=run_id,
        limit=optional_int(config.get("limit")),
        seed=optional_int(config.get("seed")),
        temperature=float(config.get("temperature", 0.0)),
        max_output_tokens=int(config.get("max_output_tokens", 200)),
        reasoning=str(config.get("reasoning", "")),
        thinking=str(config.get("thinking", "")),
        dry_run=bool(config.get("dry_run", False)),
        resume=resume or bool(config.get("resume", False)),
        max_retries=int(config.get("max_retries", 3)),
        timeout=float(config.get("timeout", 60.0)),
        backoff_base=float(config.get("backoff_base", 1.0)),
        max_concurrency=int(config.get("max_concurrency", 1)),
        sleep_between_requests=float(config.get("sleep_between_requests", 0.0)),
    )


def run_api_experiment(options: ApiRunOptions) -> ApiRunResult:
    """Run one experiment and persist all raw and normalized layers."""

    items = select_items(load_items(options.input_path), limit=options.limit, seed=options.seed)
    storage = RunStorage(options.output_dir, run_id=options.run_id)
    existing_manifest = load_existing_manifest(storage.manifest_path) if options.resume else None
    completed = storage.completed_keys() if options.resume else set()
    stats: dict[str, Any] = {
        "planned": count_planned(options.provider_specs, items),
        "processed": 0,
        "skipped_resume": 0,
        "succeeded": 0,
        "failed": 0,
        "dry_run": 0,
    }
    manifest = build_manifest(options, items, stats["planned"])
    if existing_manifest is None:
        storage.write_manifest(manifest)
    storage.append_log(f"{now_iso()} start run_id={options.run_id} dry_run={options.dry_run}")
    if options.max_concurrency > 1:
        storage.append_log(
            f"{now_iso()} max_concurrency={options.max_concurrency} requested; current runner executes sequentially."
        )

    providers: dict[str, BaseProvider] = {}
    for spec in normalize_provider_specs(options.provider_specs):
        provider = providers.get(spec.name)
        if provider is None:
            provider = create_provider(spec.name, allow_missing_api_key=options.dry_run)
            providers[spec.name] = provider

        for model in spec.models:
            for prompt_id in spec.prompt_ids:
                for item in items:
                    item_id = get_item_id(item)
                    key = (spec.name, model, prompt_id, item_id)
                    if key in completed:
                        stats["skipped_resume"] += 1
                        continue
                    row = process_one_item(
                        storage=storage,
                        provider=provider,
                        spec=spec,
                        model=model,
                        prompt_id=prompt_id,
                        item=item,
                        options=options,
                    )
                    stats["processed"] += 1
                    request_status = row["request_status"]
                    if request_status == "succeeded":
                        stats["succeeded"] += 1
                    elif request_status == "dry_run":
                        stats["dry_run"] += 1
                    else:
                        stats["failed"] += 1
                    if options.sleep_between_requests and not options.dry_run:
                        time.sleep(options.sleep_between_requests)

    storage.append_log(f"{now_iso()} finish stats={json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
    if existing_manifest is not None:
        storage.write_manifest(append_resume_event(existing_manifest, options.run_id, stats))
    else:
        storage.write_manifest(build_manifest(options, items, stats["planned"], final_stats=stats))
    return ApiRunResult(
        run_id=options.run_id,
        output_dir=storage.output_dir,
        paths=[
            storage.manifest_path,
            storage.log_path,
            storage.raw_requests_path,
            storage.raw_responses_path,
            storage.normalized_csv_path,
            storage.normalized_jsonl_path,
        ],
        stats=stats,
    )


def process_one_item(
    *,
    storage: RunStorage,
    provider: BaseProvider,
    spec: ProviderSpec,
    model: str,
    prompt_id: str,
    item: dict[str, Any],
    options: ApiRunOptions,
) -> dict[str, Any]:
    """Render, execute, parse, and persist one provider/model/prompt/item task."""

    item_id = get_item_id(item)
    timestamp = now_iso()
    rendered = render_prompt(prompt_id, item)
    request_options = merged_options(options, spec)
    attempts = model_attempts(spec, model)
    response = ProviderResponse("failed", None, None, "", error_text="no attempt executed")
    parsed = ParsedResponse(None, "", "", "failed", None)
    actual_model = model
    fallback_used = False

    for attempt_index, attempt_model in enumerate(attempts):
        actual_model = attempt_model
        fallback_used = attempt_model != model
        request = provider.build_request(
            model=attempt_model,
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
            options=request_options,
        )
        request_record = {
            "run_id": options.run_id,
            "timestamp": timestamp,
            "provider": spec.name,
            "requested_model": model,
            "model": attempt_model,
            "fallback_used": fallback_used,
            "fallback_attempt": attempt_index,
            "prompt_id": prompt_id,
            "item_id": item_id,
            "pair_id": item.get("pair_id", ""),
            "variant": item.get("variant", ""),
            "system_prompt": rendered.system_prompt,
            "user_prompt": rendered.user_prompt,
            "request": request.to_safe_dict(),
        }
        storage.append_raw_request(request_record)

        if options.dry_run:
            response = ProviderResponse(
                request_status="dry_run",
                status_code=None,
                raw_json=None,
                raw_text="",
                latency_ms=0,
            )
            parsed = ParsedResponse(None, "", "", "skipped", None)
        else:
            response = provider.execute(
                request,
                max_retries=options.max_retries,
                timeout=options.timeout,
                backoff_base=options.backoff_base,
            )
            parsed = (
                parse_model_response(response.raw_text)
                if response.request_status == "succeeded"
                else ParsedResponse(None, "", "", "failed", None)
            )

        response_record = {
            "run_id": options.run_id,
            "timestamp": now_iso(),
            "provider": spec.name,
            "requested_model": model,
            "model": attempt_model,
            "fallback_used": fallback_used,
            "fallback_attempt": attempt_index,
            "prompt_id": prompt_id,
            "item_id": item_id,
            "pair_id": item.get("pair_id", ""),
            "variant": item.get("variant", ""),
            "request_status": response.request_status,
            "status_code": response.status_code,
            "error_text": response.error_text,
            "retry_count": response.retry_count,
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "response_headers": response.response_headers,
            "provider_response": response.raw_json,
            "raw_text": response.raw_text,
        }
        storage.append_raw_response(response_record)

        if response.request_status in {"succeeded", "dry_run"}:
            break
        if attempt_index < len(attempts) - 1 and is_anthropic_availability_error(spec, response):
            storage.append_log(
                f"{now_iso()} fallback provider={spec.name} requested_model={model} "
                f"failed_model={attempt_model} next_model={attempts[attempt_index + 1]} "
                f"status_code={response.status_code}"
            )
            continue
        break

    normalized = {
        "run_id": options.run_id,
        "provider": spec.name,
        "requested_model": model,
        "model": actual_model,
        "fallback_used": fallback_used,
        "prompt_id": prompt_id,
        "item_id": item_id,
        "pair_id": item.get("pair_id", ""),
        "variant": item.get("variant", ""),
        "sentence": item.get("sentence", ""),
        "gold_label": item.get("gold_label", ""),
        "gold_correction": item.get("gold_correction", ""),
        "expected_action": item.get("expected_action", ""),
        "rule_id": item.get("rule_id", ""),
        "rule_name": item.get("rule_name", ""),
        "difficulty": item.get("difficulty", ""),
        "has_error_pred": parsed.has_error_pred,
        "error_type_pred": parsed.error_type_pred,
        "correction_pred": parsed.correction_pred,
        "parse_status": parsed.parse_status,
        "request_status": response.request_status,
        "raw_text": response.raw_text,
        "raw_json": parsed.raw_json,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "timestamp": now_iso(),
    }
    storage.append_normalized_result(normalized)
    storage.append_log(
        f"{now_iso()} {response.request_status} provider={spec.name} requested_model={model} "
        f"model={actual_model} fallback_used={fallback_used} prompt={prompt_id} "
        f"item={item_id} parse={parsed.parse_status}"
    )
    return normalized


def build_manifest(
    options: ApiRunOptions,
    items: list[dict[str, Any]],
    planned_count: int,
    *,
    final_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for the run."""

    manifest: dict[str, Any] = {
        "schema_version": "api_run_v1",
        "run_id": options.run_id,
        "created_at": now_iso(),
        "input_path": str(options.input_path),
        "output_dir": str(options.output_dir),
        "limit": options.limit,
        "seed": options.seed,
        "temperature": options.temperature,
        "max_output_tokens": options.max_output_tokens,
        "reasoning": options.reasoning,
        "thinking": options.thinking,
        "dry_run": options.dry_run,
        "resume": options.resume,
        "max_retries": options.max_retries,
        "timeout": options.timeout,
        "backoff_base": options.backoff_base,
        "max_concurrency": options.max_concurrency,
        "sleep_between_requests": options.sleep_between_requests,
        "provider_specs": [
            {"name": spec.name, "models": spec.models, "prompts": spec.prompt_ids, "options": spec.options}
            for spec in options.provider_specs
        ],
        "selected_item_count": len(items),
        "selected_item_ids": [get_item_id(item) for item in items],
        "planned_request_count": planned_count,
    }
    if final_stats is not None:
        manifest["final_stats"] = final_stats
    return manifest


def load_existing_manifest(path: Path) -> dict[str, Any] | None:
    """Load an existing manifest if present."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else None


def append_resume_event(manifest: dict[str, Any], run_id: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Preserve original manifest stats and append a resume audit event."""

    updated = dict(manifest)
    events = list(updated.get("resume_events", []))
    events.append({"timestamp": now_iso(), "run_id": run_id, "stats": stats})
    updated["resume_events"] = events
    updated["last_resume_at"] = events[-1]["timestamp"]
    return updated


def find_existing_run_id(output_dir: Path) -> str:
    """Find a prior run id in normalized results or manifest for resume mode."""

    normalized_csv = output_dir / "normalized_results.csv"
    if normalized_csv.exists():
        with normalized_csv.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("run_id"):
                    return str(row["run_id"])
    manifest = load_existing_manifest(output_dir / "run_manifest.json")
    if manifest and manifest.get("run_id"):
        return str(manifest["run_id"])
    return ""


def select_items(items: list[dict[str, Any]], *, limit: int | None, seed: int | None) -> list[dict[str, Any]]:
    """Select an optional deterministic subset of final items."""

    selected = list(items)
    if limit is not None and seed is not None:
        random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def normalize_provider_specs(specs: Iterable[ProviderSpec]) -> list[ProviderSpec]:
    """Fill defaults and validate provider specs."""

    normalized: list[ProviderSpec] = []
    for spec in specs:
        if spec.name not in PROVIDER_CLASSES:
            known = ", ".join(sorted(PROVIDER_CLASSES))
            raise ValueError(f"Unknown provider {spec.name!r}. Known providers: {known}")
        if not spec.models:
            raise ValueError(f"Provider {spec.name!r} must define at least one model")
        prompt_ids = spec.prompt_ids or ["direct_zero_shot"]
        unknown_prompts = sorted(set(prompt_ids) - set(list_prompt_ids()))
        if unknown_prompts:
            known = ", ".join(list_prompt_ids())
            raise ValueError(f"Unknown prompt ids: {', '.join(unknown_prompts)}. Known prompts: {known}")
        normalized.append(
            ProviderSpec(
                name=spec.name,
                models=list(spec.models),
                prompt_ids=prompt_ids,
                options=dict(spec.options),
            )
        )
    return normalized


def merged_options(options: ApiRunOptions, spec: ProviderSpec) -> dict[str, Any]:
    """Merge global run options with provider-specific overrides."""

    merged: dict[str, Any] = {
        "temperature": options.temperature,
        "max_output_tokens": options.max_output_tokens,
        "reasoning": options.reasoning,
        "thinking": options.thinking,
    }
    merged.update({key: value for key, value in spec.options.items() if value not in (None, "")})
    return merged


def model_attempts(spec: ProviderSpec, model: str) -> list[str]:
    """Return the primary model plus Anthropic-only fallback model ids."""

    attempts = [model]
    if spec.name != "anthropic":
        return attempts
    fallbacks = spec.options.get("model_fallbacks", [])
    if isinstance(fallbacks, str):
        fallbacks = [part.strip() for part in fallbacks.split(",")]
    for fallback in fallbacks:
        fallback_model = str(fallback).strip()
        if fallback_model and fallback_model not in attempts:
            attempts.append(fallback_model)
    return attempts


def is_anthropic_availability_error(spec: ProviderSpec, response: ProviderResponse) -> bool:
    """Return whether an Anthropic failure should trigger a model fallback."""

    if spec.name != "anthropic" or response.request_status != "failed":
        return False
    if response.status_code in {400, 401, 403}:
        return False
    error_text = response.error_text.casefold()
    availability_markers = (
        "no available accounts",
        "gateway unavailable",
        "service unavailable",
        "temporarily unavailable",
        "overloaded",
        "provider unavailable",
        "upstream unavailable",
    )
    return response.status_code in {502, 503, 504} or any(marker in error_text for marker in availability_markers)


def count_planned(specs: Iterable[ProviderSpec], items: list[dict[str, Any]]) -> int:
    """Count all provider/model/prompt/item combinations."""

    return sum(len(spec.models) * len(spec.prompt_ids or ["direct_zero_shot"]) * len(items) for spec in specs)


def get_item_id(item: dict[str, Any]) -> str:
    """Return a stable item id, falling back to pair/variant if necessary."""

    item_id = str(item.get("item_id") or "").strip()
    if item_id:
        return item_id
    return f"{item.get('pair_id', 'pair')}_{item.get('variant', 'item')}"


def optional_int(value: Any) -> int | None:
    """Parse optional integer config values."""

    if value in (None, ""):
        return None
    return int(value)


def now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
