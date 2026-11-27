"""Tests for the API experiment layer without real network calls."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_runner import (
    ApiRunOptions,
    ProviderSpec,
    is_anthropic_availability_error,
    load_matrix_config,
    model_attempts,
    run_api_experiment,
)
from metrics import compute_and_export_metrics
from parser import parse_model_response
from prompts import render_prompt
from providers.anthropic_provider import AnthropicProvider
from providers.base import ProviderResponse
from providers.deepseek_provider import DeepSeekProvider
from providers.openai_provider import OpenAIProvider


SAMPLE_ITEMS = [
    {
        "item_id": "pair_001_correct",
        "pair_id": "pair_001",
        "variant": "correct",
        "sentence": "Студент внимательно прочитал короткую статью.",
        "gold_label": "1",
        "gold_correction": "Студент внимательно прочитал короткую статью.",
        "expected_action": "no_change",
        "rule_id": "verb_form",
        "rule_name": "Форма глагола",
        "difficulty": "medium",
    },
    {
        "item_id": "pair_001_incorrect",
        "pair_id": "pair_001",
        "variant": "incorrect",
        "sentence": "Студент внимательно прочитали короткую статью.",
        "gold_label": "0",
        "gold_correction": "Студент внимательно прочитал короткую статью.",
        "expected_action": "correct",
        "rule_id": "verb_form",
        "rule_name": "Форма глагола",
        "difficulty": "medium",
    },
]


class ApiRunnerTests(unittest.TestCase):
    """Local API-runner behavior that should not require provider credentials."""

    def test_provider_payloads_are_rendered_without_keys(self) -> None:
        """Provider adapters should build comparable JSON-oriented payloads."""

        prompt = render_prompt("direct_zero_shot", SAMPLE_ITEMS[0])
        options = {"temperature": 0.0, "max_output_tokens": 120, "thinking": "disabled"}

        openai_request = OpenAIProvider(allow_missing_api_key=True).build_request(
            model="gpt-test",
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            options=options,
        )
        anthropic_request = AnthropicProvider(allow_missing_api_key=True).build_request(
            model="claude-test",
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            options=options,
        )
        deepseek_request = DeepSeekProvider(allow_missing_api_key=True).build_request(
            model="deepseek-test",
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            options=options,
        )

        self.assertIn("/responses", openai_request.url)
        self.assertEqual(openai_request.payload["text"]["format"]["type"], "json_schema")
        self.assertIn("system", anthropic_request.payload)
        self.assertEqual(anthropic_request.payload["max_tokens"], 120)
        self.assertIn("/chat/completions", deepseek_request.url)
        self.assertEqual(deepseek_request.payload["response_format"]["type"], "json_object")
        self.assertEqual(openai_request.to_safe_dict()["headers"]["Authorization"], "<redacted>")

    def test_openai_gpt5_payload_omits_unsupported_temperature(self) -> None:
        """GPT-5 style OpenAI models should not receive the temperature field."""

        prompt = render_prompt("direct_zero_shot", SAMPLE_ITEMS[0])
        request = OpenAIProvider(allow_missing_api_key=True).build_request(
            model="gpt-5.5",
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            options={"temperature": 0.0, "max_output_tokens": 120},
        )

        self.assertNotIn("temperature", request.payload)
        self.assertEqual(request.payload["max_output_tokens"], 120)

    def test_anthropic_base_url_env_is_supported_and_key_is_redacted(self) -> None:
        """Anthropic-compatible gateways should be configurable through env."""

        prompt = render_prompt("direct_zero_shot", SAMPLE_ITEMS[0])
        with patch.dict(
            os.environ,
            {"ANTHROPIC_BASE_URL": "https://api.oneprovider.dev", "ANTHROPIC_API_KEY": "sk-test"},
            clear=False,
        ):
            request = AnthropicProvider().build_request(
                model="claude-test",
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                options={"max_output_tokens": 120},
            )

        self.assertEqual(request.url, "https://api.oneprovider.dev/v1/messages")
        self.assertEqual(request.headers["x-api-key"], "sk-test")
        self.assertEqual(request.to_safe_dict()["headers"]["x-api-key"], "<redacted>")

    def test_anthropic_model_fallbacks_are_limited_to_availability_errors(self) -> None:
        """Anthropic fallback should not hide auth or validation failures."""

        spec = ProviderSpec(
            "anthropic",
            ["claude-haiku-4-5"],
            ["direct_zero_shot"],
            {"model_fallbacks": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]},
        )
        self.assertEqual(
            model_attempts(spec, "claude-haiku-4-5"),
            ["claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        )
        self.assertTrue(
            is_anthropic_availability_error(
                spec,
                ProviderResponse("failed", 503, None, "", error_text="No available accounts"),
            )
        )
        self.assertFalse(
            is_anthropic_availability_error(
                spec,
                ProviderResponse("failed", 401, None, "", error_text="invalid api key"),
            )
        )

    def test_dry_run_writes_raw_and_normalized_files(self) -> None:
        """Dry-run should persist reproducible request metadata without API calls."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = write_items(tmp_path / "final_items.csv")
            output_dir = tmp_path / "run"
            options = ApiRunOptions(
                input_path=input_path,
                output_dir=output_dir,
                provider_specs=[ProviderSpec("openai", ["gpt-test"], ["direct_zero_shot", "minimal_edit"])],
                run_id="dry_test",
                limit=1,
                seed=7,
                dry_run=True,
            )

            result = run_api_experiment(options)

            self.assertEqual(result.stats["dry_run"], 2)
            self.assertTrue((output_dir / "raw_requests.jsonl").exists())
            rows = read_csv(output_dir / "normalized_results.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["request_status"], "dry_run")
            self.assertEqual(rows[0]["parse_status"], "skipped")

    def test_resume_skips_completed_combinations(self) -> None:
        """Resume mode should skip already normalized provider/model/prompt/item keys."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = write_items(tmp_path / "final_items.csv", rows=SAMPLE_ITEMS[:1])
            output_dir = tmp_path / "run"
            base_options = ApiRunOptions(
                input_path=input_path,
                output_dir=output_dir,
                provider_specs=[ProviderSpec("openai", ["gpt-test"], ["direct_zero_shot"])],
                run_id="resume_test",
                dry_run=True,
            )
            run_api_experiment(base_options)
            resumed = run_api_experiment(
                ApiRunOptions(
                    input_path=input_path,
                    output_dir=output_dir,
                    provider_specs=base_options.provider_specs,
                    run_id="resume_test",
                    dry_run=True,
                    resume=True,
                )
            )

            self.assertEqual(resumed.stats["skipped_resume"], 1)
            self.assertEqual(len(read_csv(output_dir / "normalized_results.csv")), 1)

    def test_matrix_config_can_run_as_dry_run(self) -> None:
        """Matrix config should support multiple providers without credentials in dry-run."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = write_items(tmp_path / "final_items.csv", rows=SAMPLE_ITEMS[:1])
            output_dir = tmp_path / "matrix_run"
            config_path = tmp_path / "matrix.json"
            config = {
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "dry_run": True,
                "limit": 1,
                "providers": [
                    {"name": "openai", "models": ["gpt-test"], "prompts": ["direct_zero_shot"]},
                    {"name": "deepseek", "models": ["deepseek-test"], "prompts": ["minimal_edit"]},
                ],
            }
            with config_path.open("w", encoding="utf-8") as file:
                json.dump(config, file)

            result = run_api_experiment(load_matrix_config(config_path))

            self.assertEqual(result.stats["dry_run"], 2)
            self.assertEqual(len(read_csv(output_dir / "normalized_results.csv")), 2)
            resumed = run_api_experiment(load_matrix_config(config_path, resume=True))
            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(resumed.stats["skipped_resume"], 2)
            self.assertEqual(manifest["run_id"], result.run_id)
            self.assertEqual(len(manifest["resume_events"]), 1)

    def test_parser_extracts_json_from_markdown_fence(self) -> None:
        """Parser should recover the required object from fenced JSON."""

        parsed = parse_model_response(
            '```json\n{"has_error": true, "error_type": "agreement", '
            '"correction": "Студент внимательно прочитал короткую статью."}\n```'
        )

        self.assertEqual(parsed.parse_status, "ok")
        self.assertTrue(parsed.has_error_pred)
        self.assertEqual(parsed.error_type_pred, "agreement")

    def test_metrics_are_exported_for_normalized_results(self) -> None:
        """Metric command should create the required summary slices."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            normalized_path = tmp_path / "normalized_results.csv"
            write_normalized(normalized_path)
            metric_paths = compute_and_export_metrics(normalized_path, tmp_path / "metrics")

            names = {path.name for path in metric_paths}
            self.assertIn("metrics_summary.csv", names)
            self.assertIn("metrics_by_rule.csv", names)
            summary = read_csv(tmp_path / "metrics" / "metrics_summary.csv")
            self.assertEqual(summary[0]["total"], "2")
            self.assertEqual(summary[0]["detection_accuracy"], "1.000000")

    def test_exact_match_uses_incorrect_denominator(self) -> None:
        """Exact-match should not be inflated by copied correct controls."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            normalized_path = tmp_path / "normalized_results.csv"
            rows = normalized_rows()
            rows[0]["correction_pred"] = rows[0]["gold_correction"]
            rows[1]["correction_pred"] = rows[1]["sentence"]
            write_normalized_rows(normalized_path, rows)

            compute_and_export_metrics(normalized_path, tmp_path / "metrics")

            summary = read_csv(tmp_path / "metrics" / "metrics_summary.csv")[0]
            self.assertEqual(summary["correction_accuracy"], "0.000000")
            self.assertEqual(summary["exact_match"], "0.000000")
            self.assertEqual(summary["exact_match_all"], "0.500000")
            self.assertEqual(summary["exact_match_correct"], "1.000000")


def write_items(path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    """Write sample final_items rows."""

    data = rows or SAMPLE_ITEMS
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return path


def write_normalized(path: Path) -> Path:
    """Write two normalized rows with perfect predictions."""

    write_normalized_rows(path, normalized_rows())
    return path


def normalized_rows() -> list[dict[str, str]]:
    """Return two normalized rows with perfect predictions."""

    rows = [
        {
            **SAMPLE_ITEMS[0],
            "run_id": "metrics_test",
            "provider": "mock",
            "model": "mock-model",
            "prompt_id": "direct_zero_shot",
            "has_error_pred": "false",
            "error_type_pred": "none",
            "correction_pred": SAMPLE_ITEMS[0]["gold_correction"],
            "parse_status": "ok",
            "request_status": "succeeded",
            "raw_text": json.dumps({"has_error": False, "error_type": "none", "correction": SAMPLE_ITEMS[0]["gold_correction"]}),
            "raw_json": "{}",
            "input_tokens": "10",
            "output_tokens": "8",
            "latency_ms": "15",
            "timestamp": "2026-05-08T00:00:00+00:00",
        },
        {
            **SAMPLE_ITEMS[1],
            "run_id": "metrics_test",
            "provider": "mock",
            "model": "mock-model",
            "prompt_id": "direct_zero_shot",
            "has_error_pred": "true",
            "error_type_pred": "agreement",
            "correction_pred": SAMPLE_ITEMS[1]["gold_correction"],
            "parse_status": "ok",
            "request_status": "succeeded",
            "raw_text": json.dumps({"has_error": True, "error_type": "agreement", "correction": SAMPLE_ITEMS[1]["gold_correction"]}),
            "raw_json": "{}",
            "input_tokens": "11",
            "output_tokens": "9",
            "latency_ms": "16",
            "timestamp": "2026-05-08T00:00:01+00:00",
        },
    ]
    return rows


def write_normalized_rows(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write normalized rows to CSV."""

    fieldnames = tuple(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


if __name__ == "__main__":
    unittest.main()
