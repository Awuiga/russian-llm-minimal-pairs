"""Metric computation for normalized API experiment results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from storage import read_normalized_results, write_rows_csv


METRIC_FIELDNAMES: tuple[str, ...] = (
    "group",
    "value",
    "total",
    "parse_ok",
    "detection_accuracy",
    "correction_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "exact_match",
    "exact_match_all",
    "exact_match_correct",
)


def compute_and_export_metrics(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Compute required metric slices and write CSV files."""

    rows = read_normalized_results(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics_summary.csv": [compute_metrics_row(rows, group="all", value="all")],
        "metrics_by_rule.csv": grouped_metrics(rows, "rule_id"),
        "metrics_by_prompt.csv": grouped_metrics(rows, "prompt_id"),
        "metrics_by_model.csv": grouped_metrics(rows, "model"),
        "metrics_by_provider.csv": grouped_metrics(rows, "provider"),
        "metrics_by_difficulty.csv": grouped_metrics(rows, "difficulty"),
        "metrics_by_variant.csv": grouped_metrics(rows, "variant"),
    }
    paths: list[Path] = []
    for filename, metric_rows in outputs.items():
        path = target_dir / filename
        write_rows_csv(metric_rows, path, METRIC_FIELDNAMES)
        paths.append(path)
    return paths


def grouped_metrics(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Compute metrics grouped by one normalized-result field."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "") or "unknown")].append(row)
    return [compute_metrics_row(group_rows, group=field, value=value) for value, group_rows in sorted(groups.items())]


def compute_metrics_row(rows: Iterable[dict[str, Any]], *, group: str, value: str) -> dict[str, Any]:
    """Compute detection/correction metrics for one group."""

    row_list = list(rows)
    total = len(row_list)
    parse_ok = sum(1 for row in row_list if row.get("parse_status") in {"ok", "partial"})
    detection_correct = 0
    exact_matches_all = 0
    exact_matches_incorrect = 0
    exact_matches_correct = 0
    correction_correct = 0
    incorrect_total = 0
    correct_total = 0
    false_positives = 0
    false_negatives = 0

    for row in row_list:
        expected_has_error = str(row.get("gold_label", "")).strip() == "0"
        predicted_has_error = parse_bool(row.get("has_error_pred"))
        correction_match = normalize_sentence(row.get("correction_pred")) == normalize_sentence(row.get("gold_correction"))
        if predicted_has_error is not None and predicted_has_error == expected_has_error:
            detection_correct += 1
        if correction_match:
            exact_matches_all += 1

        if expected_has_error:
            incorrect_total += 1
            if correction_match:
                exact_matches_incorrect += 1
            if predicted_has_error is False:
                false_negatives += 1
            if predicted_has_error is True and correction_match:
                correction_correct += 1
        else:
            correct_total += 1
            if correction_match:
                exact_matches_correct += 1
            if predicted_has_error is True:
                false_positives += 1

    return {
        "group": group,
        "value": value,
        "total": total,
        "parse_ok": parse_ok,
        "detection_accuracy": safe_rate(detection_correct, total),
        "correction_accuracy": safe_rate(correction_correct, incorrect_total),
        "false_positive_rate": safe_rate(false_positives, correct_total),
        "false_negative_rate": safe_rate(false_negatives, incorrect_total),
        # Keep exact_match aligned with correction evaluation: only items that
        # actually require correction are in the denominator. The old global
        # value is still exported as exact_match_all for no-change analysis.
        "exact_match": safe_rate(exact_matches_incorrect, incorrect_total),
        "exact_match_all": safe_rate(exact_matches_all, total),
        "exact_match_correct": safe_rate(exact_matches_correct, correct_total),
    }


def parse_bool(value: Any) -> bool | None:
    """Parse booleans persisted through CSV/JSONL."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().casefold()
    if lowered in {"true", "1", "yes", "да"}:
        return True
    if lowered in {"false", "0", "no", "нет"}:
        return False
    return None


def normalize_sentence(value: Any) -> str:
    """Normalize sentence strings for exact-match metrics."""

    return " ".join(str(value or "").strip().casefold().split())


def safe_rate(numerator: int, denominator: int) -> str:
    """Format a rate or return an empty value for zero-denominator groups."""

    if denominator == 0:
        return ""
    return f"{numerator / denominator:.6f}"
