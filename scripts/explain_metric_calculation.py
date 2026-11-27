"""Explain how correction and exact-match metrics are calculated.

This helper is intentionally small and dependency-free. It reads a
``normalized_results.csv`` file, recalculates the confusion counts and
correction counts, then writes CSV/Markdown files that show the exact
numerators and denominators behind the reported metrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("runs/main_experiment_v4/normalized_results.csv.gz")
DEFAULT_OUTPUT_DIR = Path("scratch/metric_calculation_proof")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Explain metric calculations from normalized_results.csv.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to normalized_results.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for explanation files")
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["model"],
        help="One or more grouping fields, for example: model prompt_id provider",
    )
    return parser.parse_args()


def main() -> int:
    """Run the explanation export."""

    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_path)
    for field in args.group_by:
        grouped_rows = explain_by_group(rows, field)
        write_csv(output_dir / f"metric_calculation_by_{field}.csv", grouped_rows)

    all_rows = explain_one_group("all", "all", rows)
    write_csv(output_dir / "metric_calculation_overall.csv", [all_rows])

    claude_cases = exact_without_detection_cases(rows, model="claude-haiku-4-5-20251001")
    write_csv(output_dir / "claude_exact_match_without_detection_cases.csv", claude_cases)

    write_readme(output_dir, input_path)
    print(f"Wrote metric calculation proof files to {output_dir}")
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read UTF-8 CSV rows."""

    if path.name.lower().endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, preserving discovered field order."""

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def explain_by_group(rows: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    """Return metric explanation rows grouped by one field."""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(field, "") or "unknown"].append(row)
    return [explain_one_group(field, value, group_rows) for value, group_rows in sorted(groups.items())]


def explain_one_group(group: str, value: str, rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Calculate counts and formulas for one group."""

    row_list = list(rows)
    correct_rows = [row for row in row_list if row.get("gold_label") == "1"]
    incorrect_rows = [row for row in row_list if row.get("gold_label") == "0"]

    tp = sum(1 for row in incorrect_rows if parse_bool(row.get("has_error_pred")) is True)
    fn = sum(1 for row in incorrect_rows if parse_bool(row.get("has_error_pred")) is False)
    tn = sum(1 for row in correct_rows if parse_bool(row.get("has_error_pred")) is False)
    fp = sum(1 for row in correct_rows if parse_bool(row.get("has_error_pred")) is True)
    unknown = len(row_list) - tp - fn - tn - fp

    exact_incorrect = sum(1 for row in incorrect_rows if is_exact(row))
    exact_correct = sum(1 for row in correct_rows if is_exact(row))
    exact_all = exact_incorrect + exact_correct
    correction_success = sum(
        1 for row in incorrect_rows if parse_bool(row.get("has_error_pred")) is True and is_exact(row)
    )

    total = len(row_list)
    correct_total = len(correct_rows)
    incorrect_total = len(incorrect_rows)

    return {
        "group": group,
        "value": value,
        "total": total,
        "correct_total": correct_total,
        "incorrect_total": incorrect_total,
        "TP_detected_error": tp,
        "FN_missed_error": fn,
        "TN_accepted_correct": tn,
        "FP_overcorrected_correct": fp,
        "unknown_detection": unknown,
        "correction_success_count": correction_success,
        "exact_incorrect_count": exact_incorrect,
        "exact_correct_count": exact_correct,
        "exact_all_count": exact_all,
        "detection_accuracy_formula": formula(tp + tn, total),
        "detection_accuracy": rate(tp + tn, total),
        "correction_accuracy_formula": formula(correction_success, incorrect_total),
        "correction_accuracy": rate(correction_success, incorrect_total),
        "false_positive_rate_formula": formula(fp, correct_total),
        "false_positive_rate": rate(fp, correct_total),
        "false_negative_rate_formula": formula(fn, incorrect_total),
        "false_negative_rate": rate(fn, incorrect_total),
        "exact_match_formula": formula(exact_incorrect, incorrect_total),
        "exact_match": rate(exact_incorrect, incorrect_total),
        "exact_match_correct_formula": formula(exact_correct, correct_total),
        "exact_match_correct": rate(exact_correct, correct_total),
        "exact_match_all_formula": formula(exact_all, total),
        "exact_match_all": rate(exact_all, total),
        "TPR_plus_FNR": rate(tp + fn, incorrect_total),
        "TNR_plus_FPR": rate(tn + fp, correct_total),
    }


def exact_without_detection_cases(rows: Iterable[dict[str, str]], *, model: str) -> list[dict[str, str]]:
    """List incorrect rows where text is exact but has_error_pred is not true."""

    cases: list[dict[str, str]] = []
    for row in rows:
        if row.get("model") != model:
            continue
        if row.get("gold_label") != "0":
            continue
        if not is_exact(row):
            continue
        if parse_bool(row.get("has_error_pred")) is True:
            continue
        cases.append(
            {
                "model": row.get("model", ""),
                "prompt_id": row.get("prompt_id", ""),
                "item_id": row.get("item_id", ""),
                "rule_id": row.get("rule_id", ""),
                "sentence": row.get("sentence", ""),
                "has_error_pred": row.get("has_error_pred", ""),
                "correction_pred": row.get("correction_pred", ""),
                "gold_correction": row.get("gold_correction", ""),
                "why_exact_but_not_correction_accuracy": (
                    "correction_pred equals gold_correction, but has_error_pred is not true"
                ),
            }
        )
    return cases


def write_readme(output_dir: Path, input_path: Path) -> None:
    """Write a short explanation file next to CSV outputs."""

    text = f"""# Metric Calculation Proof

Source file: `{input_path}`

The key difference:

```text
exact_match = exact_incorrect_count / incorrect_total
correction_accuracy = correction_success_count / incorrect_total
```

Where:

```text
exact_incorrect_count = incorrect rows where correction_pred == gold_correction
correction_success_count = incorrect rows where has_error_pred=true AND correction_pred == gold_correction
```

Therefore:

```text
correction_accuracy <= exact_match
```

If the two values differ, see:

```text
claude_exact_match_without_detection_cases.csv
```

That file lists rows where the correction text was exact, but the model did not explicitly mark
`has_error_pred=true`, so `exact_match` was counted and `correction_accuracy` was not.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_bool(value: Any) -> bool | None:
    """Parse booleans persisted through CSV."""

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


def is_exact(row: dict[str, str]) -> bool:
    """Return whether predicted correction exactly matches the gold correction."""

    return normalize_sentence(row.get("correction_pred")) == normalize_sentence(row.get("gold_correction"))


def normalize_sentence(value: Any) -> str:
    """Normalize text for exact-match comparison."""

    return " ".join(str(value or "").strip().casefold().split())


def rate(numerator: int, denominator: int) -> str:
    """Format a rate with six decimals."""

    if denominator == 0:
        return ""
    return f"{numerator / denominator:.6f}"


def formula(numerator: int, denominator: int) -> str:
    """Return a human-readable formula string."""

    if denominator == 0:
        return f"{numerator} / 0 = undefined"
    return f"{numerator} / {denominator} = {numerator / denominator:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
