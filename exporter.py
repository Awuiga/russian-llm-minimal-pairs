"""Export generated minimal pairs to CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from generator import MinimalPair, pairs_to_evaluation_items


FIELDNAMES: tuple[str, ...] = (
    "pair_id",
    "rule_id",
    "rule_name",
    "error_type",
    "difficulty",
    "template_id",
    "lexical_signature",
    "correct_sentence",
    "incorrect_sentence",
    "gold_label_correct",
    "gold_label_incorrect",
    "gold_correction",
    "changed_span",
    "comment",
    "manual_reviewed",
    "manual_status",
    "manual_comment",
    "naturalness_score",
    "normative_confidence",
    "review_priority",
    "auto_flag",
    "auto_flag_reason",
)

ITEM_FIELDNAMES: tuple[str, ...] = (
    "item_id",
    "pair_id",
    "rule_id",
    "rule_name",
    "error_type",
    "difficulty",
    "template_id",
    "lexical_signature",
    "variant",
    "sentence",
    "gold_label",
    "gold_correction",
    "changed_span",
    "expected_action",
    "manual_reviewed",
    "manual_status",
    "manual_comment",
    "naturalness_score",
    "normative_confidence",
    "review_priority",
    "auto_flag",
    "auto_flag_reason",
    "comment",
)


def export_pairs(
    pairs: Iterable[MinimalPair],
    *,
    output_dir: str | Path,
    formats: Iterable[str],
    output_name: str = "pairs",
    include_items: bool = False,
) -> list[Path]:
    """Export pairs to the requested file formats.

    Args:
        pairs: Generated minimal pairs.
        output_dir: Directory that will be created if needed.
        formats: Iterable containing ``csv`` and/or ``json``.
        output_name: File stem without extension.
        include_items: Also write ``*_items`` files for API evaluation.

    Returns:
        Paths to written files.
    """

    pair_list = list(pairs)
    pair_dicts = [pair.to_dict() for pair in pair_list]
    item_dicts = [item.to_dict() for item in pairs_to_evaluation_items(pair_list)]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    normalized_formats = {fmt.lower() for fmt in formats}
    unknown = normalized_formats - {"csv", "json"}
    if unknown:
        raise ValueError(f"Unsupported export format(s): {', '.join(sorted(unknown))}")

    if "csv" in normalized_formats:
        csv_path = target_dir / f"{output_name}.csv"
        write_csv(pair_dicts, csv_path, fieldnames=FIELDNAMES)
        written.append(csv_path)
        if include_items:
            items_csv_path = target_dir / f"{output_name}_items.csv"
            write_csv(item_dicts, items_csv_path, fieldnames=ITEM_FIELDNAMES)
            written.append(items_csv_path)

    if "json" in normalized_formats:
        json_path = target_dir / f"{output_name}.json"
        write_json(pair_dicts, json_path)
        written.append(json_path)
        if include_items:
            items_json_path = target_dir / f"{output_name}_items.json"
            write_json(item_dicts, items_json_path)
            written.append(items_json_path)

    return written


def write_csv(rows: list[dict[str, object]], path: str | Path, *, fieldnames: tuple[str, ...]) -> None:
    """Write rows as UTF-8 CSV with a stable field order."""

    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write rows as pretty UTF-8 JSON."""

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")
