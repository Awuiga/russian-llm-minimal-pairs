"""Storage utilities for API experiment runs."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


NORMALIZED_FIELDNAMES: tuple[str, ...] = (
    "run_id",
    "provider",
    "requested_model",
    "model",
    "fallback_used",
    "prompt_id",
    "item_id",
    "pair_id",
    "variant",
    "sentence",
    "gold_label",
    "gold_correction",
    "expected_action",
    "rule_id",
    "rule_name",
    "difficulty",
    "has_error_pred",
    "error_type_pred",
    "correction_pred",
    "parse_status",
    "request_status",
    "raw_text",
    "raw_json",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "timestamp",
)


class RunStorage:
    """File layout and append-only persistence for one API run."""

    def __init__(self, output_dir: str | Path, *, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_requests_path = self.output_dir / "raw_requests.jsonl"
        self.raw_responses_path = self.output_dir / "raw_responses.jsonl"
        self.normalized_jsonl_path = self.output_dir / "normalized_results.jsonl"
        self.normalized_csv_path = self.output_dir / "normalized_results.csv"
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.log_path = self.output_dir / "run_log.txt"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write run metadata as pretty JSON."""

        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def append_log(self, message: str) -> None:
        """Append one log line."""

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(message.rstrip() + "\n")

    def append_raw_request(self, row: dict[str, Any]) -> None:
        """Append one raw request record."""

        append_jsonl(self.raw_requests_path, row)

    def append_raw_response(self, row: dict[str, Any]) -> None:
        """Append one raw response record."""

        append_jsonl(self.raw_responses_path, row)

    def append_normalized_result(self, row: dict[str, Any]) -> None:
        """Append one normalized result to JSONL and CSV."""

        safe_row = {field: serialize_cell(row.get(field, "")) for field in NORMALIZED_FIELDNAMES}
        append_jsonl(self.normalized_jsonl_path, row)
        csv_exists = self.normalized_csv_path.exists()
        with self.normalized_csv_path.open("a", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=NORMALIZED_FIELDNAMES)
            if not csv_exists:
                writer.writeheader()
            writer.writerow(safe_row)

    def completed_keys(self) -> set[tuple[str, str, str, str]]:
        """Return completed provider/model/prompt/item keys for resume."""

        if not self.normalized_csv_path.exists():
            return set()
        keys: set[tuple[str, str, str, str]] = set()
        with self.normalized_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                requested_model = row.get("requested_model") or row["model"]
                keys.add((row["provider"], requested_model, row["prompt_id"], row["item_id"]))
        return keys


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append a JSON-serializable row to a JSONL file."""

    with Path(path).open("a", encoding="utf-8") as file:
        json.dump(to_jsonable(row), file, ensure_ascii=False)
        file.write("\n")


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Load final_items from CSV or JSON."""

    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError(f"{source} must contain a list of item objects")
        return [dict(row) for row in payload]
    raise ValueError(f"Unsupported item input format: {source.suffix}")


def read_normalized_results(path: str | Path) -> list[dict[str, Any]]:
    """Read normalized results from CSV, gzip CSV, or JSONL."""

    source = Path(path)
    if source.name.lower().endswith(".csv.gz"):
        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if source.suffix.lower() in {".jsonl", ".json"}:
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported normalized results format: {source.suffix}")


def write_rows_csv(rows: Iterable[dict[str, Any]], path: str | Path, fieldnames: tuple[str, ...]) -> None:
    """Write rows to CSV with a stable field order."""

    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_cell(row.get(field, "")) for field in fieldnames})


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and nested values to JSON-serializable data."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, tuple):
        return [to_jsonable(inner) for inner in value]
    return value


def serialize_cell(value: Any) -> str:
    """Serialize nested values for CSV cells."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
