"""Unit tests for the manual review workflow."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from exporter import FIELDNAMES, write_csv
from generator import generate_pairs
from review import (
    REVIEW_FIELDNAMES,
    filter_approved_pairs,
    finalize_review,
    prepare_review,
    review_report,
)


class ReviewWorkflowTests(unittest.TestCase):
    """Tests for prepare/finalize/report review commands."""

    def test_prepare_review_creates_review_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "pairs.csv"
            rows = [pair.to_dict() for pair in generate_pairs(num_per_rule=1, seed=5)]
            write_csv(rows, input_path, fieldnames=FIELDNAMES)

            result = prepare_review(input_path, temp_path / "review")

            expected = {
                "review_pairs.csv",
                "review_pairs.json",
                "suspicious_pairs.csv",
                "suspicious_pairs.json",
            }
            self.assertEqual({path.name for path in result.paths}, expected)
            with (temp_path / "review" / "review_pairs.csv").open("r", encoding="utf-8-sig", newline="") as file:
                review_rows = list(csv.DictReader(file))
            self.assertEqual(len(review_rows), len(rows))
            self.assertIn("auto_flag", review_rows[0])
            self.assertIn("review_priority", review_rows[0])

    def test_filter_approved_pairs(self) -> None:
        rows = [
            {"manual_reviewed": "true", "manual_status": "approved"},
            {"manual_reviewed": "false", "manual_status": "approved"},
            {"manual_reviewed": "true", "manual_status": "rejected"},
            {"manual_reviewed": "true", "manual_status": "accepted"},
        ]
        approved = filter_approved_pairs(rows)
        self.assertEqual(len(approved), 2)

    def test_finalize_review_exports_clean_pairs_and_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "pairs.csv"
            rows = [pair.to_dict() for pair in generate_pairs(num_per_rule=2, rule_ids=["ne_ni"], seed=8)]
            write_csv(rows, input_path, fieldnames=FIELDNAMES)

            prepare_review(input_path, temp_path / "review")
            review_csv = temp_path / "review" / "review_pairs.csv"
            review_rows = _read_csv(review_csv)
            review_rows[0]["manual_reviewed"] = "true"
            review_rows[0]["manual_status"] = "approved"
            review_rows[0]["naturalness_score"] = "4"
            review_rows[0]["normative_confidence"] = "5"
            review_rows[1]["manual_reviewed"] = "true"
            review_rows[1]["manual_status"] = "rejected"
            _write_review_csv(review_rows, review_csv)

            result = finalize_review(review_csv, temp_path / "final")

            self.assertEqual(result.stats["total_pairs"], 2)
            self.assertEqual(result.stats["approved_pairs"], 1)
            self.assertEqual(result.stats["rejected_pairs"], 1)
            with (temp_path / "final" / "final_pairs.csv").open("r", encoding="utf-8-sig", newline="") as file:
                final_pairs = list(csv.DictReader(file))
            with (temp_path / "final" / "final_items.json").open("r", encoding="utf-8") as file:
                final_items = json.load(file)
            self.assertEqual(len(final_pairs), 1)
            self.assertEqual(len(final_items), 2)
            self.assertNotIn("manual_status", final_pairs[0])
            self.assertEqual({"correct", "incorrect"}, {item["variant"] for item in final_items})

    def test_review_report_creates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "pairs.csv"
            rows = [pair.to_dict() for pair in generate_pairs(num_per_rule=1, rule_ids=["case_government"], seed=9)]
            rows[0]["manual_reviewed"] = True
            rows[0]["manual_status"] = "approved"
            rows[0]["naturalness_score"] = "4"
            rows[0]["normative_confidence"] = "5"
            write_csv(rows, input_path, fieldnames=FIELDNAMES)

            result = review_report(input_path, temp_path / "report")

            self.assertIn(temp_path / "report" / "review_report.txt", result.paths)
            self.assertIn(temp_path / "report" / "review_report.json", result.paths)
            self.assertEqual(result.stats["approved_pairs"], 1)
            self.assertEqual(result.stats["average_naturalness_score"], 4.0)

    def test_prepare_review_builds_suspicious_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "pairs.csv"
            pair = generate_pairs(num_per_rule=1, rule_ids=["subject_verb_agreement"], seed=10)[0].to_dict()
            rows = []
            for index in range(4):
                duplicated = dict(pair)
                duplicated["pair_id"] = f"dup_{index}"
                rows.append(duplicated)
            write_csv(rows, input_path, fieldnames=FIELDNAMES)

            prepare_review(input_path, temp_path / "review")

            with (temp_path / "review" / "suspicious_pairs.json").open("r", encoding="utf-8") as file:
                suspicious_rows = json.load(file)
            self.assertGreaterEqual(len(suspicious_rows), 1)
            self.assertTrue(any(row["auto_flag"] for row in suspicious_rows))
            self.assertTrue(any("duplicate sentence pair" in row["auto_flag_reason"] for row in suspicious_rows))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_review_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
