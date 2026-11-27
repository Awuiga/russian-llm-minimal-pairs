"""Unit tests for core generation, corruption, validation, and export."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from corrupter import create_corruption
from exporter import export_pairs
from generator import generate_correct_sentence, generate_pairs, pairs_to_evaluation_items
from lexicon import get_choices
from templates import get_templates
from validator import validate_pairs


class MinimalPairGeneratorTests(unittest.TestCase):
    """Focused tests for the deterministic corpus pipeline."""

    def test_generates_correct_sentence(self) -> None:
        sentence = generate_correct_sentence("subject_verb_agreement")
        self.assertIn("врачи", sentence)
        self.assertTrue(sentence.endswith("."))

    def test_applies_corruption_transformation(self) -> None:
        template = get_templates("tsya_tsa")[0]
        fields = get_choices("tsya_tsa")[0].as_dict()
        correct_sentence = template.render_correct(fields)
        corruption = create_corruption(template, fields)
        self.assertNotEqual(correct_sentence, corruption.incorrect_sentence)
        self.assertEqual("учится", corruption.gold_correction)
        self.assertIn("учиться", corruption.changed_span)

    def test_generated_pairs_have_no_empty_strings(self) -> None:
        pairs = generate_pairs(num_per_rule=1, seed=7)
        for pair in pairs:
            self.assertTrue(pair.correct_sentence.strip())
            self.assertTrue(pair.incorrect_sentence.strip())
            self.assertTrue(pair.gold_correction.strip())

    def test_exports_csv_and_json(self) -> None:
        pairs = generate_pairs(num_per_rule=1, rule_ids=["ne_ni"], seed=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_pairs(
                pairs,
                output_dir=temp_dir,
                formats=["csv", "json"],
                output_name="sample",
                include_items=True,
            )
            self.assertEqual(len(paths), 4)

            csv_path = Path(temp_dir) / "sample.csv"
            json_path = Path(temp_dir) / "sample.json"
            items_csv_path = Path(temp_dir) / "sample_items.csv"
            items_json_path = Path(temp_dir) / "sample_items.json"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            with json_path.open("r", encoding="utf-8") as file:
                json_rows = json.load(file)
            with items_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                item_rows = list(csv.DictReader(file))
            with items_json_path.open("r", encoding="utf-8") as file:
                json_item_rows = json.load(file)

            self.assertEqual(len(rows), 1)
            self.assertEqual(len(json_rows), 1)
            self.assertEqual(len(item_rows), 2)
            self.assertEqual(len(json_item_rows), 2)
            self.assertEqual(rows[0]["pair_id"], json_rows[0]["pair_id"])
            self.assertEqual({"correct", "incorrect"}, {row["variant"] for row in item_rows})

    def test_no_duplicates_in_small_dataset(self) -> None:
        pairs = generate_pairs(num_per_rule=2, seed=11)
        report = validate_pairs(pairs)
        self.assertTrue(report.ok, report.errors)
        sentence_pairs = {(pair.correct_sentence, pair.incorrect_sentence) for pair in pairs}
        self.assertEqual(len(sentence_pairs), len(pairs))

    def test_evaluation_items_include_correct_controls(self) -> None:
        pairs = generate_pairs(num_per_rule=1, rule_ids=["subject_verb_agreement"], seed=3)
        items = pairs_to_evaluation_items(pairs)
        self.assertEqual(len(items), 2)
        correct_item = next(item for item in items if item.variant == "correct")
        incorrect_item = next(item for item in items if item.variant == "incorrect")
        self.assertEqual(correct_item.gold_label, 1)
        self.assertEqual(correct_item.expected_action, "accept")
        self.assertEqual(incorrect_item.gold_label, 0)
        self.assertEqual(incorrect_item.expected_action, "correct")

    def test_manual_review_fields_are_initialized(self) -> None:
        pair = generate_pairs(num_per_rule=1, rule_ids=["introductory_comma"], seed=4)[0]
        self.assertFalse(pair.manual_reviewed)
        self.assertEqual(pair.manual_status, "pending")
        self.assertEqual(pair.naturalness_score, "")
        self.assertEqual(pair.normative_confidence, "")


if __name__ == "__main__":
    unittest.main()
