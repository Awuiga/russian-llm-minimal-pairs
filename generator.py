"""Deterministic generation of Russian minimal pairs."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from math import ceil
from typing import Iterable

from corrupter import create_corruption
from lexicon import get_choices
from rules import get_rule, normalize_rule_ids
from templates import SentenceTemplate, get_templates


@dataclass(frozen=True)
class MinimalPair:
    """A correct sentence paired with one controlled erroneous variant."""

    pair_id: str
    rule_id: str
    rule_name: str
    error_type: str
    difficulty: str
    template_id: str
    lexical_signature: str
    correct_sentence: str
    incorrect_sentence: str
    gold_label_correct: int
    gold_label_incorrect: int
    gold_correction: str
    changed_span: str
    comment: str
    manual_reviewed: bool
    manual_status: str
    manual_comment: str
    naturalness_score: str
    normative_confidence: str
    review_priority: int
    auto_flag: bool
    auto_flag_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict with all export fields."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationItem:
    """A single API-facing test object derived from one minimal pair."""

    item_id: str
    pair_id: str
    rule_id: str
    rule_name: str
    error_type: str
    difficulty: str
    template_id: str
    lexical_signature: str
    variant: str
    sentence: str
    gold_label: int
    gold_correction: str
    changed_span: str
    expected_action: str
    manual_reviewed: bool
    manual_status: str
    manual_comment: str
    naturalness_score: str
    normative_confidence: str
    review_priority: int
    auto_flag: bool
    auto_flag_reason: str
    comment: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict with all export fields."""

        return asdict(self)


def generate_pairs(
    *,
    num_per_rule: int,
    rule_ids: Iterable[str] | None = None,
    seed: int = 42,
) -> list[MinimalPair]:
    """Generate a deterministic dataset of minimal pairs.

    Args:
        num_per_rule: Number of pairs to sample for each selected rule.
        rule_ids: Optional iterable of rule identifiers. ``None`` means all rules.
        seed: Random seed used only to shuffle deterministic candidates.

    Raises:
        ValueError: If a rule has fewer unique candidates than requested.
    """

    if num_per_rule < 1:
        raise ValueError("num_per_rule must be positive")

    selected_rule_ids = normalize_rule_ids(rule_ids)
    all_pairs: list[MinimalPair] = []

    for rule_index, rule_id in enumerate(selected_rule_ids):
        candidates = _build_rule_candidates(rule_id)
        rng = random.Random(seed + rule_index)
        rng.shuffle(candidates)
        candidates.sort(key=lambda candidate: _difficulty_rank(candidate.difficulty))

        unique_candidates = _deduplicate_candidates(candidates)
        selected_candidates = _select_diverse_candidates(unique_candidates, num_per_rule=num_per_rule)
        if len(selected_candidates) < num_per_rule:
            raise ValueError(
                f"Rule {rule_id!r} has only {len(selected_candidates)} diverse candidates; "
                f"requested {num_per_rule}. Add templates or lexical choices for this rule."
            )

        for pair_index, pair in enumerate(selected_candidates, start=1):
            all_pairs.append(
                MinimalPair(
                    pair_id=f"{rule_id}_{pair_index:04d}",
                    rule_id=pair.rule_id,
                    rule_name=pair.rule_name,
                    error_type=pair.error_type,
                    difficulty=pair.difficulty,
                    template_id=pair.template_id,
                    lexical_signature=pair.lexical_signature,
                    correct_sentence=pair.correct_sentence,
                    incorrect_sentence=pair.incorrect_sentence,
                    gold_label_correct=1,
                    gold_label_incorrect=0,
                    gold_correction=pair.gold_correction,
                    changed_span=pair.changed_span,
                    comment=pair.comment,
                    manual_reviewed=False,
                    manual_status="pending",
                    manual_comment="",
                    naturalness_score="",
                    normative_confidence="",
                    review_priority=0,
                    auto_flag=False,
                    auto_flag_reason="",
                )
            )

    return all_pairs


def generate_correct_sentence(rule_id: str, template: SentenceTemplate | None = None) -> str:
    """Generate one correct sentence for a rule.

    This helper is intentionally simple and is used in unit tests and examples.
    For full datasets, use :func:`generate_pairs`.
    """

    chosen_template = template or get_templates(rule_id)[0]
    fields = get_choices(rule_id)[0].as_dict()
    return chosen_template.render_correct(fields)


@dataclass(frozen=True)
class _CandidatePair:
    """Internal pair before final deterministic ids are assigned."""

    rule_id: str
    rule_name: str
    error_type: str
    difficulty: str
    template_id: str
    lexical_signature: str
    correct_sentence: str
    incorrect_sentence: str
    gold_correction: str
    changed_span: str
    comment: str


def _build_rule_candidates(rule_id: str) -> list[_CandidatePair]:
    """Render every template-choice combination for a rule."""

    rule = get_rule(rule_id)
    candidates: list[_CandidatePair] = []
    for template in get_templates(rule_id):
        for lexical_choice in get_choices(rule_id):
            fields = lexical_choice.as_dict()
            correct_sentence = template.render_correct(fields)
            corruption = create_corruption(template, fields)
            comments = [template.comment]
            if lexical_choice.comment:
                comments.append(lexical_choice.comment)
            effective_difficulty = _effective_difficulty(rule.rule_id, rule.difficulty, lexical_choice.difficulty, fields)
            candidates.append(
                _CandidatePair(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    error_type=rule.error_type,
                    difficulty=effective_difficulty,
                    template_id=template.template_id,
                    lexical_signature=_lexical_signature(rule_id, fields),
                    correct_sentence=correct_sentence,
                    incorrect_sentence=corruption.incorrect_sentence,
                    gold_correction=corruption.gold_correction,
                    changed_span=corruption.changed_span,
                    comment=" ".join(comments),
                )
            )
    return candidates


def pairs_to_evaluation_items(pairs: Iterable[MinimalPair]) -> list[EvaluationItem]:
    """Expand each pair into correct and incorrect API-facing items.

    The correct item is essential for measuring false positives: a model should
    accept it without inventing a correction.
    """

    items: list[EvaluationItem] = []
    for pair in pairs:
        items.append(
            EvaluationItem(
                item_id=f"{pair.pair_id}_correct",
                pair_id=pair.pair_id,
                rule_id=pair.rule_id,
                rule_name=pair.rule_name,
                error_type=pair.error_type,
                difficulty=pair.difficulty,
                template_id=pair.template_id,
                lexical_signature=pair.lexical_signature,
                variant="correct",
                sentence=pair.correct_sentence,
                gold_label=1,
                gold_correction=pair.correct_sentence,
                changed_span="",
                expected_action="accept",
                manual_reviewed=pair.manual_reviewed,
                manual_status=pair.manual_status,
                manual_comment=pair.manual_comment,
                naturalness_score=pair.naturalness_score,
                normative_confidence=pair.normative_confidence,
                review_priority=pair.review_priority,
                auto_flag=pair.auto_flag,
                auto_flag_reason=pair.auto_flag_reason,
                comment=pair.comment,
            )
        )
        items.append(
            EvaluationItem(
                item_id=f"{pair.pair_id}_incorrect",
                pair_id=pair.pair_id,
                rule_id=pair.rule_id,
                rule_name=pair.rule_name,
                error_type=pair.error_type,
                difficulty=pair.difficulty,
                template_id=pair.template_id,
                lexical_signature=pair.lexical_signature,
                variant="incorrect",
                sentence=pair.incorrect_sentence,
                gold_label=0,
                gold_correction=pair.correct_sentence,
                changed_span=pair.changed_span,
                expected_action="correct",
                manual_reviewed=pair.manual_reviewed,
                manual_status=pair.manual_status,
                manual_comment=pair.manual_comment,
                naturalness_score=pair.naturalness_score,
                normative_confidence=pair.normative_confidence,
                review_priority=pair.review_priority,
                auto_flag=pair.auto_flag,
                auto_flag_reason=pair.auto_flag_reason,
                comment=pair.comment,
            )
        )
    return items


def _deduplicate_candidates(candidates: Iterable[_CandidatePair]) -> list[_CandidatePair]:
    """Drop duplicate correct/incorrect sentence pairs while preserving order."""

    seen: set[tuple[str, str]] = set()
    unique: list[_CandidatePair] = []
    for candidate in candidates:
        key = (candidate.correct_sentence, candidate.incorrect_sentence)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _select_diverse_candidates(
    candidates: list[_CandidatePair],
    *,
    num_per_rule: int,
    similarity_threshold: float = 0.82,
) -> list[_CandidatePair]:
    """Select candidates while limiting repeated lexical and template frames."""

    if num_per_rule >= len(candidates):
        return candidates[:num_per_rule]

    template_count = max(1, len({candidate.template_id for candidate in candidates}))
    signature_count = max(1, len({candidate.lexical_signature for candidate in candidates}))
    template_limit = max(1, ceil(num_per_rule / template_count))
    signature_limit = max(1, ceil(num_per_rule / signature_count))

    selected: list[_CandidatePair] = []
    template_counts: dict[str, int] = {}
    signature_counts: dict[str, int] = {}

    for candidate in candidates:
        if len(selected) == num_per_rule:
            break
        if template_counts.get(candidate.template_id, 0) >= template_limit:
            continue
        if signature_counts.get(candidate.lexical_signature, 0) >= signature_limit:
            continue
        if any(_sentence_similarity(candidate.correct_sentence, chosen.correct_sentence) >= similarity_threshold for chosen in selected):
            continue
        selected.append(candidate)
        template_counts[candidate.template_id] = template_counts.get(candidate.template_id, 0) + 1
        signature_counts[candidate.lexical_signature] = signature_counts.get(candidate.lexical_signature, 0) + 1

    if len(selected) < num_per_rule:
        for candidate in candidates:
            if len(selected) == num_per_rule:
                break
            if candidate in selected:
                continue
            if template_counts.get(candidate.template_id, 0) >= template_limit + 1:
                continue
            if signature_counts.get(candidate.lexical_signature, 0) >= signature_limit + 1:
                continue
            selected.append(candidate)
            template_counts[candidate.template_id] = template_counts.get(candidate.template_id, 0) + 1
            signature_counts[candidate.lexical_signature] = signature_counts.get(candidate.lexical_signature, 0) + 1

    return selected


def _lexical_signature(rule_id: str, fields: dict[str, str]) -> str:
    """Build a coarse signature used to limit near-duplicate examples."""

    key_fields_by_rule = {
        "subject_verb_agreement": ("subject", "obj"),
        "adjective_noun_agreement": ("noun", "verb"),
        "case_government": ("subject", "verb"),
        "verb_form": ("subject", "helper", "obj"),
        "tsya_tsa": ("actor", "context"),
        "ne_ni": ("prefix", "suffix"),
        "n_nn": ("correct", "noun"),
        "fused_separate_spelling": ("prefix", "correct"),
        "complex_sentence_comma": ("main", "connector", "subordinate"),
        "introductory_comma": ("intro", "body"),
        "participial_phrase_comma": ("main", "participle"),
        "adverbial_participle_comma": ("gerund", "main"),
    }
    values = [fields.get(field_name, "") for field_name in key_fields_by_rule.get(rule_id, ())]
    return " | ".join(value.casefold() for value in values if value)


def _sentence_similarity(first: str, second: str) -> float:
    """Return a token-level similarity score between two correct sentences."""

    first_tokens = first.casefold().split()
    second_tokens = second.casefold().split()
    return SequenceMatcher(a=first_tokens, b=second_tokens).ratio()


def _effective_difficulty(rule_id: str, rule_difficulty: str, choice_difficulty: str, fields: dict[str, str]) -> str:
    """Assign a coarse difficulty label from rule metadata and lexical context."""

    if choice_difficulty:
        return choice_difficulty

    correct = fields.get("correct", "").casefold()
    incorrect = fields.get("incorrect", "").casefold()
    subject = fields.get("subject", "").casefold()
    prefix = fields.get("prefix", "").casefold()
    body = fields.get("body", "").casefold()

    if rule_id == "subject_verb_agreement" and any(
        marker in subject for marker in (" из ", " от ", " новых ", " обеих ", "главного")
    ):
        return "medium"
    if rule_id == "adjective_noun_agreement" and len(correct) > 0 and len(incorrect) > 0:
        if correct[:4] == incorrect[:4] or correct[-2:] == incorrect[-2:]:
            return "medium"
    if rule_id == "ne_ni" and (" " in correct or prefix == "" or correct.startswith("ни")):
        return "medium"
    if rule_id == "introductory_comma" and any(marker in body for marker in ("значит", "правда", "однако")):
        return "subtle"
    if rule_id in {"case_government", "complex_sentence_comma", "participial_phrase_comma", "adverbial_participle_comma"}:
        return "medium"

    return rule_difficulty


def _difficulty_rank(difficulty: str) -> int:
    """Prefer subtler candidates while preserving shuffled order inside groups."""

    return {"subtle": 0, "medium": 1, "easy": 2}.get(difficulty, 3)
