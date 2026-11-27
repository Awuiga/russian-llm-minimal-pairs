"""Automatic quality checks for generated minimal pairs."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import ceil
from typing import Iterable

from generator import MinimalPair


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)?")

MECHANICAL_FRAGMENTS: tuple[str, ...] = (
    "сейчас сразу",
    "днем в кабинете",
    "старую карту быстро",
    "быстро ранним утром",
    "сегодня сегодня",
    "вчера вчера",
    "утром утром",
)


@dataclass
class ValidationReport:
    """Container for validation errors and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` if no blocking validation errors were found."""

        return not self.errors


def validate_pairs(
    pairs: Iterable[MinimalPair],
    *,
    min_words: int = 7,
    max_words: int = 14,
    similarity_threshold: float = 0.88,
) -> ValidationReport:
    """Run deterministic quality checks over generated pairs.

    These checks cannot prove linguistic correctness, but they catch common
    corpus-construction mistakes: empty strings, duplicates, overly long
    sentences, identical pair members, and broad multi-fragment changes.
    """

    pair_list = list(pairs)
    report = ValidationReport()
    seen_pair_ids: set[str] = set()
    seen_sentence_pairs: set[tuple[str, str]] = set()
    by_rule: dict[str, list[MinimalPair]] = defaultdict(list)

    for pair in pair_list:
        by_rule[pair.rule_id].append(pair)

        if pair.pair_id in seen_pair_ids:
            report.errors.append(f"{pair.pair_id}: duplicate pair_id")
        seen_pair_ids.add(pair.pair_id)

        sentence_key = (pair.correct_sentence, pair.incorrect_sentence)
        if sentence_key in seen_sentence_pairs:
            report.errors.append(f"{pair.pair_id}: duplicate sentence pair")
        seen_sentence_pairs.add(sentence_key)

        if not pair.correct_sentence.strip():
            report.errors.append(f"{pair.pair_id}: correct_sentence is empty")
        if not pair.incorrect_sentence.strip():
            report.errors.append(f"{pair.pair_id}: incorrect_sentence is empty")
        if pair.correct_sentence == pair.incorrect_sentence:
            report.errors.append(f"{pair.pair_id}: correct and incorrect sentences are identical")
        if not pair.gold_correction.strip():
            report.errors.append(f"{pair.pair_id}: gold_correction is empty")
        if pair.manual_status not in {"pending", "approved", "rejected"}:
            report.errors.append(f"{pair.pair_id}: manual_status has an unsupported value")

        for label, sentence in (
            ("correct_sentence", pair.correct_sentence),
            ("incorrect_sentence", pair.incorrect_sentence),
        ):
            word_count = count_words(sentence)
            if word_count < min_words:
                report.errors.append(
                    f"{pair.pair_id}: {label} has {word_count} words; minimum is {min_words}"
                )
            if word_count > max_words:
                report.errors.append(
                    f"{pair.pair_id}: {label} has {word_count} words; maximum is {max_words}"
                )

        diff = single_contiguous_diff(pair.correct_sentence, pair.incorrect_sentence)
        if diff is None:
            report.errors.append(f"{pair.pair_id}: could not locate a single changed fragment")
        else:
            correct_fragment, incorrect_fragment = diff
            changed_words = max(count_words(correct_fragment), count_words(incorrect_fragment))
            if changed_words > 4:
                report.warnings.append(
                    f"{pair.pair_id}: changed fragment has {changed_words} words; inspect manually"
                )

        for flag in naturalness_flags(pair.correct_sentence):
            report.warnings.append(f"{pair.pair_id}: naturalness flag in correct_sentence: {flag}")
        for flag in naturalness_flags(pair.incorrect_sentence):
            report.warnings.append(f"{pair.pair_id}: naturalness flag in incorrect_sentence: {flag}")

    _validate_diversity(by_rule, report, similarity_threshold=similarity_threshold)

    return report


def count_words(sentence: str) -> int:
    """Count word-like tokens in Russian and Latin text."""

    return len(WORD_RE.findall(sentence))


def naturalness_flags(sentence: str) -> list[str]:
    """Return simple heuristic flags for mechanical-sounding fragments."""

    lowered = sentence.casefold()
    flags = [fragment for fragment in MECHANICAL_FRAGMENTS if fragment in lowered]
    tokens = normalize_tokens(sentence)
    if len(tokens) != len(set(tokens)) and any(tokens.count(token) > 1 for token in ("сразу", "сегодня", "вчера", "утром")):
        flags.append("repeated time/adverb token")
    return flags


def normalize_tokens(sentence: str) -> list[str]:
    """Return lower-cased word tokens for diversity checks."""

    return [match.group(0).casefold() for match in WORD_RE.finditer(sentence)]


def sentence_similarity(first: str, second: str) -> float:
    """Return token-level sequence similarity for two sentences."""

    return SequenceMatcher(a=normalize_tokens(first), b=normalize_tokens(second)).ratio()


def single_contiguous_diff(first: str, second: str) -> tuple[str, str] | None:
    """Return the single differing substring between two strings.

    Any pair of strings has a maximal prefix and suffix in common. If both
    strings are equal, there is no changed fragment, so ``None`` is returned.
    """

    if first == second:
        return None

    prefix_len = 0
    min_len = min(len(first), len(second))
    while prefix_len < min_len and first[prefix_len] == second[prefix_len]:
        prefix_len += 1

    first_suffix = len(first)
    second_suffix = len(second)
    while (
        first_suffix > prefix_len
        and second_suffix > prefix_len
        and first[first_suffix - 1] == second[second_suffix - 1]
    ):
        first_suffix -= 1
        second_suffix -= 1

    return first[prefix_len:first_suffix], second[prefix_len:second_suffix]


def _validate_diversity(
    by_rule: dict[str, list[MinimalPair]],
    report: ValidationReport,
    *,
    similarity_threshold: float,
) -> None:
    """Warn about repeated frames and very similar correct sentences."""

    for rule_id, rule_pairs in by_rule.items():
        if len(rule_pairs) < 2:
            continue

        template_counts: dict[str, int] = defaultdict(int)
        signature_counts: dict[str, int] = defaultdict(int)
        for pair in rule_pairs:
            template_counts[pair.template_id] += 1
            signature_counts[pair.lexical_signature] += 1

        template_limit = max(2, ceil(len(rule_pairs) / max(1, len(template_counts))) + 1)
        signature_limit = max(2, ceil(len(rule_pairs) / max(1, len(signature_counts))) + 1)

        for template_id, count in sorted(template_counts.items()):
            if count > template_limit:
                report.warnings.append(
                    f"{rule_id}: template {template_id!r} appears {count} times; target limit is {template_limit}"
                )
        for lexical_signature, count in sorted(signature_counts.items()):
            if lexical_signature and count > signature_limit:
                report.warnings.append(
                    f"{rule_id}: lexical frame {lexical_signature!r} appears {count} times; "
                    f"target limit is {signature_limit}"
                )

        for left_index, left_pair in enumerate(rule_pairs):
            for right_pair in rule_pairs[left_index + 1 :]:
                if left_pair.lexical_signature == right_pair.lexical_signature:
                    continue
                similarity = sentence_similarity(left_pair.correct_sentence, right_pair.correct_sentence)
                if similarity >= similarity_threshold:
                    report.warnings.append(
                        f"{left_pair.pair_id}/{right_pair.pair_id}: correct sentences are too similar "
                        f"({similarity:.2f})"
                    )
