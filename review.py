"""Manual review workflow for generated Russian minimal-pair corpora.

The review stage is intentionally deterministic and dependency-free. It takes
the generated pair-level CSV/JSON files, adds review-oriented fields and
semi-automatic risk flags, then exports a clean approved-only dataset for the
LLM API experiment.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import ceil
from pathlib import Path
from typing import Iterable

from exporter import write_csv, write_json
from validator import count_words, naturalness_flags, normalize_tokens, single_contiguous_diff


MANUAL_STATUSES: tuple[str, ...] = ("approved", "rejected", "pending")

REVIEW_FIELDNAMES: tuple[str, ...] = (
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
    "manual_reviewed",
    "manual_status",
    "naturalness_score",
    "normative_confidence",
    "manual_comment",
    "review_priority",
    "auto_flag",
    "auto_flag_reason",
    "comment",
)

CLEAN_PAIR_FIELDNAMES: tuple[str, ...] = (
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
)

CLEAN_ITEM_FIELDNAMES: tuple[str, ...] = (
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
)

SUSPICIOUS_FIELDNAMES: tuple[str, ...] = REVIEW_FIELDNAMES

DIFFICULTY_ORDER: dict[str, int] = {"subtle": 0, "medium": 1, "easy": 2}

SERVICE_WORDS: set[str] = {
    "в",
    "во",
    "на",
    "за",
    "к",
    "ко",
    "с",
    "со",
    "из",
    "от",
    "по",
    "для",
    "перед",
    "после",
    "при",
    "без",
    "не",
    "ни",
    "что",
    "чтобы",
    "когда",
    "как",
    "так",
    "же",
    "ли",
    "бы",
    "потому",
    "если",
}

BLACKLIST_FRAGMENTS: tuple[str, ...] = (
    "сейчас сразу",
    "сегодня сегодня",
    "вчера вчера",
    "утром утром",
    "днем в кабинете",
    "быстро ранним утром",
    "старую карту быстро",
    "общую работу сейчас сразу",
)

UNUSUAL_PUNCTUATION_RE = re.compile(r"[;:!?()\[\]{}«»\"“”]")
REPEATED_PUNCTUATION_RE = re.compile(r"([,.])\1+")


@dataclass(frozen=True)
class ReviewRunResult:
    """Paths and stats produced by a review command."""

    paths: list[Path]
    stats: dict[str, object]


@dataclass(frozen=True)
class ReviewContext:
    """Corpus-level counts used by row-level risk heuristics."""

    template_counts: Counter[str]
    lexical_signature_counts: Counter[str]
    sentence_pair_counts: Counter[tuple[str, str]]


def load_pairs(path: str | Path) -> list[dict[str, object]]:
    """Load pair rows from CSV or JSON.

    JSON input must be a list of objects or an object with a top-level ``pairs``
    list. CSV is read as UTF-8 with BOM support.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
            payload = payload["pairs"]
        if not isinstance(payload, list):
            raise ValueError(f"{source} must contain a list of pair objects")
        return [dict(row) for row in payload]
    raise ValueError(f"Unsupported input format: {source.suffix}")


def prepare_review(input_path: str | Path, output_dir: str | Path) -> ReviewRunResult:
    """Create a normalized review table and a suspicious-pairs shortlist."""

    rows = normalize_review_rows(load_pairs(input_path))
    rows = sort_review_rows(rows)
    annotate_auto_review(rows)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    suspicious_rows = rank_suspicious_pairs(rows, limit=20)
    paths = [
        target_dir / "review_pairs.csv",
        target_dir / "review_pairs.json",
        target_dir / "suspicious_pairs.csv",
        target_dir / "suspicious_pairs.json",
    ]
    write_csv(rows, paths[0], fieldnames=REVIEW_FIELDNAMES)
    write_json(rows, paths[1])
    write_csv(suspicious_rows, paths[2], fieldnames=SUSPICIOUS_FIELDNAMES)
    write_json(suspicious_rows, paths[3])

    return ReviewRunResult(paths=paths, stats=build_review_stats(rows))


def finalize_review(input_path: str | Path, output_dir: str | Path) -> ReviewRunResult:
    """Export approved-only pair and item files for the final experiment."""

    rows = normalize_review_rows(load_pairs(input_path))
    annotate_auto_review(rows)
    approved_rows = [row for row in rows if is_approved(row)]
    clean_pairs = [clean_pair_row(row) for row in approved_rows]
    clean_items = make_clean_item_rows(approved_rows)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        target_dir / "final_pairs.csv",
        target_dir / "final_pairs.json",
        target_dir / "final_items.csv",
        target_dir / "final_items.json",
    ]
    write_csv(clean_pairs, paths[0], fieldnames=CLEAN_PAIR_FIELDNAMES)
    write_json(clean_pairs, paths[1])
    write_csv(clean_items, paths[2], fieldnames=CLEAN_ITEM_FIELDNAMES)
    write_json(clean_items, paths[3])

    return ReviewRunResult(paths=paths, stats=build_review_stats(rows, approved_rows=approved_rows))


def review_report(input_path: str | Path, output_dir: str | Path | None = None) -> ReviewRunResult:
    """Build a text and JSON report for a review table."""

    rows = normalize_review_rows(load_pairs(input_path))
    annotate_auto_review(rows)
    stats = build_review_stats(rows)
    suspicious_rows = rank_suspicious_pairs(rows, limit=20)
    stats["top_suspicious_pairs"] = [
        {
            "pair_id": row.get("pair_id", ""),
            "rule_id": row.get("rule_id", ""),
            "review_priority": row.get("review_priority", 0),
            "auto_flag": row.get("auto_flag", False),
            "auto_flag_reason": row.get("auto_flag_reason", ""),
            "correct_sentence": row.get("correct_sentence", ""),
            "incorrect_sentence": row.get("incorrect_sentence", ""),
        }
        for row in suspicious_rows
    ]

    target_dir = Path(output_dir) if output_dir is not None else Path(input_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        target_dir / "review_report.txt",
        target_dir / "review_report.json",
        target_dir / "suspicious_pairs.csv",
        target_dir / "suspicious_pairs.json",
    ]
    paths[0].write_text(format_text_report(stats), encoding="utf-8")
    write_json(stats, paths[1])
    write_csv(suspicious_rows, paths[2], fieldnames=SUSPICIOUS_FIELDNAMES)
    write_json(suspicious_rows, paths[3])
    return ReviewRunResult(paths=paths, stats=stats)


def normalize_review_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Add missing review fields and normalize status/boolean values."""

    normalized_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        normalized = {field: row.get(field, "") for field in REVIEW_FIELDNAMES}
        normalized["pair_id"] = str(normalized.get("pair_id") or f"pair_{index:04d}")
        normalized["gold_label_correct"] = normalized.get("gold_label_correct") or 1
        normalized["gold_label_incorrect"] = normalized.get("gold_label_incorrect") or 0
        normalized["manual_reviewed"] = parse_bool(normalized.get("manual_reviewed", False))
        normalized["manual_status"] = normalize_status(normalized.get("manual_status", "pending"))
        normalized["review_priority"] = parse_int(normalized.get("review_priority", 0), default=0)
        normalized["auto_flag"] = parse_bool(normalized.get("auto_flag", False))
        normalized["auto_flag_reason"] = str(normalized.get("auto_flag_reason", "") or "")
        normalized_rows.append(normalized)
    return normalized_rows


def sort_review_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Sort rows by rule, difficulty, and pair id for efficient manual review."""

    return sorted(
        rows,
        key=lambda row: (
            str(row.get("rule_id", "")),
            DIFFICULTY_ORDER.get(str(row.get("difficulty", "")), 99),
            str(row.get("template_id", "")),
            str(row.get("pair_id", "")),
        ),
    )


def annotate_auto_review(rows: list[dict[str, object]]) -> None:
    """Fill ``auto_flag``, ``auto_flag_reason`` and ``review_priority`` in place."""

    context = build_review_context(rows)
    for row in rows:
        reasons = auto_review_flags(row, all_rows=rows, context=context)
        row["auto_flag"] = bool(reasons)
        row["auto_flag_reason"] = "; ".join(reasons)
        row["review_priority"] = suspicious_score(row, context=context, flag_reasons=reasons)


def auto_review_flags(
    row: dict[str, object],
    *,
    all_rows: Iterable[dict[str, object]] | None = None,
    context: ReviewContext | None = None,
    min_words: int = 7,
    max_words: int = 14,
    lexical_repeat_limit: int = 3,
    changed_span_word_limit: int = 4,
) -> list[str]:
    """Return risk flags for one pair.

    The heuristic flags likely review priorities. It does not reject a pair by
    itself and does not attempt any LLM-based linguistic judgment.
    """

    if context is None:
        context = build_review_context(list(all_rows or [row]))

    reasons: list[str] = []
    correct_sentence = str(row.get("correct_sentence", "") or "")
    incorrect_sentence = str(row.get("incorrect_sentence", "") or "")
    changed_span = str(row.get("changed_span", "") or "")

    for label, sentence in (("correct_sentence", correct_sentence), ("incorrect_sentence", incorrect_sentence)):
        word_count = count_words(sentence)
        if word_count < min_words:
            reasons.append(f"{label} too short ({word_count})")
        if word_count > max_words:
            reasons.append(f"{label} too long ({word_count})")
        for flag in naturalness_flags(sentence):
            reasons.append(f"{label} mechanical phrase: {flag}")
        if has_long_service_word_run(sentence):
            reasons.append(f"{label} has too many service words in a row")
        if has_unusual_punctuation(sentence):
            reasons.append(f"{label} has unusual punctuation")

    if correct_sentence == incorrect_sentence:
        reasons.append("correct_sentence and incorrect_sentence are identical")
    elif is_almost_identical_without_substantive_change(correct_sentence, incorrect_sentence, changed_span):
        reasons.append("sentences are almost identical without a clear changed_span")

    lexical_signature = str(row.get("lexical_signature", "") or "")
    if lexical_signature and context.lexical_signature_counts[lexical_signature] > lexical_repeat_limit:
        reasons.append(
            f"lexical_signature repeated {context.lexical_signature_counts[lexical_signature]} times"
        )

    changed_span_words = changed_span_word_count(changed_span)
    if changed_span_words > changed_span_word_limit:
        reasons.append(f"changed_span too long ({changed_span_words} words)")

    lowered_correct = correct_sentence.casefold()
    for fragment in BLACKLIST_FRAGMENTS:
        if fragment in lowered_correct:
            reasons.append(f"blacklisted phrase: {fragment}")

    pair_key = (correct_sentence, incorrect_sentence)
    if context.sentence_pair_counts[pair_key] > 1:
        reasons.append(f"duplicate sentence pair ({context.sentence_pair_counts[pair_key]} copies)")

    return reasons


def suspicious_score(
    row: dict[str, object],
    *,
    context: ReviewContext | None = None,
    flag_reasons: list[str] | None = None,
) -> int:
    """Rank review priority using simple, explainable heuristics."""

    if context is None:
        context = build_review_context([row])
    if flag_reasons is None:
        flag_reasons = auto_review_flags(row, context=context)

    score = 0
    naturalness = parse_score(row.get("naturalness_score"))
    confidence = parse_score(row.get("normative_confidence"))
    if naturalness is not None:
        score += max(0, 4 - naturalness) * 3
    if confidence is not None:
        score += max(0, 4 - confidence) * 3
    if flag_reasons:
        score += 10 + len(flag_reasons)

    template_id = str(row.get("template_id", "") or "")
    lexical_signature = str(row.get("lexical_signature", "") or "")
    if template_id:
        template_target = max(3, ceil(sum(context.template_counts.values()) / max(1, len(context.template_counts))))
        score += max(0, context.template_counts[template_id] - template_target)
    if lexical_signature:
        score += max(0, context.lexical_signature_counts[lexical_signature] - 2) * 2

    changed_span_words = changed_span_word_count(str(row.get("changed_span", "") or ""))
    score += max(0, changed_span_words - 2)
    return score


def rank_suspicious_pairs(rows: list[dict[str, object]], *, limit: int = 20) -> list[dict[str, object]]:
    """Return the highest-priority pairs for manual inspection."""

    context = build_review_context(rows)
    ranked = []
    for row in rows:
        ranked.append(
            (
                suspicious_score(row, context=context),
                parse_bool(row.get("auto_flag", False)),
                str(row.get("pair_id", "")),
                row,
            )
        )
    ranked.sort(key=lambda item: (-item[0], not item[1], item[2]))
    return [row for score, _flag, _pair_id, row in ranked[:limit] if score > 0 or limit > 0]


def filter_approved_pairs(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows approved by a human reviewer."""

    return [row for row in rows if is_approved(row)]


def is_approved(row: dict[str, object]) -> bool:
    """Return whether a row is approved for the final clean corpus."""

    return parse_bool(row.get("manual_reviewed", False)) and normalize_status(row.get("manual_status", "")) == "approved"


def effective_review_status(row: dict[str, object]) -> str:
    """Return the status used in reports and finalization stats."""

    if is_approved(row):
        return "approved"
    if parse_bool(row.get("manual_reviewed", False)) and normalize_status(row.get("manual_status", "")) == "rejected":
        return "rejected"
    return "pending"


def build_review_stats(
    rows: list[dict[str, object]],
    *,
    approved_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build corpus quality and review progress statistics."""

    status_counts = Counter(effective_review_status(row) for row in rows)
    total = len(rows)
    approved_total = len(approved_rows) if approved_rows is not None else status_counts["approved"]
    rejected_total = status_counts["rejected"]
    pending_total = total - approved_total - rejected_total

    by_rule: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "approved": 0, "rejected": 0, "pending": 0})
    for row in rows:
        rule_id = str(row.get("rule_id", "") or "unknown")
        status = effective_review_status(row)
        by_rule[rule_id]["total"] += 1
        by_rule[rule_id][status] += 1

    by_difficulty = Counter(str(row.get("difficulty", "") or "unknown") for row in rows)
    duplicate_count = count_duplicate_sentence_pairs(rows)

    return {
        "total_pairs": total,
        "approved_pairs": approved_total,
        "rejected_pairs": rejected_total,
        "pending_pairs": pending_total,
        "status_counts": dict(status_counts),
        "by_rule": dict(sorted(by_rule.items())),
        "approved_by_rule": {
            rule_id: data["approved"] for rule_id, data in sorted(by_rule.items())
        },
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "average_naturalness_score": average_score(row.get("naturalness_score") for row in rows),
        "average_normative_confidence": average_score(row.get("normative_confidence") for row in rows),
        "duplicate_text_pairs": duplicate_count,
        "auto_flagged_pairs": sum(1 for row in rows if parse_bool(row.get("auto_flag", False))),
    }


def format_text_report(stats: dict[str, object]) -> str:
    """Render review stats as a readable plain-text report."""

    lines = [
        "Russian Minimal Pairs Review Report",
        "",
        f"Total pairs: {stats['total_pairs']}",
        f"Approved: {stats['approved_pairs']}",
        f"Rejected: {stats['rejected_pairs']}",
        f"Pending: {stats['pending_pairs']}",
        f"Auto-flagged pairs: {stats['auto_flagged_pairs']}",
        f"Duplicate text pairs: {stats['duplicate_text_pairs']}",
        f"Average naturalness_score: {format_optional_float(stats['average_naturalness_score'])}",
        f"Average normative_confidence: {format_optional_float(stats['average_normative_confidence'])}",
        "",
        "By difficulty:",
    ]
    for difficulty, count in dict(stats["by_difficulty"]).items():
        lines.append(f"  {difficulty}: {count}")
    lines.extend(["", "By rule:"])
    for rule_id, data in dict(stats["by_rule"]).items():
        lines.append(
            f"  {rule_id}: total={data['total']}, approved={data['approved']}, "
            f"rejected={data['rejected']}, pending={data['pending']}"
        )
    lines.extend(["", "Top suspicious pairs:"])
    for row in stats.get("top_suspicious_pairs", []):
        lines.append(
            f"  {row['pair_id']} [{row['rule_id']}], priority={row['review_priority']}, "
            f"auto_flag={row['auto_flag']}: {row['auto_flag_reason']}"
        )
    return "\n".join(lines) + "\n"


def build_review_context(rows: list[dict[str, object]]) -> ReviewContext:
    """Build repeated-frame counters for review heuristics."""

    return ReviewContext(
        template_counts=Counter(str(row.get("template_id", "") or "") for row in rows),
        lexical_signature_counts=Counter(str(row.get("lexical_signature", "") or "") for row in rows),
        sentence_pair_counts=Counter(
            (
                str(row.get("correct_sentence", "") or ""),
                str(row.get("incorrect_sentence", "") or ""),
            )
            for row in rows
        ),
    )


def clean_pair_row(row: dict[str, object]) -> dict[str, object]:
    """Return only experiment-ready pair-level fields."""

    return {field: row.get(field, "") for field in CLEAN_PAIR_FIELDNAMES}


def make_clean_item_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return item-level rows for approved pairs only."""

    items: list[dict[str, object]] = []
    for row in rows:
        pair_id = str(row.get("pair_id", "") or "")
        common = {
            "pair_id": pair_id,
            "rule_id": row.get("rule_id", ""),
            "rule_name": row.get("rule_name", ""),
            "error_type": row.get("error_type", ""),
            "difficulty": row.get("difficulty", ""),
            "template_id": row.get("template_id", ""),
            "lexical_signature": row.get("lexical_signature", ""),
        }
        correct_sentence = str(row.get("correct_sentence", "") or "")
        incorrect_sentence = str(row.get("incorrect_sentence", "") or "")
        items.append(
            {
                "item_id": f"{pair_id}_correct",
                **common,
                "variant": "correct",
                "sentence": correct_sentence,
                "gold_label": 1,
                "gold_correction": correct_sentence,
                "changed_span": "",
                "expected_action": "accept",
            }
        )
        items.append(
            {
                "item_id": f"{pair_id}_incorrect",
                **common,
                "variant": "incorrect",
                "sentence": incorrect_sentence,
                "gold_label": 0,
                "gold_correction": correct_sentence,
                "changed_span": row.get("changed_span", ""),
                "expected_action": "correct",
            }
        )
    return items


def count_duplicate_sentence_pairs(rows: Iterable[dict[str, object]]) -> int:
    """Count extra copies of identical correct/incorrect sentence pairs."""

    counts = Counter(
        (
            str(row.get("correct_sentence", "") or ""),
            str(row.get("incorrect_sentence", "") or ""),
        )
        for row in rows
    )
    return sum(count - 1 for count in counts.values() if count > 1)


def changed_span_word_count(changed_span: str) -> int:
    """Count words in the actual changed fragments, not metadata labels."""

    fragments = re.findall(r"(?:correct|incorrect)='([^']*)'", changed_span)
    if not fragments:
        return count_words(changed_span)
    return max(count_words(fragment) for fragment in fragments)


def has_long_service_word_run(sentence: str, *, limit: int = 4) -> bool:
    """Detect long runs of function words that often sound mechanical."""

    run = 0
    for token in normalize_tokens(sentence):
        if token in SERVICE_WORDS:
            run += 1
            if run >= limit:
                return True
        else:
            run = 0
    return False


def has_unusual_punctuation(sentence: str) -> bool:
    """Return whether punctuation looks unusual for this controlled corpus."""

    return bool(UNUSUAL_PUNCTUATION_RE.search(sentence) or REPEATED_PUNCTUATION_RE.search(sentence))


def is_almost_identical_without_substantive_change(
    correct_sentence: str,
    incorrect_sentence: str,
    changed_span: str,
    *,
    threshold: float = 0.98,
) -> bool:
    """Flag pairs whose difference is not clearly represented by metadata."""

    if changed_span.strip():
        return False
    diff = single_contiguous_diff(correct_sentence, incorrect_sentence)
    if diff is None:
        return True
    correct_diff, incorrect_diff = diff
    punctuation_only = not count_words(correct_diff) and not count_words(incorrect_diff)
    similarity = SequenceMatcher(a=correct_sentence, b=incorrect_sentence).ratio()
    return similarity >= threshold and not punctuation_only


def parse_bool(value: object) -> bool:
    """Parse common truthy strings used in CSV review tables."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().casefold() in {"true", "1", "yes", "y", "да", "истина"}


def normalize_status(value: object) -> str:
    """Normalize manual status values, including legacy names."""

    status = str(value or "pending").strip().casefold()
    aliases = {
        "accepted": "approved",
        "accept": "approved",
        "ok": "approved",
        "approve": "approved",
        "reject": "rejected",
        "revise": "pending",
        "": "pending",
    }
    status = aliases.get(status, status)
    return status if status in MANUAL_STATUSES else "pending"


def parse_int(value: object, *, default: int = 0) -> int:
    """Parse an integer with a safe default."""

    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def parse_score(value: object) -> float | None:
    """Parse a 1-5 review score or return ``None`` when missing."""

    if value in (None, ""):
        return None
    try:
        score = float(str(value).replace(",", "."))
    except ValueError:
        return None
    if 1 <= score <= 5:
        return score
    return None


def average_score(values: Iterable[object]) -> float | None:
    """Average non-empty 1-5 scores."""

    scores = [score for score in (parse_score(value) for value in values) if score is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def format_optional_float(value: object) -> str:
    """Format optional numeric values for text reports."""

    if value is None:
        return "n/a"
    return str(value)
