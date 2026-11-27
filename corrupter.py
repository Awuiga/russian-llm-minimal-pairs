"""Create controlled erroneous versions of generated Russian sentences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from templates import SentenceTemplate


@dataclass(frozen=True)
class CorruptionResult:
    """The incorrect sentence and metadata for the changed fragment."""

    incorrect_sentence: str
    gold_correction: str
    changed_span: str


def create_corruption(template: SentenceTemplate, fields: Mapping[str, str]) -> CorruptionResult:
    """Render the erroneous sentence for a template and lexical choice.

    The function does not guess a new error. It only applies the one
    transformation encoded in the paired correct/incorrect template.
    """

    incorrect_sentence = template.render_incorrect(fields)
    correct_span = template.render_correct_span(fields)
    incorrect_span = template.render_incorrect_span(fields)
    changed_span = format_changed_span(correct_span=correct_span, incorrect_span=incorrect_span)
    return CorruptionResult(
        incorrect_sentence=incorrect_sentence,
        gold_correction=correct_span,
        changed_span=changed_span,
    )


def format_changed_span(*, correct_span: str, incorrect_span: str) -> str:
    """Return a compact scalar description suitable for CSV and JSON."""

    return f"correct={correct_span!r}; incorrect={incorrect_span!r}"
