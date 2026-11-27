"""Metadata for supported Russian minimal-pair error rules.

The module intentionally contains only stable rule descriptions. Sentence
patterns live in :mod:`templates`, and lexical material lives in
:mod:`lexicon`, so new rules can be added without changing the generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Rule:
    """A linguistic error class used to generate minimal pairs."""

    rule_id: str
    rule_name: str
    error_type: str
    difficulty: str
    description: str


RULES: tuple[Rule, ...] = (
    Rule(
        rule_id="subject_verb_agreement",
        rule_name="Согласование подлежащего и сказуемого",
        error_type="agreement",
        difficulty="easy",
        description="Нарушение согласования числа или рода между подлежащим и сказуемым.",
    ),
    Rule(
        rule_id="adjective_noun_agreement",
        rule_name="Согласование прилагательного и существительного",
        error_type="agreement",
        difficulty="easy",
        description="Нарушение согласования рода, числа или падежа прилагательного с существительным.",
    ),
    Rule(
        rule_id="case_government",
        rule_name="Падежное управление",
        error_type="government",
        difficulty="medium",
        description="Неверная падежная форма зависимого слова при глаголе или предлоге.",
    ),
    Rule(
        rule_id="verb_form",
        rule_name="Форма глагола",
        error_type="morphology",
        difficulty="medium",
        description="Неверная форма смыслового глагола после модального или фазового слова.",
    ),
    Rule(
        rule_id="tsya_tsa",
        rule_name="-тся / -ться",
        error_type="orthography",
        difficulty="easy",
        description="Смешение личной формы на -тся и инфинитива на -ться.",
    ),
    Rule(
        rule_id="ne_ni",
        rule_name="не / ни",
        error_type="orthography",
        difficulty="easy",
        description="Замена отрицательной частицы «не» на «ни» в простом отрицании.",
    ),
    Rule(
        rule_id="n_nn",
        rule_name="н / нн",
        error_type="orthography",
        difficulty="medium",
        description="Ошибочное количество букв н в общеупотребительных прилагательных.",
    ),
    Rule(
        rule_id="fused_separate_spelling",
        rule_name="Слитное / раздельное написание",
        error_type="orthography",
        difficulty="medium",
        description="Смешение слитного и раздельного написания устойчивых слов и сочетаний.",
    ),
    Rule(
        rule_id="complex_sentence_comma",
        rule_name="Запятая в сложном предложении",
        error_type="punctuation",
        difficulty="medium",
        description="Пропуск запятой между главным и придаточным предложением.",
    ),
    Rule(
        rule_id="introductory_comma",
        rule_name="Запятая при вводных словах",
        error_type="punctuation",
        difficulty="easy",
        description="Пропуск запятой после вводного слова в начале предложения.",
    ),
    Rule(
        rule_id="participial_phrase_comma",
        rule_name="Запятая при причастном обороте",
        error_type="punctuation",
        difficulty="medium",
        description="Пропуск запятой перед причастным оборотом после определяемого слова.",
    ),
    Rule(
        rule_id="adverbial_participle_comma",
        rule_name="Запятая при деепричастном обороте",
        error_type="punctuation",
        difficulty="medium",
        description="Пропуск запятой после деепричастного оборота в начале предложения.",
    ),
)

RULES_BY_ID: dict[str, Rule] = {rule.rule_id: rule for rule in RULES}


def list_rule_ids() -> list[str]:
    """Return rule identifiers in the canonical generation order."""

    return [rule.rule_id for rule in RULES]


def get_rule(rule_id: str) -> Rule:
    """Return metadata for a rule id.

    Raises:
        KeyError: If the rule id is not registered.
    """

    return RULES_BY_ID[rule_id]


def normalize_rule_ids(rule_ids: Iterable[str] | None) -> list[str]:
    """Normalize CLI rule ids and validate that they exist."""

    if not rule_ids:
        return list_rule_ids()

    normalized: list[str] = []
    for raw_id in rule_ids:
        for part in raw_id.split(","):
            rule_id = part.strip()
            if rule_id:
                normalized.append(rule_id)

    unknown = sorted(set(normalized) - set(RULES_BY_ID))
    if unknown:
        known = ", ".join(list_rule_ids())
        raise ValueError(f"Unknown rule id(s): {', '.join(unknown)}. Known rules: {known}")

    return normalized
