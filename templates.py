"""Sentence templates for deterministic Russian minimal-pair generation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class SentenceTemplate:
    """A correct/incorrect sentence pattern for one rule.

    ``correct_pattern`` and ``incorrect_pattern`` should differ in one
    controlled fragment. ``correct_span_pattern`` and ``incorrect_span_pattern``
    provide human-readable metadata for that fragment.
    """

    template_id: str
    rule_id: str
    correct_pattern: str
    incorrect_pattern: str
    correct_span_pattern: str
    incorrect_span_pattern: str
    comment: str

    def render_correct(self, fields: Mapping[str, str]) -> str:
        """Render a grammatical sentence."""

        return normalize_sentence(self.correct_pattern.format(**fields))

    def render_incorrect(self, fields: Mapping[str, str]) -> str:
        """Render a sentence with one controlled error."""

        return normalize_sentence(self.incorrect_pattern.format(**fields))

    def render_correct_span(self, fields: Mapping[str, str]) -> str:
        """Render the normative replacement fragment."""

        return self.correct_span_pattern.format(**fields)

    def render_incorrect_span(self, fields: Mapping[str, str]) -> str:
        """Render the erroneous fragment."""

        return self.incorrect_span_pattern.format(**fields)


def normalize_sentence(sentence: str) -> str:
    """Collapse extra whitespace and capitalize the sentence start."""

    normalized = " ".join(sentence.split())
    if not normalized:
        return normalized
    return normalized[0].upper() + normalized[1:]


def target_template(
    template_id: str,
    rule_id: str,
    correct_pattern: str,
    incorrect_pattern: str,
    comment: str,
) -> SentenceTemplate:
    """Create a template where the changed fragment is ``correct/incorrect``."""

    return SentenceTemplate(
        template_id=template_id,
        rule_id=rule_id,
        correct_pattern=correct_pattern,
        incorrect_pattern=incorrect_pattern,
        correct_span_pattern="{correct}",
        incorrect_span_pattern="{incorrect}",
        comment=comment,
    )


def comma_template(
    template_id: str,
    rule_id: str,
    correct_pattern: str,
    incorrect_pattern: str,
    comment: str,
) -> SentenceTemplate:
    """Create a punctuation template for a missing comma."""

    return SentenceTemplate(
        template_id=template_id,
        rule_id=rule_id,
        correct_pattern=correct_pattern,
        incorrect_pattern=incorrect_pattern,
        correct_span_pattern=",",
        incorrect_span_pattern="",
        comment=comment,
    )


TEMPLATES_BY_RULE: Mapping[str, tuple[SentenceTemplate, ...]] = MappingProxyType(
    {
        "subject_verb_agreement": (
            target_template(
                "subject_verb_1",
                "subject_verb_agreement",
                "{subject} {adverb} {correct} {obj} {time}.",
                "{subject} {adverb} {incorrect} {obj} {time}.",
                "Замена формы сказуемого при неизменном подлежащем.",
            ),
            target_template(
                "subject_verb_2",
                "subject_verb_agreement",
                "{subject} {correct} {obj} {place} {time}.",
                "{subject} {incorrect} {obj} {place} {time}.",
                "Сказуемое отделено от начала предложения обстоятельством времени.",
            ),
            target_template(
                "subject_verb_3",
                "subject_verb_agreement",
                "{subject} {time} {correct} {obj} {place}.",
                "{subject} {time} {incorrect} {obj} {place}.",
                "Нейтральный порядок слов с обстоятельством в начале.",
            ),
        ),
        "adjective_noun_agreement": (
            target_template(
                "adjective_noun_1",
                "adjective_noun_agreement",
                "{correct} {noun} {verb} {place} {time}.",
                "{incorrect} {noun} {verb} {place} {time}.",
                "Прилагательное перед существительным.",
            ),
            target_template(
                "adjective_noun_2",
                "adjective_noun_agreement",
                "{time_start} {verb} {correct} {noun} {place}.",
                "{time_start} {verb} {incorrect} {noun} {place}.",
                "Прилагательное в середине предложения.",
            ),
            target_template(
                "adjective_noun_3",
                "adjective_noun_agreement",
                "{time_start} {correct} {noun} {verb} {place}.",
                "{time_start} {incorrect} {noun} {verb} {place}.",
                "Согласование после обстоятельства времени.",
            ),
        ),
        "case_government": (
            target_template(
                "case_government_1",
                "case_government",
                "{subject} {adverb} {verb} {correct} {time}.",
                "{subject} {adverb} {verb} {incorrect} {time}.",
                "Зависимое слово стоит сразу после управляющего глагола.",
            ),
            target_template(
                "case_government_2",
                "case_government",
                "{time_start} {subject} {verb} {correct} {place}.",
                "{time_start} {subject} {verb} {incorrect} {place}.",
                "Обстоятельство времени вынесено в начало.",
            ),
            target_template(
                "case_government_3",
                "case_government",
                "{subject} {verb} {correct} {place}.",
                "{subject} {verb} {incorrect} {place}.",
                "Между подлежащим и сказуемым есть обстоятельство.",
            ),
        ),
        "verb_form": (
            target_template(
                "verb_form_1",
                "verb_form",
                "{subject} {helper} {correct} {obj} {time}.",
                "{subject} {helper} {incorrect} {obj} {time}.",
                "Неверная форма смыслового глагола после служебного слова.",
            ),
            target_template(
                "verb_form_2",
                "verb_form",
                "{time_start} {subject} {helper} {correct} {obj} {place}.",
                "{time_start} {subject} {helper} {incorrect} {obj} {place}.",
                "Форма глагола проверяется в расширенном контексте.",
            ),
            target_template(
                "verb_form_3",
                "verb_form",
                "{subject} {helper} {adverb} {correct} {obj}.",
                "{subject} {helper} {adverb} {incorrect} {obj}.",
                "Добавлено обстоятельство образа действия.",
            ),
        ),
        "tsya_tsa": (
            target_template(
                "tsya_tsa_1",
                "tsya_tsa",
                "{actor} {correct} {context} {time}.",
                "{actor} {incorrect} {context} {time}.",
                "Контраст личной формы и инфинитива в коротком предложении.",
            ),
            target_template(
                "tsya_tsa_2",
                "tsya_tsa",
                "{time_start} {actor} {correct} {context} {place}.",
                "{time_start} {actor} {incorrect} {context} {place}.",
                "Обстоятельство времени вынесено в начало.",
            ),
            target_template(
                "tsya_tsa_3",
                "tsya_tsa",
                "{actor} {adverb} {correct} {context} {time}.",
                "{actor} {adverb} {incorrect} {context} {time}.",
                "Слово-мишень отделено обстоятельством.",
            ),
        ),
        "ne_ni": (
            target_template(
                "ne_ni_1",
                "ne_ni",
                "{prefix} {correct} {suffix}.",
                "{prefix} {incorrect} {suffix}.",
                "Замена частицы в простом отрицательном предложении.",
            ),
            target_template(
                "ne_ni_2",
                "ne_ni",
                "{time_start} {prefix_lc} {correct} {suffix}.",
                "{time_start} {prefix_lc} {incorrect} {suffix}.",
                "Отрицание после обстоятельства времени.",
            ),
            target_template(
                "ne_ni_3",
                "ne_ni",
                "{prefix} {adverb} {correct} {suffix}.",
                "{prefix} {adverb} {incorrect} {suffix}.",
                "Частица отделена от подлежащего коротким обстоятельством.",
            ),
        ),
        "n_nn": (
            target_template(
                "n_nn_1",
                "n_nn",
                "{correct} {noun} {predicate} {place} {time}.",
                "{incorrect} {noun} {predicate} {place} {time}.",
                "Количество н меняется в прилагательном перед существительным.",
            ),
            target_template(
                "n_nn_2",
                "n_nn",
                "{time_start} {correct} {noun} {predicate} {place}.",
                "{time_start} {incorrect} {noun} {predicate} {place}.",
                "Орфограмма находится после обстоятельства времени.",
            ),
            target_template(
                "n_nn_3",
                "n_nn",
                "В комнате {predicate} {correct} {noun} {place_tail}.",
                "В комнате {predicate} {incorrect} {noun} {place_tail}.",
                "Прилагательное стоит в середине короткого предложения.",
            ),
        ),
        "fused_separate_spelling": (
            target_template(
                "fused_separate_1",
                "fused_separate_spelling",
                "{prefix} {correct} {suffix}.",
                "{prefix} {incorrect} {suffix}.",
                "Слитное или раздельное написание в середине предложения.",
            ),
            target_template(
                "fused_separate_2",
                "fused_separate_spelling",
                "{time_start} {prefix_lc} {correct} {suffix}.",
                "{time_start} {prefix_lc} {incorrect} {suffix}.",
                "Та же орфограмма после обстоятельства времени.",
            ),
            target_template(
                "fused_separate_3",
                "fused_separate_spelling",
                "Кроме того, {prefix_lc} {correct} {suffix}.",
                "Кроме того, {prefix_lc} {incorrect} {suffix}.",
                "Добавлено нейтральное обстоятельство образа действия.",
            ),
        ),
        "complex_sentence_comma": (
            comma_template(
                "complex_comma_1",
                "complex_sentence_comma",
                "{main}, {connector} {subordinate}.",
                "{main} {connector} {subordinate}.",
                "Придаточная часть следует за главной.",
            ),
            comma_template(
                "complex_comma_2",
                "complex_sentence_comma",
                "{time_start} {main_lc}, {connector} {subordinate}.",
                "{time_start} {main_lc} {connector} {subordinate}.",
                "Сложное предложение начинается с обстоятельства времени.",
            ),
            comma_template(
                "complex_comma_3",
                "complex_sentence_comma",
                "{main}, {connector} {subordinate} {time_tail}.",
                "{main} {connector} {subordinate} {time_tail}.",
                "После придаточной части добавлено короткое обстоятельство.",
            ),
        ),
        "introductory_comma": (
            comma_template(
                "introductory_comma_1",
                "introductory_comma",
                "{intro}, {body}.",
                "{intro} {body}.",
                "Вводное слово стоит в начале предложения.",
            ),
            comma_template(
                "introductory_comma_2",
                "introductory_comma",
                "{intro}, {body} {time_tail}.",
                "{intro} {body} {time_tail}.",
                "После основы предложения добавлено время.",
            ),
            comma_template(
                "introductory_comma_3",
                "introductory_comma",
                "{intro}, {body} {place_tail}.",
                "{intro} {body} {place_tail}.",
                "После основы предложения добавлено место.",
            ),
        ),
        "participial_phrase_comma": (
            comma_template(
                "participial_comma_1",
                "participial_phrase_comma",
                "{main}, {participle}.",
                "{main} {participle}.",
                "Причастный оборот стоит после определяемого слова в конце предложения.",
            ),
            comma_template(
                "participial_comma_2",
                "participial_phrase_comma",
                "{time_start} {main_lc}, {participle}.",
                "{time_start} {main_lc} {participle}.",
                "Перед главным фрагментом добавлено обстоятельство времени.",
            ),
            comma_template(
                "participial_comma_3",
                "participial_phrase_comma",
                "{main}, {participle} {time_tail}.",
                "{main} {participle} {time_tail}.",
                "Оборот расширен коротким обстоятельством.",
            ),
        ),
        "adverbial_participle_comma": (
            comma_template(
                "adverbial_participle_comma_1",
                "adverbial_participle_comma",
                "{gerund}, {main}.",
                "{gerund} {main}.",
                "Деепричастный оборот стоит в начале предложения.",
            ),
            comma_template(
                "adverbial_participle_comma_2",
                "adverbial_participle_comma",
                "{gerund}, {main} {time_tail}.",
                "{gerund} {main} {time_tail}.",
                "После главной части добавлено обстоятельство времени.",
            ),
            comma_template(
                "adverbial_participle_comma_3",
                "adverbial_participle_comma",
                "{time_start} {gerund_lc}, {main}.",
                "{time_start} {gerund_lc} {main}.",
                "Перед деепричастным оборотом добавлено обстоятельство.",
            ),
        ),
    }
)


def get_templates(rule_id: str) -> tuple[SentenceTemplate, ...]:
    """Return all sentence templates for a rule id."""

    return TEMPLATES_BY_RULE[rule_id]
