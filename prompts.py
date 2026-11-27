"""Prompt families for the grammar-detection API experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


JSON_OUTPUT_CONTRACT = """Верни только валидный JSON без Markdown и без пояснений:
{
  "has_error": true,
  "error_type": "agreement",
  "correction": "исправленное предложение"
}

Правила полей:
- has_error: true, если в предложении есть ошибка; иначе false.
- error_type: короткий тип ошибки на английском или "none", если ошибки нет.
- correction: минимально исправленное предложение; если ошибки нет, повтори исходное предложение без изменений.
"""


@dataclass(frozen=True)
class PromptTemplate:
    """One prompt family used in the experiment."""

    prompt_id: str
    system_prompt: str
    user_template: str

    def render(self, item: Mapping[str, object]) -> "RenderedPrompt":
        """Render the prompt for one item row."""

        sentence = str(item.get("sentence", ""))
        return RenderedPrompt(
            prompt_id=self.prompt_id,
            system_prompt=self.system_prompt,
            user_prompt=self.user_template.format(sentence=sentence),
        )


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered system/developer instruction and user input."""

    prompt_id: str
    system_prompt: str
    user_prompt: str


PROMPTS: dict[str, PromptTemplate] = {
    "direct_zero_shot": PromptTemplate(
        prompt_id="direct_zero_shot",
        system_prompt=(
            "Ты проверяешь одно русское предложение на наличие одной нормативной ошибки. "
            "Не переписывай предложение без необходимости. " + JSON_OUTPUT_CONTRACT
        ),
        user_template="Предложение: {sentence}",
    ),
    "role_teacher": PromptTemplate(
        prompt_id="role_teacher",
        system_prompt=(
            "Ты преподаватель русского языка и проверяешь короткие предложения для учебного корпуса. "
            "Твоя задача — определить, есть ли ошибка, и дать минимальную нормативную правку. "
            + JSON_OUTPUT_CONTRACT
        ),
        user_template="Проверь предложение как преподаватель русского языка: {sentence}",
    ),
    "minimal_edit": PromptTemplate(
        prompt_id="minimal_edit",
        system_prompt=(
            "Исправляй только минимальный ошибочный фрагмент. Не меняй стиль, порядок слов и лексику, "
            "если это не требуется для исправления. " + JSON_OUTPUT_CONTRACT
        ),
        user_template="Найди ошибку, если она есть, и сделай минимальное исправление: {sentence}",
    ),
    "step_by_step": PromptTemplate(
        prompt_id="step_by_step",
        system_prompt=(
            "Сначала внутренне реши, есть ли ошибка, затем внутренне определи тип ошибки, затем дай "
            "исправленное предложение. В ответе не показывай рассуждения, верни только итоговый JSON. "
            + JSON_OUTPUT_CONTRACT
        ),
        user_template="Проанализируй предложение по шагам и верни итог в JSON: {sentence}",
    ),
    "grammar_prompt": PromptTemplate(
        prompt_id="grammar_prompt",
        system_prompt=(
            "Проверь соблюдение норм русского языка, включая согласование, управление, форму глагола, "
            "орфографию, слитное/раздельное написание и пунктуацию. " + JSON_OUTPUT_CONTRACT
        ),
        user_template="Предложение для нормативной проверки: {sentence}",
    ),
}


def list_prompt_ids() -> list[str]:
    """Return registered prompt ids."""

    return list(PROMPTS)


def render_prompt(prompt_id: str, item: Mapping[str, object]) -> RenderedPrompt:
    """Render one prompt by id."""

    try:
        prompt = PROMPTS[prompt_id]
    except KeyError as exc:
        known = ", ".join(list_prompt_ids())
        raise ValueError(f"Unknown prompt_id {prompt_id!r}. Known prompts: {known}") from exc
    return prompt.render(item)
