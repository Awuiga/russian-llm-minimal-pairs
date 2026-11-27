# API Experiment Schema

Этот файл фиксирует формат данных для этапа API-эксперимента. Скрипт не использует LLM для генерации или постобработки корпуса: модели вызываются только как объект исследования.

## Input Item

Один объект из `final_items.csv` или `final_items.json`:

| Поле | Тип | Описание |
| --- | --- | --- |
| `item_id` | string | Уникальный идентификатор item-level задания. |
| `pair_id` | string | Идентификатор исходной минимальной пары. |
| `variant` | string | `correct` или `incorrect`. |
| `sentence` | string | Предложение, которое подается модели. |
| `gold_label` | integer/string | `1` для корректного предложения, `0` для предложения с ошибкой. |
| `gold_correction` | string | Ожидаемая нормативная версия предложения. |
| `expected_action` | string | `no_change` для корректного варианта, `correct` для ошибочного. |
| `rule_id` | string | Идентификатор правила. |
| `rule_name` | string | Человекочитаемое название правила. |
| `difficulty` | string | Уровень сложности примера. |

## Raw Request JSONL

Файл `raw_requests.jsonl` содержит один JSON-объект на каждую комбинацию `(provider, model, prompt_id, item_id)`:

| Поле | Описание |
| --- | --- |
| `run_id` | Идентификатор прогона. |
| `timestamp` | UTC timestamp создания запроса. |
| `provider` | `openai`, `anthropic` или `deepseek`. |
| `model` | Модель из CLI или matrix config. |
| `prompt_id` | Идентификатор семейства промпта. |
| `item_id`, `pair_id`, `variant` | Метаданные задания. |
| `system_prompt` | Общая инструкция. |
| `user_prompt` | Текст пользовательского запроса с предложением. |
| `request` | Полный payload провайдера с замаскированными секретами в headers. |

## Raw Response JSONL

Файл `raw_responses.jsonl` сохраняет транспортный результат без потери диагностической информации:

| Поле | Описание |
| --- | --- |
| `run_id`, `provider`, `model`, `prompt_id`, `item_id` | Ключи эксперимента. |
| `request_status` | `succeeded`, `failed` или `dry_run`. |
| `status_code` | HTTP status, если был ответ сервера. |
| `error_text` | Текст ошибки после retries, если запрос не удался. |
| `retry_count` | Сколько повторов потребовалось до результата. |
| `latency_ms` | Задержка последней попытки в миллисекундах. |
| `input_tokens`, `output_tokens` | Usage, если провайдер его вернул. |
| `provider_response` | JSON-ответ провайдера как есть. |
| `raw_text` | Извлеченный текст ответа модели. |

## Normalized Result

Файлы `normalized_results.csv` и `normalized_results.jsonl` используют единый формат:

| Поле | Описание |
| --- | --- |
| `run_id` | Идентификатор прогона. |
| `provider` | Провайдер API. |
| `model` | Модель. |
| `prompt_id` | Семейство промпта. |
| `item_id`, `pair_id`, `variant` | Метаданные задания. |
| `sentence` | Исходное предложение. |
| `gold_label` | Эталон: `1` корректно, `0` есть ошибка. |
| `gold_correction` | Эталонная коррекция. |
| `expected_action` | Ожидаемое действие. |
| `rule_id`, `rule_name`, `difficulty` | Лингвистические метаданные. |
| `has_error_pred` | Нормализованный прогноз модели: `true`, `false` или пусто. |
| `error_type_pred` | Нормализованный тип ошибки из ответа модели. |
| `correction_pred` | Нормализованная коррекция. |
| `parse_status` | `ok`, `partial`, `failed` или `skipped` для dry-run. |
| `request_status` | `succeeded`, `failed` или `dry_run`. |
| `raw_text` | Текст ответа модели. |
| `raw_json` | JSON-объект, извлеченный из текста модели, если он был найден. |
| `input_tokens`, `output_tokens` | Token usage, если доступен. |
| `latency_ms` | Задержка запроса. |
| `timestamp` | UTC timestamp нормализации результата. |

## Resume Key

Для продолжения прерванного запуска используется ключ:

```text
(provider, model, prompt_id, item_id)
```

Если ключ уже есть в `normalized_results.csv`, режим `--resume` пропускает эту комбинацию.

## Metric Outputs

Команда `--api-mode metrics` создает:

- `metrics_summary.csv`
- `metrics_by_rule.csv`
- `metrics_by_prompt.csv`
- `metrics_by_model.csv`
- `metrics_by_provider.csv`
- `metrics_by_difficulty.csv`
- `metrics_by_variant.csv`

Основные показатели:

- `detection_accuracy` — доля правильных решений о наличии ошибки по всем item.
- `correction_accuracy` — доля ошибочных item, где модель одновременно обнаружила ошибку и дала точное исправление.
- `false_positive_rate` — доля correct-вариантов, ошибочно помеченных как содержащие ошибку.
- `false_negative_rate` — доля incorrect-вариантов, ошибочно помеченных как корректные.
- `exact_match` — точное совпадение `correction_pred` с `gold_correction` только для incorrect-вариантов.
- `exact_match_all` — точное совпадение `correction_pred` с `gold_correction` по всем item, включая correct-контроли.
- `exact_match_correct` — точное сохранение correct-вариантов без изменения.
