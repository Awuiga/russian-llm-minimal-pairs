# Генератор русских минимальных пар

Детерминированный шаблонно-словарный генератор корпуса минимальных пар для исследования того, как разные промпты влияют на способность LLM находить и исправлять нарушения правил русского языка.

Скрипт не использует LLM для генерации предложений. Все предложения собираются из заранее заданных шаблонов и словарей, поэтому корпус воспроизводим при одном и том же `--seed`.

## Установка

Нужен Python 3.11+. Внешние зависимости не требуются.

```bash
python --version
```

## Генерация корпуса

Preview-набор:

```bash
python main.py --mode preview --num-per-rule 3 --output-dir ./output --output-name demo_pairs --format csv json --seed 42
```

Полный набор по 20 пар на правило:

```bash
python main.py --mode full --output-dir ./output --format csv json --seed 42
```

По умолчанию рядом с pair-файлами создаются API-файлы `*_items.csv/json`, где correct и incorrect варианты идут отдельными строками. Это нужно, чтобы измерять не только исправление ошибок, но и ложные срабатывания на правильных предложениях.

Отключить item-level экспорт:

```bash
python main.py --mode preview --no-items
```

Выбор конкретных правил:

```bash
python main.py --num-per-rule 10 --rules tsya_tsa ne_ni n_nn --output-dir ./output --format csv json
```

## Review Workflow

После генерации полного корпуса подготовьте таблицу для ручной вычитки:

```bash
python main.py --review-mode prepare-review --input ./output/full_pairs.csv --output-dir ./review
```

Будут созданы:

- `review/review_pairs.csv`
- `review/review_pairs.json`
- `review/suspicious_pairs.csv`
- `review/suspicious_pairs.json`

Файл `review_pairs.csv` удобно открыть в таблицах и заполнить:

- `manual_reviewed`
- `manual_status`
- `manual_comment`
- `naturalness_score`
- `normative_confidence`

После ручной вычитки соберите финальный clean-корпус:

```bash
python main.py --review-mode finalize-review --input ./review/review_pairs.csv --output-dir ./final
```

Будут созданы:

- `final/final_pairs.csv`
- `final/final_pairs.json`
- `final/final_items.csv`
- `final/final_items.json`

Финальный набор содержит только строки, где `manual_reviewed=true` и `manual_status=approved`. В final-файлах нет ручных и автоматических review-полей, только чистые экспериментальные поля и ключевые метаданные.

Сводный отчет по review-таблице:

```bash
python main.py --review-mode review-report --input ./review/review_pairs.csv --output-dir ./review
```

Будут созданы:

- `review/review_report.txt`
- `review/review_report.json`
- обновленные `review/suspicious_pairs.csv/json`

## Критерии ручной проверки

`naturalness_score` оценивает естественность корректного предложения:

- `1` — предложение звучит неестественно, механически, искусственно;
- `2` — предложение допустимо, но шероховатое;
- `3` — нормальное нейтральное предложение;
- `4` — естественное и хорошее;
- `5` — очень естественное и полностью убедительное.

`normative_confidence` оценивает нормативную однозначность ошибки:

- `1` — случай спорный или сомнительный;
- `2` — ошибка неочевидна;
- `3` — в целом допустимо для корпуса, но лучше проверить;
- `4` — ошибка ясная и нормативно уверенная;
- `5` — ошибка абсолютно однозначная.

`manual_status`:

- `approved` — пример можно включать в clean-корпус;
- `rejected` — пример нужно исключить;
- `pending` — пример еще не проверен или требует решения.

`manual_comment` — короткое пояснение, особенно если пример отклонен или требует доработки.

Практическое правило: одобрять пары с естественным correct-предложением, одной однозначной ошибкой в incorrect-предложении и понятной gold-коррекцией. Отклонять спорные, слишком искусственные, слишком похожие на другие или методологически шумные пары.

## Полуавтоматические флаги

`review.py` добавляет:

- `auto_flag`
- `auto_flag_reason`
- `review_priority`

Флаги ставятся, если предложение слишком короткое или длинное, есть механические фразы, слишком длинный `changed_span`, необычная пунктуация, дубли текста, слишком частый повтор `lexical_signature`, длинные цепочки служебных слов или фрагменты из blacklist.

`review_priority` ранжирует пары для ручной проверки. Чем выше число, тем раньше пару стоит посмотреть. Файл `suspicious_pairs.csv/json` содержит топ подозрительных примеров.

## Архитектура

- `main.py` — CLI для генерации и review-команд.
- `rules.py` — метаданные правил.
- `lexicon.py` — расширяемые словари и rule-specific подстановки.
- `templates.py` — шаблоны корректных и ошибочных предложений.
- `generator.py` — сборка минимальных пар и API items.
- `corrupter.py` — применение одной контролируемой ошибки.
- `validator.py` — автоматические проверки качества генерации.
- `exporter.py` — CSV/JSON экспорт pair-level и item-level файлов.
- `review.py` — подготовка review-таблицы, auto flags, suspicious ranking, final clean export, review report.
- `schema.md` — описание полей и критериев ручной вычитки.
- `tests/` — unit-тесты на `unittest`.

## Поддерживаемые классы ошибок

1. Согласование подлежащего и сказуемого.
2. Согласование прилагательного и существительного.
3. Падежное управление.
4. Форма глагола.
5. `-тся / -ться`.
6. `не / ни`.
7. `н / нн`.
8. Слитное / раздельное написание.
9. Запятая в сложном предложении.
10. Запятая при вводных словах.
11. Запятая при причастном обороте.
12. Запятая при деепричастном обороте.

## Как расширять

Новое правило:

1. Добавьте `Rule` в `rules.py`.
2. Добавьте `choice(...)` для нового `rule_id` в `lexicon.py`.
3. Добавьте шаблоны для этого `rule_id` в `templates.py`.
4. Запустите тесты и preview-generation.

Новый шаблон добавляется через `target_template(...)` или `comma_template(...)` в `templates.py`.

Новая словарная группа добавляется в `lexicon.py`; каждый `choice(...)` должен содержать все поля, используемые шаблоном.

## API-эксперимент

После ручной вычитки используйте `final/final_items.csv` или `final/final_items.json` как вход для API-прогона. На этом этапе каждое предложение проверяется как самостоятельный тестовый объект: и корректные варианты, и варианты с ошибками. Это нужно, чтобы измерять не только нахождение ошибок, но и ложные срабатывания.

Ключи API читаются только из переменных окружения:

```bash
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:DEEPSEEK_API_KEY="..."
```

Если Anthropic-совместимый провайдер работает через gateway, можно задать base URL:

```bash
$env:ANTHROPIC_BASE_URL="https://api.oneprovider.dev"
```

Значение может быть базовым хостом, путем `/v1` или полным endpoint `/v1/messages`; адаптер приведет его к endpoint Messages API.

Один провайдер и одна модель:

```bash
python main.py --api-mode run --provider openai --model gpt-5.5 --input ./final/final_items.csv --prompts direct_zero_shot role_teacher --output-dir ./runs/run_openai
```

Несколько моделей одного провайдера:

```bash
python main.py --api-mode run --provider openai --models gpt-5.4-mini gpt-5.4 gpt-5.5 --input ./final/final_items.csv --prompts direct_zero_shot minimal_edit --output-dir ./runs/run_openai_models
```

Dry-run без вызова API и без ключей:

```bash
python main.py --api-mode dry-run --provider openai --model gpt-5.5 --input ./final/final_items.csv --prompts direct_zero_shot --limit 5 --output-dir ./runs/dry_run_openai
```

Матричный прогон по нескольким провайдерам и моделям:

```bash
python main.py --api-mode run-matrix --matrix-config ./configs/experiment_matrix_main_v4.json
```

Продолжение прерванного прогона:

```bash
python main.py --api-mode run --provider openai --model gpt-5.5 --input ./final/final_items.csv --prompts direct_zero_shot --output-dir ./runs/run_openai --run-id run_openai --resume
```

Метрики по нормализованным результатам:

```bash
python main.py --api-mode metrics --input ./runs/run_openai/normalized_results.csv --output-dir ./runs/run_openai/metrics
```

Для каждого запуска создаются:

- `run_manifest.json` - параметры запуска и выбранные item ids;
- `run_log.txt` - прозрачный журнал выполнения;
- `raw_requests.jsonl` - payload запросов с замаскированными ключами;
- `raw_responses.jsonl` - сырые ответы или ошибки API;
- `normalized_results.csv`;
- `normalized_results.jsonl`.

Resume работает по ключу `(provider, model, prompt_id, item_id)`: если такая комбинация уже есть в `normalized_results.csv`, она пропускается.

Промпты хранятся в `prompts.py`. Сейчас доступны:

- `direct_zero_shot`;
- `role_teacher`;
- `minimal_edit`;
- `step_by_step`;
- `grammar_prompt`.

Все промпты требуют единый JSON-ответ:

```json
{
  "has_error": true,
  "error_type": "agreement",
  "correction": "исправленное предложение"
}
```

Если модель вернула невалидный JSON, `parser.py` пытается извлечь JSON-объект из текста. Если это не удалось, сохраняется сырой ответ и выставляется `parse_status=failed` или `partial`.

Схема API-слоя описана в `schema_api.md`.

## Тесты

```bash
python -m unittest discover -s tests -v
```
