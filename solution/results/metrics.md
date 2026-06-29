# Метрики оценки — GQA-ru + MMBench-ru (этап 05)

Accuracy (%). Единый протокол: greedy-декодирование, свой процессор/chat-шаблон у каждой
модели, общий текст задачи и метрика. Протокол и воспроизведение — `solution/eval/README.md`.

- **GQA-ru**: testdev_balanced, exact-match с нормализацией ответа. N = **1000** вопросов.
- **MMBench-ru**: dev, single-choice, accuracy по распарсенной букве. N = **3910** вопросов.

GQA-ячейка: **extracted (exact)**. `extracted` — короткий ответ, извлечённый из
(возможно многословного) ответа: для да/нет — первое слово-полярность, иначе gold как
ведущее слово/среди первых трёх (D11). Применяется ко всем моделям одинаково; эталоны уже
краткие → не меняются, многословные мои модели возвращают потерянные совпадения. `exact` —
строгий стандарт GQA (штрафует многословие). `lenient` (доп. таблица) — потолок: gold где
угодно в ответе.

| Модель | GQA-ru extracted (exact) | MMBench-ru (acc) |
|---|---|---|
| deepvk/llava-gemma-2b-lora | 45.50 (45.40) | 62.99 |
| deepvk/llava-saiga-8b | 54.30 (54.00) | 70.15 |
| Моя 2B | 21.60 (1.40) | 34.63 |
| Моя 8B | 28.60 (6.60) | 57.08 |

## Доп. показатели

| Прогон | n | exact | extracted | lenient / доп |
|---|---|---|---|---|
| deepvk/llava-gemma-2b-lora · gqa | 1000 | 45.40% | 45.50% | lenient=45.50% |
| deepvk/llava-gemma-2b-lora · mmbench | 3910 | 62.99% | — | letter_rate=100.0%, no_letter=0 |
| deepvk/llava-saiga-8b · gqa | 1000 | 54.00% | 54.30% | lenient=54.30% |
| deepvk/llava-saiga-8b · mmbench | 3910 | 70.15% | — | letter_rate=100.0%, no_letter=0 |
| Моя 2B · gqa | 1000 | 1.40% | 21.60% | lenient=24.00% |
| Моя 2B · mmbench | 3910 | 34.63% | — | letter_rate=100.0%, no_letter=0 |
| Моя 8B · gqa | 1000 | 6.60% | 28.60% | lenient=30.10% |
| Моя 8B · mmbench | 3910 | 57.08% | — | letter_rate=100.0%, no_letter=0 |
