# VK-VLM — русскоязычная визуально-языковая модель (LLaVA, QLoRA)

Обучение визуально-языковой модели (VLM) на открытых данных VK (коллекция `deepvk` на
Hugging Face) по рецепту **LLaVA** и честная оценка на бенчмарках **GQA-ru** и **MMBench-ru**.
Архитектура: визуальный энкодер (CLIP ViT-L/14-336) → проектор (2-слойный MLP) →
LLM (Saiga-8b). Обучение в 2 стадии с QLoRA под 16 ГБ VRAM (RTX 5080, Blackwell).

**Само решение — в [`solution/`](solution/).** Начать стоит с
[`solution/SOLUTION.md`](solution/SOLUTION.md) (подробное описание) и
[`solution/README.md`](solution/README.md) (карточка модели + запуск).

## Документы проекта

| Файл | Что внутри |
|---|---|
| [`CONTEXT.md`](CONTEXT.md) | Спецификация: задача, данные, архитектура, стек, железо, цель |
| [`PLAN.md`](PLAN.md) | План работ по алгоритму VK (этапы 1–10) |
| [`STATE.md`](STATE.md) | Журнал прогресса и таблица метрик |
| [`docs/datasets.md`](docs/datasets.md) | Разбор открытых данных deepvk |
| [`docs/decisions.md`](docs/decisions.md) | Лог технических решений |

## Структура

```
.
├── README.md           ← этот файл
├── CONTEXT.md          ← спецификация проекта
├── PLAN.md             ← план работ (этапы 1–10)
├── STATE.md            ← журнал прогресса + метрики
├── docs/
│   ├── datasets.md     ← разбор открытых данных deepvk
│   └── decisions.md    ← лог технических решений
└── solution/           ← код и артефакты решения
    ├── env/            ← окружение (requirements, проверка GPU)
    ├── data/           ← загрузка LLaVA-Instruct-ru + COCO, сборка манифеста
    ├── train/          ← код обучения (model/collator/train/infer), конфиги
    ├── eval/           ← оценка GQA-ru + MMBench-ru, единый протокол
    ├── db/             ← SQLite-база результатов (схема + загрузчик + запросы)
    ├── model/          ← обученные адаптеры/проектор + процессор + мета
    ├── results/        ← метрики + сырые предсказания
    ├── publish/        ← публикация весов на HF Hub
    ├── README.md       ← карточка модели + инструкция запуска
    └── SOLUTION.md     ← подробное описание решения
```

## Быстрый старт

```bash
# инференс обученной модели (WSL2 + GPU):
python solution/train/infer.py --adapters solution/model/8b --chat
# оценка на бенчмарках:
bash solution/eval/run_eval.sh
```

Подробности окружения, обучения и воспроизведения — в [`solution/`](solution/) и его README.
