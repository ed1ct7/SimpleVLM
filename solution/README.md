# VK-VLM — русскоязычная визуально-языковая модель (Saiga-8b + CLIP, LLaVA, QLoRA)

Карточка модели + инструкция запуска (обязательное требование VK №3). Подробное описание
решения — [`SOLUTION.md`](SOLUTION.md). Источник правды по проекту — [`../CONTEXT.md`](../CONTEXT.md).

## Что за модель

VLM: на вход изображение + текст (рус.), на выход текст (рус.). Архитектура — рецепт **LLaVA**:

```
изображение → CLIP ViT-L/14-336 → проектор (2-сл. MLP) → Saiga-8b (Llama-3-8B рус.) → текст
```

- **Энкодер:** `openai/clip-vit-large-patch14-336` (336px, слой −2, 576 image-токенов).
- **Проектор:** 2-слойный MLP (`LlavaMultiModalProjector`), обучен с нуля.
- **LLM:** `IlyaGusev/saiga_llama3_8b` (та же база, что у эталона `deepvk/llava-saiga-8b`).
- **Обучение:** 2-стадийный LLaVA (alignment проектора → instruction tuning) + **QLoRA**
  (4-bit nf4 база, LoRA r16/α32, проектор в `modules_to_save`).
- **Что публикуется:** LoRA-адаптеры LLM + обученный проектор (~261 МБ) + процессор. Базовые
  веса (CLIP, Saiga) тянутся с HF при загрузке.

Метрики (accuracy %, единый протокол — см. [`SOLUTION.md`](SOLUTION.md) §5):

| Модель | GQA-ru extracted (exact) | MMBench-ru |
|---|---|---|
| Эталон `deepvk/llava-saiga-8b` | 54.30 (54.00) | 70.15 |
| **VK-VLM 8B (эта модель)** | 28.60 (6.60) | 57.08 |

> GQA: **extracted** — короткий ответ из (многословного) предсказания, одинаково для всех
> моделей; в скобках строгий `exact` (стандарт GQA, штрафует многословие). Эталоны краткие →
> extracted≈exact; мои модели многословны → extracted честнее (D11).
> Цель — обучить рабочую VLM и **честно измерить** её против эталонов (выполнено). Модель
> обучена на **подвыборке** `LLaVA-Instruct-ru` (debug-grade объём) → ожидаемо ниже эталона;
> обойти эталон — stretch-цель, рычаг — полный манифест, см. [`SOLUTION.md`](SOLUTION.md) §7.

## Веса

**Адаптеры + проектор + процессор:** ⟶ **`https://huggingface.co/<HF_USERNAME>/vk-vlm-saiga8b-clip-lora`**

> ⚠️ Плейсхолдер. Опубликуйте веса своим аккаунтом (скрипт готов, авторизация — ваша):
> ```bash
> python solution/publish/upload_hf.py --repo-id <HF_USERNAME>/vk-vlm-saiga8b-clip-lora
> ```
> После публикации замените `<HF_USERNAME>/...` на реальный repo id здесь и в
> [`SOLUTION.md`](SOLUTION.md). Детали — [`publish/README.md`](publish/README.md).

В git коммитятся только конфиги/процессор/мета (`solution/model/8b/` без `*.safetensors` —
см. [`.gitignore`](.gitignore)); сами веса — на HF Hub либо из локального прогона.

## Как загрузить и сделать инференс

Окружение — WSL2 + стек из [`env/requirements.txt`](env/requirements.txt) (см. [`env/README.md`](env/README.md)).

### Вариант A — из локального прогона (веса в `solution/model/8b/`)

```bash
source ~/vk-vlm-env/bin/activate
cd /mnt/d/Fork/SimpleVLM
export HF_HOME=~/vk-vlm-data/hf-cache          # кэш базовых весов в WSL-ФС, не на C:
python solution/train/infer.py --adapters solution/model/8b --n 3
```

`infer.py` читает `training_meta.json` (id энкодера/LLM), детерминированно пересобирает базу
(`build_vlm`) и догружает адаптеры+проектор через `PeftModel.from_pretrained`. Проверено:
3/3 связных русских ответа по картинке.

**Спросить модель самому** (своя картинка + вопрос):

```bash
# разовый вопрос:
python solution/train/infer.py --adapters solution/model/8b \
    --image ~/my.jpg --question "Что изображено на картинке?"
# интерактивный чат (вводишь картинку и вопросы, 'exit' — выход):
python solution/train/infer.py --adapters solution/model/8b --chat
```

### Вариант B — из HF Hub

```bash
# 1) скачать адаптеры в локальный каталог
huggingface-cli download <HF_USERNAME>/vk-vlm-saiga8b-clip-lora --local-dir solution/model/8b-hub
# 2) тот же инференс, указав каталог
python solution/train/infer.py --adapters solution/model/8b-hub --n 3
```

### Минимальный код инференса (своя картинка/вопрос)

```python
import json, torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
import sys; sys.path.append("solution/train")
from model import build_vlm

adapters = "solution/model/8b"                      # или каталог, скачанный с HF Hub
meta = json.load(open(f"{adapters}/training_meta.json"))
model, _, _ = build_vlm(meta["encoder_id"], meta["llm_id"],
                        load_in_4bit=True, image_size=meta["image_size"],
                        compute_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapters).eval()
proc = AutoProcessor.from_pretrained(adapters)

img = Image.open("your.jpg").convert("RGB")
msg = [{"role": "user", "content": "<image>\nЧто изображено на картинке?"}]
prompt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inp = proc(images=[img], text=[prompt], return_tensors="pt").to("cuda")
out = model.generate(**inp, max_new_tokens=128, do_sample=False)
print(proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

## Как воспроизвести оценку

```bash
source ~/vk-vlm-env/bin/activate
cd /mnt/d/Fork/SimpleVLM
bash solution/eval/run_eval.sh                  # все 4 модели, GQA-ru + MMBench-ru (резюмируемо)
# отдельная модель:
python solution/eval/eval_gqa.py --model mine-8b
python solution/eval/eval_mmbench.py --model mine-8b
python solution/eval/aggregate.py               # свод → results/metrics.md
```

Протокол и ключи моделей — [`eval/README.md`](eval/README.md). Результаты —
[`results/metrics.md`](results/metrics.md), сырые предсказания — `results/raw/*.jsonl`.

## Как воспроизвести обучение

```bash
# данные (LLaVA-Instruct-ru + COCO train2014, сшивка по id):
python solution/data/download.py && python solution/data/build_dataset.py
# 8B, обе стадии + инференс (предварительно один раз синхронно скачать Saiga):
wsl -d Ubuntu-24.04 -- bash -lc "bash solution/train/run_8b.sh > solution/train/logs/run-8b.log 2>&1"
```

Стадии по отдельности, гиперы, заметки по памяти — [`train/README.md`](train/README.md) и
[`train/RUN-8b.md`](train/RUN-8b.md).

## Структура `solution/`

```
solution/
├── env/        # окружение: requirements, проверка GPU (sm_120)
├── data/       # загрузка LLaVA-Instruct-ru + COCO, сборка манифеста
├── train/      # код обучения (model/collator/train/infer), конфиги, раннеры
├── eval/       # оценка GQA-ru + MMBench-ru, единый протокол
├── model/      # 8b/ (целевая), 8b-stage1/ (проектор), 2b-debug/ — конфиги+процессор+мета
│               #   (веса *.safetensors не в git — HF Hub / локальный прогон)
├── results/    # metrics.md + raw/*.jsonl (сырые предсказания)
├── publish/    # скрипт публикации весов на HF Hub + model card
├── README.md   # этот файл (карточка модели + запуск)
└── SOLUTION.md # подробное описание решения (требование №4)
```

## Лицензии и происхождение

- Данные: `deepvk/LLaVA-Instruct-ru`, COCO 2014 — открытые. Бенчмарки `deepvk/GQA-ru`,
  `deepvk/MMBench-ru` — открытые.
- Базовые модели: `openai/clip-vit-large-patch14-336`, `IlyaGusev/saiga_llama3_8b` (не gated).
- Публикуются только обученные адаптеры+проектор; базовые веса — по их исходным лицензиям с HF.
