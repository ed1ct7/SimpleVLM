---
language:
- ru
license: apache-2.0
library_name: peft
pipeline_tag: image-text-to-text
base_model: IlyaGusev/saiga_llama3_8b
tags:
- vlm
- llava
- qlora
- russian
- visual-question-answering
datasets:
- deepvk/LLaVA-Instruct-ru
- deepvk/GQA-ru
- deepvk/MMBench-ru
---

# VK-VLM 8B — русскоязычная визуально-языковая модель (Saiga-8b + CLIP, LLaVA, QLoRA)

Русскоязычная VLM по рецепту **LLaVA**: на вход изображение + текст, на выход текст (рус.).
Публикуются **LoRA-адаптеры LLM + обученный проектор** (база CLIP/Saiga тянется с HF при загрузке).

```
изображение → CLIP ViT-L/14-336 → проектор (2-сл. MLP) → Saiga-8b (Llama-3-8B рус.) → текст
```

- **Энкодер:** `openai/clip-vit-large-patch14-336` (336px, слой −2, 576 image-токенов)
- **Проектор:** 2-слойный MLP, обучен с нуля
- **LLM:** `IlyaGusev/saiga_llama3_8b` (та же база, что у эталона `deepvk/llava-saiga-8b`)
- **Обучение:** 2 стадии LLaVA (alignment проектора → instruction tuning) + QLoRA (4-bit nf4,
  LoRA r16/α32, проектор в `modules_to_save`) на RTX 5080 16 ГБ

## Данные

- **Обучение:** `deepvk/LLaVA-Instruct-ru` (подвыборки 6k/15k) + изображения **COCO 2014**
  (сшивка по 12-значному id COCO).
- **Оценка:** `deepvk/GQA-ru`, `deepvk/MMBench-ru`.

## Метрики (accuracy %, единый протокол, greedy)

| Модель | GQA-ru extracted (exact) | MMBench-ru |
|---|---|---|
| Эталон `deepvk/llava-saiga-8b` | 54.30 (54.00) | 70.15 |
| **VK-VLM 8B (эта модель)** | 28.60 (6.60) | 57.08 |

Цель — рабочая VLM + честный замер против эталонов (выполнено). Модель обучена на **подвыборке**
(debug-grade объём) → ожидаемо ниже эталона; обойти эталон — stretch (рычаг: полный манифест).
GQA: `extracted` — короткий ответ из (многословного) предсказания, одинаково для всех моделей;
`exact` (в скобках) — строгий стандарт GQA, штрафующий многословие. Подробности — в репозитории
кода (`SOLUTION.md`).

## Инференс

```python
import json, torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
from huggingface_hub import snapshot_download
# build_vlm — из репозитория кода (solution/train/model.py)
from model import build_vlm

adapters = snapshot_download("<HF_USERNAME>/vk-vlm-saiga8b-clip-lora")
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

> `build_vlm` детерминированно пересобирает базу (CLIP + Saiga, 4-bit) и догружает адаптеры —
> код в репозитории решения (`solution/train/`).

## Ограничения

Debug-grade объём обучения; ответы бывают многословны (бьёт по строгому GQA-exact). Возможны
галлюцинации/неточности. Не для продакшена без дообучения на полном корпусе.

## Лицензия

Адаптеры — Apache-2.0. Базовые модели/данные — по их исходным лицензиям (Saiga/Llama-3, CLIP,
COCO, датасеты deepvk).
