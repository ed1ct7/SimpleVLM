# VK-VLM 8B — финальные адаптеры (Saiga-8b + CLIP, LLaVA, QLoRA) — задача 04

Целевая русскоязычная VLM: **CLIP ViT-L/14-336 → проектор (2-сл. MLP) → Saiga-8b**, обучена
по 2-стадийному рецепту LLaVA с QLoRA (CONTEXT §3, docs/decisions.md D1–D2, D9–D10).

> Веса (`*.safetensors`) в git **не коммитятся** (см. `solution/.gitignore`) — публикуются на
> HF Hub либо берутся из локального прогона. Здесь — конфиги адаптеров, процессор и мета.

## Состав

| Файл | Что это |
|---|---|
| `adapter_model.safetensors` (261 МБ) | LoRA-адаптеры LLM + обученный проектор (`modules_to_save`) |
| `adapter_config.json` | конфиг PEFT-LoRA (база, target-модули, r/α) |
| `training_meta.json` | id энкодера/LLM, стадия, гиперы, финальный loss, пик VRAM |
| `tokenizer*`, `processor_config.json`, `chat_template.jinja` | процессор (токенайзер с `<image>` + image_processor + chat-шаблон) |
| `checkpoint-{500,750,938}/` | промежуточные чекпойнты адаптеров (для `--resume`) |

Базовые компоненты (`training_meta.json`):
`openai/clip-vit-large-patch14-336` + `IlyaGusev/saiga_llama3_8b`, 4-bit nf4, 336px, `max_length=768`.

## Как обучено

- **Стадия 1 (alignment):** энкодер+LLM заморожены, обучается только проектор. Артефакт —
  `solution/model/8b-stage1/projector.safetensors`. Loss 9.31 → 1.58.
- **Стадия 2 (instruction tuning):** проектор стартует из стадии 1, QLoRA на LLM + проектор,
  assistant-only loss, на 15000 примерах `LLaVA-Instruct-ru`. Loss → **1.108**, token-acc 0.72.
- Пик VRAM **12.48 ГБ / 16** (RTX 5080, sm_120, WSL2). Полный лог — `solution/train/RUN-8b.md`.

## Инференс

```bash
source ~/vk-vlm-env/bin/activate
cd /mnt/d/Fork/SimpleVLM
export HF_HOME=~/vk-vlm-data/hf-cache
python solution/train/infer.py --adapters solution/model/8b --n 3
```

`infer.py` читает `training_meta.json`, детерминированно пересобирает базу (`build_vlm`) и
догружает адаптеры+проектор (`PeftModel.from_pretrained`). Проверено: 3/3 связных русских
ответа по картинке.
