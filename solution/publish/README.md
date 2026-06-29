# publish/ — публикация весов на Hugging Face Hub

Публикуются **обученные адаптеры + проектор + процессор** (база CLIP/Saiga тянется с HF при
загрузке). Авторизация — **ваша**: скрипт не хранит токен.

## Шаги

```bash
# 1) логин под своим аккаунтом (один раз)
huggingface-cli login            # либо: export HF_TOKEN=hf_...

# 2) публикация целевой 8B (из корня репозитория)
python solution/publish/upload_hf.py --repo-id <HF_USERNAME>/vk-vlm-saiga8b-clip-lora
#    приватно: добавьте --private
#    другая модель: --model-dir solution/model/2b-debug

# 3) вставьте полученную ссылку в:
#    - solution/README.md   (раздел «Веса»)
#    - solution/SOLUTION.md  (раздел «Веса»)
```

## Что загружается

| Файл | Что это |
|---|---|
| `adapter_model.safetensors` (~261 МБ) | LoRA-адаптеры LLM + обученный проектор |
| `adapter_config.json` | конфиг PEFT-LoRA |
| `training_meta.json` | id энкодера/LLM, гиперы, финальный loss, пик VRAM |
| `tokenizer*`, `processor_config.json`, `chat_template.jinja` | процессор |
| `README.md` (на репозитории) | карточка модели из `MODEL_CARD.md` |

Промежуточные чекпойнты (`checkpoint-*/`) и `training_args.bin` **не** загружаются.

## Альтернатива — архив для Облака Mail

Если предпочитаете архив вместо HF Hub:

```bash
cd solution/model && zip -r ../../vk-vlm-8b-weights.zip 8b -x '8b/checkpoint-*/*'
```

Загрузите `vk-vlm-8b-weights.zip` в Облако Mail, вставьте публичную ссылку в README/SOLUTION.
(Архив `*.zip` в git не коммитится — см. `solution/.gitignore`.)

## Файлы

- `upload_hf.py` — загрузчик на HF Hub (idempotent: `create_repo(exist_ok=True)`).
- `MODEL_CARD.md` — карточка модели (YAML frontmatter + описание) → README репозитория.
