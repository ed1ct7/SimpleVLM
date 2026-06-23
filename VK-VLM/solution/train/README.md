# Обучение VK-VLM (этапы 03 — отладочный 2B; 04 — целевой 8B Saiga, 2 стадии)

QLoRA-обучение VLM LLaVA-стиля: **энкодер (CLIP/SigLIP) → проектор (MLP) → LLM (Qwen2/Saiga-8b)**.
Этап 03 — рабочий цикл на 2B (не метрика). Этап 04 — целевое 8B по **2-стадийному рецепту LLaVA**.

## Файлы

| Файл | Назначение |
|---|---|
| `model.py` | сборка VLM (`build_vlm`, meta-init для 8B) + QLoRA (`apply_qlora`) + стадия 1 (`freeze_for_alignment`) + save/load проектора |
| `data_collator.py` | коллатор (картинка+текст); `mask_prompt` = assistant-only loss (финал) |
| `train.py` | обучение через `TRL SFTTrainer`; `--stage {1,2}`, `--resume`, `--init-projector-from` |
| `infer.py` | загрузка адаптеров + инференс на N примерах (русский ответ по картинке) |
| `configs/2b-debug.yaml` | гиперпараметры отладочного прогона 2B (этап 03) |
| `configs/8b-stage1.yaml` | 8B стадия 1 (alignment: только проектор) |
| `configs/8b-stage2.yaml` | 8B стадия 2 (instruction tuning: QLoRA + проектор) |
| `run_8b.sh` | прогон обеих стадий + инференс одним фоновым процессом (этап 04) |

## 8B (этап 04): 2-стадийный рецепт LLaVA

**Стадия 1 (alignment).** Энкодер + LLM (4-bit) **заморожены**, обучается **только проектор**
(`freeze_for_alignment`). Цель — выровнять визуальные фичи CLIP с пространством эмбеддингов
Saiga-8b. Артефакт — `projector.safetensors` (вход стадии 2). См. D10.

**Стадия 2 (instruction tuning).** Проектор стартует с выровненного (`init_projector_from`),
навешивается **QLoRA** на LLM (`apply_qlora`), совместное дообучение проектора + LoRA на
инструкциях `LLaVA-Instruct-ru` с **assistant-only loss** (`mask_prompt: true`). Артефакт —
LoRA-адаптеры + проектор в `solution/model/8b/`.

Запуск (один фоновый процесс — WSL убивает отвязанные процессы между вызовами `wsl --`):

```bash
# дым каждой стадии (8 примеров / 2 шага) — проверить сборку/память без долгого прогона
python solution/train/train.py --config solution/train/configs/8b-stage1.yaml --smoke
python solution/train/train.py --config solution/train/configs/8b-stage2.yaml --smoke

# полный прогон обеих стадий + инференс, лог в файл (фон/tmux)
wsl -d Ubuntu-24.04 -- bash -lc \
  "bash solution/train/run_8b.sh > solution/train/logs/run-8b.log 2>&1"
# обрыв стадии 2 → дообучить с последнего чекпойнта (адаптеры каждые 250 шагов):
python solution/train/train.py --config solution/train/configs/8b-stage2.yaml --resume
```

Лог запуска, время, пиковая VRAM, гиперы — `solution/train/RUN-8b.md`.

## Запуск (WSL2, venv из этапа 01)

```bash
source ~/vk-vlm-env/bin/activate
cd /mnt/d/Fork/SimpleVLM/VK-VLM

# 0) дым (8 примеров, 2 шага) — проверить цикл без долгого прогона
python solution/train/train.py --config solution/train/configs/2b-debug.yaml --smoke

# 1) отладочный прогон (2000 примеров, 1 эпоха)
python solution/train/train.py --config solution/train/configs/2b-debug.yaml

# 2) инференс на 3 примерах (грузит адаптеры + проектор)
python solution/train/infer.py --adapters solution/model/2b-debug --n 3
```

> Веса моделей кэшируются в `~/vk-vlm-data/hf-cache` (WSL-ФС, **не C:**) — `HF_HOME`
> выставляется в `train.py`/`infer.py` до импорта transformers. Это важно: скачивание весов на
> системный диск уже однажды положило WSL (инцидент disk-full, STATE этап 02).

## Как поменять LLM / энкодер (переиспользование в задаче 04)

Всё параметризовано — правьте YAML (или флаги CLI). Код сборки (`build_vlm`) не привязан к
конкретной модели:

```yaml
# другой энкодер (SigLIP — D6, часто даёт прирост):
encoder_id: google/siglip-so400m-patch14-384   # 384px; image_size подстроится под нативный

# другая / целевая LLM:
llm_id: google/gemma-2-2b-it                    # gated — нужен HF-токен (huggingface-cli login)
llm_id: IlyaGusev/saiga_llama3_8b               # задача 04 (8B) — БАЗА эталона deepvk/llava-saiga-8b (D9)
```

> NB: `deepvk/llava-saiga-8b` — это уже собранная VLM-**эталон** (для сравнения на этапе 05),
> а не базовая LLM. Для своего обучения берётся именно база `IlyaGusev/saiga_llama3_8b`.

`build_vlm` сам определяет тип энкодера (CLIP/SigLIP), число image-токенов, слой фич и стратегию
(CLIP: `default`/слой −2; SigLIP: `full`/слой −1), добавляет `<image>`-токен в словарь LLM и
ресайзит эмбеддинги. Под 8B при нехватке VRAM: уменьшить `max_length`, поднять `grad_accum`,
оставить `per_device_batch_size: 1`.

## Архитектура сборки (model.py)

1. LLM грузится 4-bit (bitsandbytes nf4, double-quant, compute bf16). В словарь добавляется
   `<image>`, эмбеддинги ресайзятся.
2. Энкодер (vision tower) грузится в bf16.
3. Строится `LlavaConfig` из под-конфигов, создаётся `LlavaForConditionalGeneration`, в него
   **пересаживаются** реальные веса энкодера и LLM. Проектор (`LlavaMultiModalProjector`,
   2-слойный MLP) инициализируется заново.
4. `apply_qlora`: `prepare_model_for_kbit_training` + LoRA на проекциях LLM
   (`q,k,v,o,gate,up,down`); **проектор обучается целиком** (`modules_to_save`), энкодер заморожен.
   Grad checkpointing включается в `SFTConfig`.

## Коллатор (data_collator.py)

На каждый батч: грузит PIL-картинку (путь относительно data-root), сводит структурный
chat-формат этапа 02 к плоскому тексту с одним `<image>`, применяет chat-шаблон LLM, гоняет
через `LlavaProcessor` (разворачивает `<image>` в N image-токенов), строит `labels` с маской
паддинга (по `attention_mask`) и image-токенов (`-100`).

> Отладочный режим: лосс по всей текстовой последовательности (исключены паддинг и image-токены).
> Assistant-only loss (маска промпта) — апгрейд для финального обучения (задача 04).

## Что сохраняется в `solution/model/2b-debug/`

LoRA-адаптеры (`adapter_model.safetensors` + `adapter_config.json`), обученный проектор
(в составе адаптеров, `modules_to_save`), процессор (токенайзер с `<image>` + image_processor +
chat-шаблон) и `training_meta.json` (id энкодера/LLM, гиперы, финальный loss). `infer.py` читает
`training_meta.json`, чтобы собрать ту же базу и догрузить адаптеры.

## Результаты прогона — см. `STATE.md` (журнал этапа 03).
