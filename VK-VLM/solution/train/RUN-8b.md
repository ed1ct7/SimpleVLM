# RUN-8b — лог целевого обучения VK-VLM 8B (Saiga-8b, QLoRA, 2 стадии)

Задача 04. Рецепт LLaVA в 2 стадии на `IlyaGusev/saiga_llama3_8b` (= база эталона
`deepvk/llava-saiga-8b`, D9) + энкодер `openai/clip-vit-large-patch14-336` (D6).
Железо — RTX 5080, 16 ГБ VRAM, sm_120, WSL2 (CONTEXT §4). Веса в `~/vk-vlm-data/hf-cache` (ext4).

## Архитектура (повторяет эталон deepvk)

```
CLIP ViT-L/14-336 (336px, 576 image-токенов, layer −2, default)
   → проектор (LlavaMultiModalProjector, 2-слойный MLP, gelu)
   → Saiga-8b (Llama-3-8B, рус.), 4-bit nf4 + double-quant, compute bf16
```

## Команды (воспроизведение)

```bash
# окружение: venv этапа 01, в WSL2; данные/кэш — в ext4 ~/, не на /mnt
source ~/vk-vlm-env/bin/activate
cd /mnt/d/Fork/SimpleVLM/VK-VLM
export HF_HOME=~/vk-vlm-data/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# дым каждой стадии (8 примеров / 2 шага) — проверка сборки/памяти
python solution/train/train.py --config solution/train/configs/8b-stage1.yaml --smoke
python solution/train/train.py --config solution/train/configs/8b-stage2.yaml --smoke

# полный 2-стадийный прогон + инференс одним фоновым процессом (лог в файл)
wsl -d Ubuntu-24.04 -- bash -lc \
  "bash solution/train/run_8b.sh > solution/train/logs/run-8b.log 2>&1"

# обрыв стадии 2 → дообучить с последнего чекпойнта (адаптеры каждые 250 шагов)
python solution/train/train.py --config solution/train/configs/8b-stage2.yaml --resume

# инференс отдельно
python solution/train/infer.py --adapters solution/model/8b --n 3
```

> WSL убивает отвязанные (`&`/nohup/setsid) процессы между вызовами `wsl --` → обе стадии
> запускаются ОДНИМ фоновым процессом (`run_8b.sh`); скачивание весов Saiga (~16 ГБ) делать
> заранее синхронно (иначе фоновый прогон рвётся на докачке).

## Гиперпараметры

| | Стадия 1 (alignment) | Стадия 2 (instruction tuning) |
|---|---|---|
| что обучается | **только проектор** (энкодер+LLM заморожены) | **QLoRA на LLM + проектор** |
| обучаемых параметров | 20.98M / 4865M (0.43%) | 65.28M / 8420M (0.78%) |
| данные (подвыборка LLaVA-Instruct-ru) | 6000 | 15000 |
| epochs / шагов | 1 / 375 | 1 / ~937 |
| batch × grad_accum (эфф.) | 1 × 16 (16) | 1 × 16 (16) |
| lr / scheduler / warmup | 1e-3 / cosine / 0.03 | 2e-4 / cosine / 0.03 |
| max_length / картинка | 768 / 336px | 768 / 336px |
| loss | по всей последовательности | **assistant-only** (`mask_prompt`) |
| 4-bit / grad-checkpointing / optim | nf4+dq / on / paged_adamw_8bit | nf4+dq / on / paged_adamw_8bit |
| проектор-инициализация | normal 0.02 (meta-init) | из стадии 1 (`projector.safetensors`) |
| чекпойнты | projector callback (~20 МБ) | адаптеры каждые 250 шагов (`save_total_limit=3`) |

Конфиги: [`configs/8b-stage1.yaml`](configs/8b-stage1.yaml), [`configs/8b-stage2.yaml`](configs/8b-stage2.yaml).

## Память (16 ГБ VRAM) — что сделано

- 4-bit база (bitsandbytes nf4 + double-quant) + bf16-проектор + grad-checkpointing
  (`use_reentrant=False`) + micro-batch=1 + grad_accum=16, картинки 336px, `max_length=768`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — меньше фрагментации у потолка.
- `meta_init` сборки — без транзиентного случайного 8B на CPU (~32 ГБ RAM; STATE 03 #3).
- **CPU-offload оптимизатора НЕ понадобился** — пик VRAM ~12 ГБ оставлял запас.

## Замеры (дым + бенч)

- Дым стадии 1 (2 шага): сборка meta-init OK, **пик VRAM 11.99 ГБ**, ~4.8 с/шаг.
- Дым стадии 2 (2 шага): QLoRA OK, assistant-mask выведен корректно
  (`<|start_header_id|>assistant<|end_header_id|>\n\n` → `[128006,78191,128007,271]`),
  **пик VRAM 12.47 ГБ**, ~5.3 с/шаг.
- Бенч стадии 1 (25 шагов, эфф. батч 16): **~8.4 с/шаг**; loss **9.31 → 1.95** за 10→20 шаг,
  token-acc 0.03 → 0.61 → проектор выравнивается быстро.

## Результаты прогона (2026-06-23, один фоновый процесс `run_8b.sh`)

| | Стадия 1 | Стадия 2 | Итого |
|---|---|---|---|
| время (wall) | 16:38:16 → 17:31:19 = **53 мин** | 17:31:19 → 20:31:53 = **3 ч 00 мин** | + инференс 1 мин 54 с → **~3 ч 56 мин** |
| шагов | 375 | 938 | — |
| с/шаг (эфф. батч 16) | ~8.4 | ~11.5 (LoRA-backward + длиннее инструкции) | — |
| **train_loss** | **9.31 → 1.58** | **→ 1.108** (token-acc 0.72) | снижается ✅ |
| **пик VRAM** | **12.00 ГБ** / 16 | **12.48 ГБ** / 16 | запас ~3.5 ГБ, без CPU-offload ✅ |
| чекпойнты | `projector.safetensors` | `checkpoint-{500,750,938}` (адаптеры) | сохранены ✅ |
| OOM / sm_120 | нет | нет | ✅ |

**Артефакты:**
- `solution/model/8b-stage1/projector.safetensors` (42 МБ) — выровненный проектор (вход стадии 2).
- `solution/model/8b/` — финальные LoRA-адаптеры + проектор (`adapter_model.safetensors` 261 МБ,
  `adapter_config.json`, процессор, `training_meta.json`) + чекпойнты.

**Инференс (3/3 связных русских ответа по картинке), `infer.py --adapters solution/model/8b`:**
- *мотоцикл на грязевой трассе → опасность?* → «…возникает опасность потерять управление… из-за
  скользкости и неровностей грунта…»
- *человек, интересующийся велосипедом?* → «…возможно, заинтересован в покупке или использовании…
  изучает характеристики…»
- *автобус и скорая на улице?* → «…припаркованы рядом… водители, возможно, общаются…»

Ответы беглые, по-русски, привязаны к изображению. **Критерии приёмки выполнены.**

## Масштабирование до полного объёма (этап 05/финал)

Для максимальной метрики против эталона `deepvk/llava-saiga-8b`: поднять `train.subset` до
`109905` (весь манифест) в обоих конфигах, при желании 2-3 эпохи стадии 2. Время ≈ линейно:
полный стадия 2 ~ (109905/15000)×3 ч ≈ 22 ч → запускать в фоне/tmux, `--resume` при обрыве.
Также кандидат на прирост — энкодер SigLIP (D6, одна строка `encoder_id` в YAML).
