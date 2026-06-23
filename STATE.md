# STATE — трекер прогресса (append-only)

> Каждая агент-сессия в конце работы **дописывает** блок в раздел «Журнал».
> Не переписывать прошлые записи. Таблицу статусов — обновлять.

## Статус задач

| Задача | Промпт | Статус | Артефакты |
|---|---|---|---|
| Планирование (этапы 1–5) | — | ✅ done | CONTEXT.md, PLAN.md, docs/ |
| Окружение | 01-env-setup | ✅ done | solution/env/{check_gpu.py, requirements.txt, README.md, pip-freeze-full.txt} |
| Данные | 02-data-pipeline | ✅ done | solution/data/{download.py, build_dataset.py, sample.py, _common.py, README.md}, solution/.gitignore |
| Обучение 2B (отладка) | 03-train-2b | ✅ done | solution/train/{model.py, data_collator.py, train.py, infer.py, configs/2b-debug.yaml, README.md}, solution/model/2b-debug/ |
| Обучение 8B (цель) | 04-train-8b | ✅ done | solution/train/{configs/8b-stage1.yaml, 8b-stage2.yaml, run_8b.sh, RUN-8b.md}, solution/model/{8b-stage1/, 8b/} |
| Оценка + эталоны | 05-eval | ⬜ todo | — |
| Отчёт + упаковка | 06-report | ⬜ todo | — |

Легенда: ⬜ todo · 🔄 wip · ✅ done · ⛔ blocked

## Метрики (заполняется на этапе 05)

| Модель | GQA-ru (acc) | MMBench-ru (acc) |
|---|---|---|
| deepvk/llava-gemma-2b-lora | — | — |
| deepvk/llava-saiga-8b | — | — |
| Моя 2B | — | — |
| Моя 8B | — | — |

## Журнал

### 2026-06-22 — Планирование (оркестратор)
- Создан управляющий каркас: CONTEXT, PLAN, STATE, README, prompts/, docs/.
- Этапы 1–5 закрыты на планировании.
- **Разблокировано:** задача 01-env-setup готова к запуску.

### 2026-06-22 — 01 Окружение (WSL2 + CUDA 12.8 + ML-стек) ✅
**Хост:** Windows 11, RTX 5080 16 ГБ, драйвер NVIDIA 596.36 (CUDA 13.2 capable, ≥12.8 ✅).
**WSL2:** был установлен (только distro `docker-desktop`); поставлен `Ubuntu-24.04`
(`wsl --install -d Ubuntu-24.04 --no-launch`, работа от root). Python 3.12.3.
GPU пробрасывается в WSL: `nvidia-smi -L` → RTX 5080.

**Окружение:** venv `~/vk-vlm-env` (= `/root/vk-vlm-env`, WSL-ФС ext4, НЕ на `/mnt/*`).

**Установлено (зафиксировано в requirements.txt + pip-freeze-full.txt):**
- torch 2.11.0+cu128, torchvision 0.26.0+cu128, triton 3.6.0 (CUDA build 12.8, cuDNN 9.19).
- transformers 5.12.1, trl 1.6.0, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.14.0,
  datasets 5.0.0, pillow 12.2.0, sentencepiece 0.2.1.

**Проверка `check_gpu.py` — ALL CHECKS PASSED:**
- `CUDA available: True`, GPU `NVIDIA GeForce RTX 5080`, compute `sm_120`.
- arch list torch: `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — Blackwell включён.
- GPU matmul fp16 2048² — без `no kernel image is available`.
- bitsandbytes 4-bit: nf4 round-trip + Linear4bit forward на GPU — без ошибки sm_120.

**Артефакты:** `solution/env/check_gpu.py`, `requirements.txt`, `README.md`, `pip-freeze-full.txt`.
**Блокеров нет.** **Разблокировано:** задача 02-data-pipeline.

> Запуск проверки: `wsl -d Ubuntu-24.04 -- bash -lc "source ~/vk-vlm-env/bin/activate && python /mnt/d/Fork/SimpleVLM/solution/env/check_gpu.py"`
> (для тренировки данные/код держать в WSL-ФС `~/`, не на `/mnt/*` — медленный I/O).

### 2026-06-22 — 02 Пайплайн данных (LLaVA-Instruct-ru + COCO) ✅
**Артефакты:** `solution/data/download.py`, `build_dataset.py`, `sample.py`, `_common.py`,
`README.md`; `solution/.gitignore`.

**⚠️ Расхождение доков с реальными данными (сообщить оркестратору):**
- Поле `image` = `coco/train2017/<id>.jpg` (**2017-нейминг**, `000000253464.jpg`), а не
  `COCO_train2014_*.jpg`, как в CONTEXT/`docs/datasets.md`. Это **те же** снимки, что в COCO
  **train2014** (LLaVA построен на train2014).
- `train`-split = **109 905** записей (доки говорили 144k).
- Решение: сопоставление **по 12-значному id**, а не по имени файла. Резолвер
  (`_common.build_coco_index`) понимает 2014- и 2017-нейминг → качать можно `train2014`
  (деф., ~13 ГБ, покрывает все id) или 2017. CONTEXT/datasets.md не правил (это источник
  плана; расхождение задокументировано здесь, в `data/README.md` и `docs/decisions.md` D8).

**Пайплайн:**
1. `download.py` — `save_to_disk` датасета + curl-докачка/распаковка COCO в
   `~/vk-vlm-data/coco/` (WSL-ФС, не /mnt). Идемпотентен (пропуск готового сплита).
2. `build_dataset.py` — резолв картинки по id, drop+log записей без файла, конвертация
   `conversations` → chat-формат, манифест JSONL (`image` относительный) + `stats.json`.
3. `sample.py` — 5 случайных примеров (reservoir), печать диалога + проверка PIL (размеры > 0),
   код возврата 0/1.
4. chat-формат: роли human→user/gpt→assistant; шаблон параметром `--template {messages,text}`
   (`messages` = структурный для TRL — деф.; `text` = плоский с `--image-token`).

**Проверка — ПОЛНЫЙ прогон выполнен (данные на диске):**
- `download.py`: датасет `save_to_disk` (train 109 905 + val 34 075); COCO `train2014.zip`
  (12.6 ГБ) скачан и распакован → **82 783 jpg** в `~/vk-vlm-data/coco/train2014/`; zip удалён.
- `build_dataset.py` (`processed/stats.json`): всего **109 905** / с картинкой **109 905** /
  **отброшено 0**; индекс COCO 82 783; манифест 109 905 строк (196 МБ).
- `sample.py`: **5/5** картинок открылись (500×376, 640×428 JPEG …). Оба шаблона
  (`messages`/`text`) работают.
- **0 отброшенных** подтвердило гипотезу: все `coco/train2017/<id>`-пути сшиваются с
  COCO train2014 по 12-значному id. `val2014` не нужен. «144k» в доках = train(109 905)+val(34 075).

**⚠️ Инцидент disk-full (для будущих сессий):** COCO рос на ext4-vdisk Ubuntu-24.04,
который лежит на **C:** (`...\AppData\Local\wsl\{...}\ext4.vhdx`). C: ушёл в **0 байт** →
ext4 EIO → distro перестал стартовать (`Wsl/Service/CreateInstance/E_FAIL`). Лечение:
освободить C: (нужно ~30 ГБ на COCO: zip 13 + распаковка 13) → `wsl --shutdown` (сброс
utility-VM; останавливает и docker-desktop) → перезапуск Ubuntu → пере-распаковка (партиал
35 758/82 783 удалён) → пайплайн добежал. **На будущих этапах следить за C:** (чекпойнты,
кэши моделей тоже растут на этом же vdisk).

**Блокеров нет.** **Разблокировано:** задача 03-train-2b (данные + лоадер на диске).

> Воспроизведение: `python solution/data/download.py` → `build_dataset.py` → `sample.py`
> (из корня репо, venv `~/vk-vlm-env`). Идемпотентно: датасет/распакованный COCO пропускаются.

### 2026-06-22 — 03 Обучение 2B (отладочный end-to-end QLoRA) ✅
**Цель этапа достигнута:** рабочий цикл «данные → train → save адаптеров → инференс» без
ошибок (OOM / sm_120). Метрика на этом этапе не цель.

**Сборка VLM (LLaVA-стиль, готовый класс — не с нуля, D1):**
- энкодер **`openai/clip-vit-large-patch14-336`** (336px, 576 image-токенов, слой −2, `default`) →
  проектор **`LlavaMultiModalProjector`** (2-слойный MLP, инициализирован заново, обучается целиком) →
  LLM **`Qwen/Qwen2-1.5B-Instruct`**. Класс `transformers.LlavaForConditionalGeneration`:
  `LlavaConfig` из двух под-конфигов + **пересадка** реальных весов энкодера/LLM (`model.py:build_vlm`).
- **Почему Qwen2-1.5B, а не Gemma-2b:** Qwen2 открыт (Apache-2.0, мультиязычный, класс 2B);
  `google/gemma-2-2b-it` — **gated**, нужен HF-токен. Обе сменяются одной строкой в YAML.
- **QLoRA (D2):** база 4-bit (bitsandbytes nf4 + double-quant, compute bf16); LoRA r16/α32 на
  `q,k,v,o,gate,up,down`; проектор — `modules_to_save` (обучается+сохраняется); энкодер заморожен.
  Обучаемых **24.76M / 1.875B (1.32%)**.

**Гиперпараметры прогона** (`configs/2b-debug.yaml`): subset **2000**, **1 эпоха** = **250 шагов**,
`per_device_batch_size=1` × `grad_accum=8` (эфф. батч 8), `lr=1e-4` cosine + warmup 0.03,
`max_length=1024`, 336px, `optim=paged_adamw_8bit`, grad checkpointing, bf16 autocast.

**Результаты:**
- **Loss 1.634 → 1.172** (train_loss 1.16), mean_token_accuracy 0.65 → 0.72 — **снижается ✅**.
- Скорость **~3.4 с/шаг**, весь прогон **~14 мин**. **Пик VRAM ~7.6 ГБ / 16** (запас большой).
- Адаптеры сохранены: `solution/model/2b-debug/` — `adapter_model.safetensors` (99 МБ: LoRA+проектор),
  `adapter_config.json`, процессор (tokenizer+`<image>`, image_processor, chat_template), `training_meta.json`.
- **Инференс (`infer.py`): 3/3 связных русских ответа по картинке** (мотоцикл на грязевой трассе /
  велосипедист / автобус+скорая). Адаптеры+проектор грузятся через `PeftModel.from_pretrained` поверх
  детерминированно пересобранной базы (id из `training_meta.json`).

**⚠️ Заметки по памяти/гиперам — критично для задачи 04 (8B):**
1. **VRAM thrash у потолка.** Первый прогон при `max_length=2048` упёрся в ~15.9 ГБ → near-OOM:
   paged-optimizer начал пейджить состояния GPU↔CPU, шаг замедлился **~7x** (3.5 → 20-30 с/шаг).
   **Лечение:** `max_length=1024` (жёсткий потолок активаций) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   → пик упал до **7.6 ГБ**, шаг вернулся к 3.4 с. Для 8B (04): seq-кэп + expandable_segments
   **обязательны**, поднять `grad_accum`, держать `per_device_batch_size=1`.
2. **Рассинхрон image-токенов (off-by-one).** `LlavaProcessor` при `strategy="default"` отдаёт
   `num_patches−1=575` токенов, энкодер (дроп CLS) — 576 фич → merge падает. **Лечение:**
   `num_additional_image_tokens=1` (см. `model.py:build_processor`, D6). TRL списывал это на
   «truncation max_length too short» — ложный след.
3. **Транзиентная RAM при сборке.** `LlavaForConditionalGeneration(config)` создаёт случайные
   энкодер+LLM на CPU перед пересадкой (для 1.5B терпимо). Для 8B (04) рассмотреть `init_empty_weights`/
   meta-device, иначе ~30+ ГБ RAM транзиентно.
4. **HF_HOME → WSL-ФС** (`~/vk-vlm-data/hf-cache`, выставлено в train.py/infer.py до импорта
   transformers): веса ~5 ГБ, иначе забьют C: (инцидент этапа 02).
5. **Отладочные упрощения (апгрейд для 04/финала):** лосс по всей последовательности (не
   assistant-only) + нет акцента на eos → ответы многословны и обрезаются на 128 токенах. Финал:
   assistant-only loss; при необходимости 2-стадийность LLaVA (D1: сначала только проектор).

**Решения:** D6 закрыт частично — энкодер отладки CLIP-336, SigLIP остаётся кандидатом на метрику
(переключение готово в `build_vlm`), окончательно — по eval на этапе 05.

**Блокеров нет.** **Разблокировано:** задача 04-train-8b (сборка/коллатор/трейнер параметризованы,
переиспользуются: сменить `encoder_id`/`llm_id` в YAML, учесть заметки по памяти выше).

> Воспроизведение: `python solution/train/train.py --config solution/train/configs/2b-debug.yaml`
> (опц. `--smoke` — 8 примеров/2 шага) → `python solution/train/infer.py --adapters solution/model/2b-debug --n 3`.
> Из корня репо, venv `~/vk-vlm-env`, в WSL2.

### 2026-06-23 — 04 Обучение 8B (Saiga-8b, QLoRA, 2 стадии LLaVA) ✅
**Цель достигнута:** целевая VLM обучена по полному 2-стадийному рецепту LLaVA, обе стадии без
OOM/sm_120, loss снижается, финальная модель грузится и отвечает связно по-русски.

**Архитектура (повторяет эталон `deepvk/llava-saiga-8b`, проверено по его config, D9):**
`openai/clip-vit-large-patch14-336` (336px, 576 img-токенов, layer −2, default) → проектор
(2-сл. MLP) → **`IlyaGusev/saiga_llama3_8b`** (= база эталона; НЕ gated), 4-bit nf4+double-quant.

**Пайплайн доработан из 03 (переиспользован, параметризован):**
- `model.py`: `meta_init` (скелет на meta → нет ~32 ГБ RAM транзиентно, заметка 03 #3),
  `freeze_for_alignment` (стадия 1 — только проектор), `save_projector`/`load_projector`.
- `data_collator.py`: `mask_prompt` = **assistant-only loss** (учим отвечать, не повторять
  промпт; заметка 03 #5). Маркер выводится из chat-шаблона, multi-turn ок, безопасный фолбэк.
- `train.py`: `--stage {1,2}`, `--resume`, `--init-projector-from`; колбэк, кладущий ТОЛЬКО
  проектор (~20 МБ) на стадии 1 (модель не PEFT → штатный чекпойнт сдампил бы весь 4-bit base).
- Конфиги `configs/8b-stage1.yaml`, `8b-stage2.yaml`; раннер `run_8b.sh` (обе стадии + инференс
  одним фоновым процессом). **Лог:** `solution/train/RUN-8b.md`.

**Прогон (один фоновый процесс, ~3 ч 56 мин):**
- **Стадия 1** (alignment, только проектор 20.98M/4865M, subset 6000, 375 шагов, lr 1e-3):
  **53 мин**, loss **9.31 → 1.58**, пик VRAM **12.00 ГБ**. → `solution/model/8b-stage1/projector.safetensors`.
- **Стадия 2** (QLoRA r16/α32 на LLM + проектор 65.28M/8420M, subset 15000, 938 шагов, lr 2e-4,
  assistant-only): **3 ч 00 мин** (~11.5 с/шаг), loss **→ 1.108** (token-acc 0.72), пик VRAM
  **12.48 ГБ**. → `solution/model/8b/` (адаптеры 261 МБ + проектор + процессор + чекпойнты 500/750/938).
- **Память:** batch=1 × grad_accum=16, grad-checkpointing, 336px, `max_length=768`,
  `expandable_segments`. **CPU-offload оптимизатора НЕ понадобился** (запас ~3.5 ГБ).
- **Инференс (`infer.py --adapters solution/model/8b --n 3`): 3/3 связных русских ответа**
  (мотоцикл/опасность · велосипед/намерения · автобус+скорая), привязаны к картинке.

**Решения:** D9 (база 8B = `saiga_llama3_8b`, meta-init), D10 (стадия 1 — на подвыборке
LLaVA-Instruct-ru, рус. caption-корпуса нет). Энкодер пока CLIP (SigLIP — кандидат на метрику, D6).

**⚠️ Заметки для будущих сессий:**
1. **WSL убивает отвязанные процессы** (`&`/nohup/setsid) между вызовами `wsl --`. Долгий прогон
   запускать ОДНИМ фоновым процессом (раннер); веса (Saiga ~16 ГБ) качать заранее синхронно,
   иначе фон рвётся на докачке. `tail -f` по логу на `/mnt/*` (DrvFs) не стримит (нет inotify).
2. **Объём — подвыборка** (6k/15k из 109 905). Для финальной метрики (этап 05) поднять `subset`
   до полного манифеста (± эпохи) — оценка времени и команда в `RUN-8b.md`.

**Блокеров нет.** **Разблокировано:** задача 05-eval (есть своя 8B-модель `solution/model/8b/`
+ эталоны deepvk для сравнения на GQA-ru/MMBench-ru).

> Воспроизведение: `wsl -d Ubuntu-24.04 -- bash -lc "bash solution/train/run_8b.sh > solution/train/logs/run-8b.log 2>&1"`
> (предварительно один раз скачать Saiga синхронно). Стадии по отдельности — см. `RUN-8b.md` / `train/README.md`.
