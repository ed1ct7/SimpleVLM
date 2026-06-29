# STATE — трекер прогресса (append-only)

> По ходу работы в раздел «Журнал» **дописываются** блоки. Прошлые записи не
> переписываются. Таблица статусов — обновляется.

## Статус задач

| Задача | Промпт | Статус | Артефакты |
|---|---|---|---|
| Планирование (этапы 1–5) | — | ✅ done | CONTEXT.md, PLAN.md, docs/ |
| Окружение | 01-env-setup | ✅ done | solution/env/{check_gpu.py, requirements.txt, README.md, pip-freeze-full.txt} |
| Данные | 02-data-pipeline | ✅ done | solution/data/{download.py, build_dataset.py, sample.py, _common.py, README.md}, solution/.gitignore |
| Обучение 2B (отладка) | 03-train-2b | ✅ done | solution/train/{model.py, data_collator.py, train.py, infer.py, configs/2b-debug.yaml, README.md}, solution/model/2b-debug/ |
| Обучение 8B (цель) | 04-train-8b | ✅ done | solution/train/{configs/8b-stage1.yaml, 8b-stage2.yaml, run_8b.sh, RUN-8b.md}, solution/model/{8b-stage1/, 8b/} |
| Оценка + эталоны | 05-eval | ✅ done | solution/eval/{eval_common,eval_gqa,eval_mmbench,aggregate}.py, run_eval.sh, README.md; solution/results/{metrics.md, raw/} |
| Отчёт + упаковка | 06-report | ✅ done | solution/{SOLUTION.md, README.md, publish/{upload_hf.py, MODEL_CARD.md, README.md}} |

Легенда: ⬜ todo · 🔄 wip · ✅ done · ⛔ blocked

## Метрики (этап 05)

Единый протокол (greedy, свой процессор/chat-шаблон у каждой модели). GQA-ru: testdev_balanced,
N=1000 (seed 7), accuracy **extracted (exact)**, %. MMBench-ru: весь dev, N=3910, accuracy по
букве, % (у всех `letter_rate=100%` — формат вывода корректен). Подробно — `solution/results/metrics.md`.

| Модель | GQA-ru extracted (exact) | MMBench-ru (acc) |
|---|---|---|
| deepvk/llava-gemma-2b-lora | 45.50 (45.40) | 62.99 |
| deepvk/llava-saiga-8b | 54.30 (54.00) | 70.15 |
| Моя 2B | 21.60 (1.40) | 34.63 |
| Моя 8B | 28.60 (6.60) | 57.08 |

> GQA: **extracted** — короткий ответ из (многословного) предсказания, одинаково для всех моделей
> (D11); `exact` (в скобках) — строгий стандарт GQA, штрафует многословие. Эталоны краткие →
> extracted≈exact (не завышает их); мои многословны → extracted честнее (8B 6.6→28.6). Мои модели
> всё равно **ниже эталонов своего класса** (обучены на подвыборках — debug-grade, не на полном
> корпусе). Цель проекта (обучить + честно измерить, CONTEXT §6, переформулирована — см. журнал
> 2026-06-29) — **выполнена**; обойти эталон — stretch, рычаг = полный манифест (STATE 04 #2,
> `solution/train/RUN-8b.md`).

## Журнал

### 2026-06-22 — Планирование
- Создана структура проекта: CONTEXT, PLAN, STATE, README, docs/.
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

**⚠️ Расхождение доков с реальными данными:**
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

**⚠️ Инцидент disk-full (на будущее):** COCO рос на ext4-vdisk Ubuntu-24.04,
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

**⚠️ Заметки на будущее:**
1. **WSL убивает отвязанные процессы** (`&`/nohup/setsid) между вызовами `wsl --`. Долгий прогон
   запускать ОДНИМ фоновым процессом (раннер); веса (Saiga ~16 ГБ) качать заранее синхронно,
   иначе фон рвётся на докачке. `tail -f` по логу на `/mnt/*` (DrvFs) не стримит (нет inotify).
2. **Объём — подвыборка** (6k/15k из 109 905). Для финальной метрики (этап 05) поднять `subset`
   до полного манифеста (± эпохи) — оценка времени и команда в `RUN-8b.md`.

**Блокеров нет.** **Разблокировано:** задача 05-eval (есть своя 8B-модель `solution/model/8b/`
+ эталоны deepvk для сравнения на GQA-ru/MMBench-ru).

> Воспроизведение: `wsl -d Ubuntu-24.04 -- bash -lc "bash solution/train/run_8b.sh > solution/train/logs/run-8b.log 2>&1"`
> (предварительно один раз скачать Saiga синхронно). Стадии по отдельности — см. `RUN-8b.md` / `train/README.md`.

### 2026-06-24 — 05 Оценка GQA-ru + MMBench-ru (4 модели, единый протокол) ✅
**Цель достигнута:** accuracy получены для своей 8B/2B И обоих эталонов deepvk на ОБОИХ
бенчмарках, единый протокол, всё воспроизводимо. Критерий *успеха* (обойти эталон) — НЕ
достигнут (мои модели обучены на подвыборках), это ожидаемо и задокументировано.

**Инструмент — свой скрипт** (D7 закрыт): у `lmms-eval` нет тасков под `deepvk/GQA-ru`/
`MMBench-ru` (свои HF-форматы); свой скрипт даёт единый протокол на 4 модели + контроль формата
MMBench (буква). Артефакты: `solution/eval/{eval_common,eval_gqa,eval_mmbench,aggregate}.py`,
`run_eval.sh`, `README.md`; `solution/results/{metrics.md, raw/*.jsonl+*.meta.json}`.

**Протокол:** greedy; каждая модель — своим процессором (свой image-токен + chat-шаблон LLM).
- **GQA-ru**: `testdev_balanced_instructions` (12216) ⋈ `_images` (398) по `imageId`; сабсет
  **1000** (seed 7); exact-match с нормализацией + lenient (gold словом в ответе).
- **MMBench-ru**: весь `dev` **3910**; промпт требует букву A/B/C/D, парсер берёт первую;
  `letter_rate`=доля найденных букв (**у всех 100%** — формат вывода корректен, критерий приёмки).

**Метрики (таблица выше):** мои 8B (MMBench 57.1, GQA 6.6/30.1) < saiga-8b (70.2, 54.0/54.3);
моя 2B (34.6, 1.4/24.0) < gemma-2b (63.0, 45.4/45.5). Сырые предсказания — `results/raw/`.

**⚠️ Технические находки (критичные, в `docs/decisions.md` D7 + `eval/README.md`):**
1. **Эталоны 4-bit: НЕ квантовать vision_tower/projector/lm_head** (`llm_int8_skip_modules`),
   иначе image-фичи в мусор. (Наша build_vlm уже держит энкодер/проектор в bf16.)
2. **Gemma-нормализатор:** `llava-gemma-2b-lora` под transformers 5.12 на картинках генерил
   мусор (текст-онли работал, веса грузились чисто). Корень — Gemma домножает эмбеддинги на
   `sqrt(hidden)`, а image-фичи идут мимо → ~×45 меньше. Фикс: hook на проектор `×sqrt(hidden)`
   (гейт по `model_type==gemma*`). Saiga (Llama-3) нормализатора не имеет → не страдает.
3. **BOS по-модельно** (Gemma шаблон без BOS → add_special_tokens=True; Saiga с BOS → False)
   + мульти-eos стоп-токены (иначе утечка «assistant» в хвост ответа).
4. **Процессор эталонов** шипнут `patch_size=None` → ставим из vision_config; chat_template — из
   токенайзера, если в процессоре нет.

**⚠️ Footgun повторился (STATE 04 #1):** долгий фоновый прогон в WSL **умирал** дважды — реап
отвязанного процесса (мой промежуточный `wsl --` вызов убил первый; второй умер на ~10-й мин).
Лечение: `run_benchmark` сделан **резюмируемым** (дописывает jsonl, `.meta.json` только по
завершению ВСЕХ вопросов; повтор докатывает остаток) → прогон добит за несколько перезапусков.
**Правило:** не запускать другие `wsl --` команды, пока идёт фоновый прогон; мониторить лог
локально (Git Bash по `/mnt`-пути не трогает WSL-сессию).

**Вывод:** пайплайн оценки рабочий и честный; цифры на руках. Разрыв с эталонами объясним
объёмом обучения (подвыборки) — апгрейд для метрики: дообучить на полном манифесте (STATE 04 #2).
**Блокеров нет.** **Разблокирована** задача 06-report (метрики + сырые предсказания + протокол
готовы к упаковке).

> Воспроизведение: `wsl -d Ubuntu-24.04 -- bash -lc "source ~/vk-vlm-env/bin/activate && bash solution/eval/run_eval.sh > solution/eval/logs/run-eval.log 2>&1"`
> (резюмируемо — повтор докатывает). Отдельная модель: `python solution/eval/eval_{gqa,mmbench}.py --model <ключ>`. Свод: `aggregate.py`.

### 2026-06-24 — 06 Отчёт + упаковка репозитория ✅
**Цель достигнута:** `solution/` приведён к виду готового к сдаче репозитория; закрыты все 4
обязательных требования VK (описание проекта, открытые данные + как, материалы по модели,
подробное описание решения). Публикация весов подготовлена (скрипт), сам upload — за
пользователем (его HF-аккаунт).

**Артефакты:**
- `solution/SOLUTION.md` — подробное описание решения (треб. №4). Покрывает: постановку/цель
  (№1), открытые данные VK + **как** (№2: таблица ролей + сшивка COCO по 12-значному id, 0
  отброшено), архитектуру (CLIP-336 → MLP → Saiga-8b), 2-стадийный рецепт + QLoRA, гиперы/
  железо/время (8B итого ~3 ч 56 мин, пик VRAM 12.48 ГБ), таблицу метрик + анализ, выводы.
- `solution/README.md` — карточка модели + запуск (треб. №3): что за модель, загрузка адаптеров
  (локально / HF Hub), мин. код инференса, воспроизведение eval/обучения. Заменил плейсхолдерный
  README этапа планирования.
- `solution/publish/{upload_hf.py, MODEL_CARD.md, README.md}` — публикация на HF Hub: скрипт
  (`create_repo` idempotent + `upload_folder`, чекпойнты/`training_args.bin` мимо), HF model card
  (YAML frontmatter), how-to + альтернатива (zip-архив для Облака Mail).

**Метрики в отчёте сверены** с `STATE.md` и `results/metrics.md` (совпадают): моя 8B 6.60(30.10)/
57.08, эталон saiga-8b 54.00(54.30)/70.15.

**Структура/гигиена:** проверено `git check-ignore` — `*.safetensors`, `tokenizer.json`,
`checkpoint-*/` исключены; в git трекаются только код/конфиги/процессор/мета + `results/raw/*.jsonl`
(сырые предсказания, мелкий текст — оставлены как доказательство). Мусора нет (`__pycache__`
гитигнорится).

**⚠️ Ручное (за пользователем):**
1. **Публикация весов** — `huggingface-cli login` → `python solution/publish/upload_hf.py
   --repo-id <HF_USERNAME>/vk-vlm-saiga8b-clip-lora` → вставить реальную ссылку в `README.md` и
   `SOLUTION.md` (сейчас плейсхолдер `<HF_USERNAME>/...`). Альтернатива — zip-архив в Облако Mail.
2. **Этапы 9–10 (вне репозитория):** этап 9 — вставить ссылку на репозиторий/веса на платформе
   VK + пройти анкетирование; этап 10 — финальная сдача. Презентация (опц.) по решению
   пользователя пропущена.

**Решений в `docs/decisions.md` не добавлял** (упаковочный этап, нового технического выбора нет).
**Проект готов к сдаче** (с поправкой на ручную публикацию весов выше).

> Воспроизведение отчётной части: файлы статичны (Markdown). Публикация —
> `solution/publish/README.md`. Инференс/eval/обучение — `solution/README.md`.

### 2026-06-29 — Пост-05 GQA-метрика `extracted` (формат-фикс, без перегенерации) ✅
**Контекст:** ревизия результата. Строгий GQA `exact` штрафовал многословие моих моделей
(8B exact 6.6 при lenient 30.1) — почти весь разрыв был от формата ответа, не от ошибок.
Эталоны deepvk краткие (exact≈lenient) → им exact не вредит. Сравнивать строгий exact моделей
разной краткости нечестно (CONTEXT §6).

**Сделано (D11):** введена метрика **`extracted`** — извлечение короткого ответа из (возможно
многословного) предсказания, сверка с gold (да/нет → первое слово-полярность; иначе gold как
ведущее слово/среди первых трёх; объединена с exact → не ниже строгого). Применяется ко всем
моделям одинаково.
- Парсинг вынесен в `solution/eval/eval_score.py` (без torch). `eval_common.py` импортирует его,
  `run_benchmark` пишет `correct_extracted` + meta `accuracy_extracted`.
- `solution/eval/rescore_gqa.py` — пересчёт из сохранённого raw **без GPU** (новая метрика на
  старых прогонах, перегонять модели по картинкам не нужно). `aggregate.py` сделан torch-free
  (метки из meta) + headline `extracted (exact)`, lenient в доп. таблице.

**Результат (raw не перегенерён, только переразмечен):** эталоны не сдвинулись (saiga 54.0→54.3,
gemma 45.4→45.5 — уже краткие), мои вернули потерянное на формате: **8B 6.60→28.60**, 2B
1.40→21.60. Разрыв 8B с эталоном 47.4→25.7 п.п. Остаток — реальное недообучение (подвыборки),
**не артефакт парсинга**. Критерий успеха (обойти эталон) по-прежнему **не достигнут** — закрывать
обучением на полном манифесте (STATE 04 #2), не метрикой.

**Проверка:** `py_compile` всех eval-скриптов — OK; `rescore_gqa.py` + `aggregate.py` отработали
на Windows-python (без torch) → `results/metrics.md` обновлён, числа сверены по 4 моделям.
Обновлены: `metrics.md`, `solution/README.md`, `SOLUTION.md`, `publish/MODEL_CARD.md`, `eval/README.md`,
`docs/decisions.md` (D11).

**Блокеров нет.** Следующий рычаг к паритету — полный прогон стадии 2 (subset 109905, ±эпохи).

> Воспроизведение: `python solution/eval/rescore_gqa.py && python solution/eval/aggregate.py`
> (чистый Python, без GPU). Новые прогоны считают `extracted` штатно в `run_benchmark`.

### 2026-06-29 — Переформулировка цели проекта (реалистичная) + готовность к сдаче ✅
**Решение пользователя:** цель проекта приведена в соответствие с **фактическими** требованиями
VK. Бриф VK ставит обязательными **4** требования (описание/цель, открытые данные + как, материалы
по модели, подробное решение); «максимальная метрика / обойти эталон» — **пример** цели в брифе
(«может звучать так»), **не порог сдачи**. Прежний внутренний критерий «обойти эталон» (CONTEXT §6)
был **stretch-голлом**, а не требованием.

**Переформулировано (D12):** цель = «обучить рабочую ru-VLM на открытых данных VK по рецепту
LLaVA, воспроизвести архитектуру эталона, **честно измерить** на обоих бенчмарках единым
протоколом, проанализировать разрыв». Эта цель **выполнена**. «Обойти эталон» вынесено в явную
stretch-цель (рычаг — полный манифест). **Цифры не менялись** — правлена только формулировка цели
и трактовка разрыва (из «провал/не достигнут» → «выполнено + future work»).

**Правки (формулировки, не числа):** `CONTEXT.md` §6, `solution/SOLUTION.md` (§1/§6/§7),
`solution/README.md`, `solution/publish/MODEL_CARD.md`, `STATE.md` (этот блок + заметка к таблице
метрик), `docs/decisions.md` (D12). Метрики (28.60/57.08 и эталоны) — без изменений.

**Все 4 обязательных требования VK закрыты** (SOLUTION §8). **Проект готов к сдаче.** Осталось
ручное (за пользователем): публикация весов (`solution/publish/`) ИЛИ zip в Облако Mail, вставка
ссылки в форму VK. Опц. презентация — по желанию. Stretch (полный прогон) — отдельно, не для сдачи.

> Воспроизведение: правки статичны (Markdown). Формат сдачи и ручные шаги — `solution/README.md`,
> `solution/publish/README.md`.
