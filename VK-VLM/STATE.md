# STATE — трекер прогресса (append-only)

> Каждая агент-сессия в конце работы **дописывает** блок в раздел «Журнал».
> Не переписывать прошлые записи. Таблицу статусов — обновлять.

## Статус задач

| Задача | Промпт | Статус | Артефакты |
|---|---|---|---|
| Планирование (этапы 1–5) | — | ✅ done | CONTEXT.md, PLAN.md, docs/ |
| Окружение | 01-env-setup | ✅ done | solution/env/{check_gpu.py, requirements.txt, README.md, pip-freeze-full.txt} |
| Данные | 02-data-pipeline | ⬜ todo | — |
| Обучение 2B (отладка) | 03-train-2b | ⬜ todo | — |
| Обучение 8B (цель) | 04-train-8b | ⬜ todo | — |
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

> Запуск проверки: `wsl -d Ubuntu-24.04 -- bash -lc "source ~/vk-vlm-env/bin/activate && python /mnt/d/Fork/SimpleVLM/VK-VLM/solution/env/check_gpu.py"`
> (для тренировки данные/код держать в WSL-ФС `~/`, не на `/mnt/*` — медленный I/O).
