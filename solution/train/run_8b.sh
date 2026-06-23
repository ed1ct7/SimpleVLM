#!/usr/bin/env bash
# Целевое обучение VK-VLM 8B (Saiga-8b) — 2-стадийный рецепт LLaVA, один прогон (задача 04).
# Запускать ОДНИМ фоновым процессом (WSL убивает отвязанные процессы между вызовами `wsl --`):
#   wsl -d Ubuntu-24.04 -- bash -lc "bash solution/train/run_8b.sh > solution/train/logs/run-8b.log 2>&1"
# Стадия 2 чекпойнтит адаптеры каждые 250 шагов → при обрыве: добавить `--resume` к её запуску.
set -euo pipefail

source ~/vk-vlm-env/bin/activate
cd /mnt/d/Fork/SimpleVLM
export HF_HOME=~/vk-vlm-data/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # меньше фрагментации у потолка 16 ГБ

ts(){ date '+%F %T'; }

echo "[$(ts)] ===== STAGE 1 (alignment: только проектор, энкодер+LLM заморожены) START ====="
python -u solution/train/train.py --config solution/train/configs/8b-stage1.yaml
echo "[$(ts)] ===== STAGE 1 DONE (проектор → solution/model/8b-stage1) ====="

echo "[$(ts)] ===== STAGE 2 (instruction tuning: QLoRA на LLM + проектор) START ====="
python -u solution/train/train.py --config solution/train/configs/8b-stage2.yaml
echo "[$(ts)] ===== STAGE 2 DONE (адаптеры+проектор → solution/model/8b) ====="

echo "[$(ts)] ===== INFERENCE TEST (3 примера, русский ответ по картинке) ====="
python -u solution/train/infer.py --adapters solution/model/8b --n 3 --max-new-tokens 96
echo "[$(ts)] ===== ALL DONE ====="
