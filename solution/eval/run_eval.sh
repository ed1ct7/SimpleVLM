#!/usr/bin/env bash
# Полный прогон оценки (этап 05): 4 модели × 2 бенчмарка, ОДНИМ фоновым процессом.
#
#   wsl -d Ubuntu-24.04 -- bash -lc \
#     "source ~/vk-vlm-env/bin/activate && bash solution/eval/run_eval.sh \
#        > solution/eval/logs/run-eval.log 2>&1"
#
# Изоляция по процессам: на каждую (модель × бенчмарк) — отдельный python-процесс. Выход процесса
# = полная очистка VRAM (без накопления фрагментации 4-bit за длинный прогон на 16 ГБ).
# WSL убивает отвязанные процессы между вызовами `wsl --` (STATE 04 заметка #1) — поэтому весь
# цикл идёт ВНУТРИ одного процесса-раннера, а не серией отдельных `wsl --`.
set -euo pipefail

cd "$(dirname "$0")/../.."          # корень репо
GQA_N="${GQA_N:-1000}"              # сабсет GQA (0 = весь testdev-balanced 12216)
MMB_N="${MMB_N:-0}"                 # MMBench (0 = весь dev 3910)
SEED="${SEED:-7}"
# Список моделей переопределяется env EVAL_MODELS (через пробел). Деф. — все 4.
read -r -a MODELS <<< "${EVAL_MODELS:-ref-gemma-2b ref-saiga-8b mine-2b mine-8b}"

mkdir -p solution/results/raw solution/eval/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== eval start $(date -u +%FT%TZ) | GQA_N=$GQA_N MMB_N=$MMB_N seed=$SEED ==="
for m in "${MODELS[@]}"; do
  echo "----- $m | GQA-ru -----"
  python solution/eval/eval_gqa.py     --model "$m" --n "$GQA_N" --seed "$SEED"
  echo "----- $m | MMBench-ru -----"
  python solution/eval/eval_mmbench.py --model "$m" --n "$MMB_N" --seed "$SEED"
done

echo "=== aggregate ==="
python solution/eval/aggregate.py
echo "=== eval done $(date -u +%FT%TZ) ==="
