#!/usr/bin/env python3
"""Пересчитать метрики GQA из сохранённых сырых предсказаний — БЕЗ GPU/torch.

Зачем: метрика `extracted` (короткий ответ из многословного, D11) добавлена после прогонов.
Перегонять 4 модели по картинкам ради новой метрики (~10 мин/модель на GPU) не нужно —
сырые предсказания (`pred`/`gold`) уже на диске. Скрипт перечитывает их, проставляет
`correct_extracted` в каждую запись и `accuracy_extracted` в `.meta.json`.

Зависит только от `eval_score` (чистый Python) → гоняется где угодно, в т.ч. Windows-python:

    python solution/eval/rescore_gqa.py            # все results/raw/gqa__*.jsonl
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from eval_score import gqa_correct

RAW_DIR = Path(__file__).resolve().parents[2] / "solution" / "results" / "raw"


def rescore_file(path: str):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    ne = nl = nx = 0
    for r in recs:
        e, l, x = gqa_correct(r["pred"], r["gold"])
        r["correct"], r["correct_lenient"], r["correct_extracted"] = e, l, x
        ne += e; nl += l; nx += x
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(recs)
    meta_path = path.replace(".jsonl", ".meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    meta["accuracy"] = round(ne / n, 4) if n else 0.0
    meta["accuracy_lenient"] = round(nl / n, 4) if n else 0.0
    meta["accuracy_extracted"] = round(nx / n, 4) if n else 0.0
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return n, ne, nl, nx


def main() -> int:
    files = sorted(glob.glob(str(RAW_DIR / "gqa__*.jsonl")))
    if not files:
        print(f"[rescore] нет gqa__*.jsonl в {RAW_DIR}")
        return 1
    for path in files:
        n, ne, nl, nx = rescore_file(path)
        name = os.path.basename(path).replace("gqa__", "").replace(".jsonl", "")
        print(f"{name:<14} n={n:>4}  exact={100*ne/n:6.2f}  "
              f"lenient={100*nl/n:6.2f}  extracted={100*nx/n:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
