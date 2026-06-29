#!/usr/bin/env python3
"""Построить SQLite-базу из результатов оценки — без GPU/torch, только stdlib.

Загружает сводки прогонов (`results/raw/*.meta.json`) и сырые предсказания
(`results/raw/*.jsonl`) в нормализованную реляционную схему (`schema.sql`) и наполняет
справочники. Результат — `solution/db/vk_vlm.sqlite`. Идемпотентен (DROP+пересоздание).

    python solution/db/build_db.py
    # затем: sqlite3 solution/db/vk_vlm.sqlite < solution/db/queries.sql
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent
RAW = DB_DIR.parents[0] / "results" / "raw"
DB_PATH = DB_DIR / "vk_vlm.sqlite"

BENCHMARKS = {"gqa": "GQA-ru (открытые вопросы)", "mmbench": "MMBench-ru (выбор A/B/C/D)"}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
    cur = con.cursor()

    metas = [json.load(open(p, encoding="utf-8")) for p in glob.glob(str(RAW / "*.meta.json"))]

    # справочник моделей
    seen = {}
    for m in metas:
        seen[m["model"]] = (m["label"], 0 if m["model"].startswith("ref-") else 1)
    cur.executemany("INSERT OR REPLACE INTO model VALUES (?,?,?)",
                    [(k, v[0], v[1]) for k, v in seen.items()])

    # справочник бенчмарков (n_total — из любого прогона этого бенчмарка)
    n_by_bench = {}
    for m in metas:
        n_by_bench[m["benchmark"]] = m["n"]
    cur.executemany("INSERT OR REPLACE INTO benchmark VALUES (?,?,?)",
                    [(b, BENCHMARKS.get(b, b), n_by_bench.get(b)) for b in n_by_bench])

    # факт-таблица прогонов
    cur.executemany(
        "INSERT OR REPLACE INTO eval_run VALUES (?,?,?,?,?,?,?,?)",
        [(m["model"], m["benchmark"], m["n"], m.get("accuracy"),
          m.get("accuracy_extracted"), m.get("accuracy_lenient"),
          m.get("letter_rate"), m.get("seconds")) for m in metas])

    # предсказания
    n_gqa = n_mmb = 0
    for path in glob.glob(str(RAW / "gqa__*.jsonl")):
        model_key = os.path.basename(path)[len("gqa__"):-len(".jsonl")]
        rows = [(model_key, r["id"], r.get("imageId"), r.get("question"), r.get("gold"),
                 r.get("pred"), int(bool(r.get("correct"))),
                 int(bool(r.get("correct_extracted")))) for r in _read_jsonl(path)]
        cur.executemany("INSERT OR REPLACE INTO gqa_prediction VALUES (?,?,?,?,?,?,?,?)", rows)
        n_gqa += len(rows)
    for path in glob.glob(str(RAW / "mmbench__*.jsonl")):
        model_key = os.path.basename(path)[len("mmbench__"):-len(".jsonl")]
        rows = [(model_key, r["index"], r.get("category"), r.get("question"), r.get("gold"),
                 r.get("pred_letter"), int(bool(r.get("correct")))) for r in _read_jsonl(path)]
        cur.executemany("INSERT OR REPLACE INTO mmbench_prediction VALUES (?,?,?,?,?,?,?)", rows)
        n_mmb += len(rows)

    con.commit()
    print(f"[db] {DB_PATH}")
    print(f"[db] models={len(seen)} benchmarks={len(n_by_bench)} eval_runs={len(metas)} "
          f"gqa_pred={n_gqa} mmbench_pred={n_mmb}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
