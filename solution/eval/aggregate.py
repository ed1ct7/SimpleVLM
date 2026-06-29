#!/usr/bin/env python3
"""Свести .meta.json всех прогонов в таблицу solution/results/metrics.md.

    python solution/eval/aggregate.py

Читает solution/results/raw/*.meta.json (авторитетный источник accuracy) и строит
сравнительную таблицу модель × бенчмарк. Без torch (только stdlib) → гоняется где угодно;
метки моделей берутся из самих meta. Порядок строк — лёгкие → тяжёлые эталоны/свои.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROW_ORDER = ["ref-gemma-2b", "ref-saiga-8b", "mine-2b", "mine-8b"]


def main() -> int:
    raw_dir = REPO_ROOT / "solution" / "results" / "raw"
    metas = {}
    for p in glob.glob(str(raw_dir / "*.meta.json")):
        m = json.load(open(p, encoding="utf-8"))
        metas[(m["model"], m["benchmark"])] = m

    def label(model):
        for bench in ("gqa", "mmbench"):
            m = metas.get((model, bench))
            if m:
                return m.get("label", model)
        return model

    def cell(model, bench):
        m = metas.get((model, bench))
        if not m:
            return "—"
        if bench == "gqa":
            # extracted (короткий ответ из многословного, честно между моделями разной
            # краткости — D11) в заголовке; exact (стандарт GQA, штрафует многословие) в скобках.
            ext = m.get("accuracy_extracted", m["accuracy"])
            return f"{ext*100:.2f} ({m['accuracy']*100:.2f})"
        s = f"{m['accuracy']*100:.2f}"
        if m.get("letter_rate", 1.0) < 0.999:
            s += f" _(буква {m['letter_rate']*100:.1f}%)_"
        return s

    gqa_n = next((m["n"] for (mk, b), m in metas.items() if b == "gqa"), "—")
    mmb_n = next((m["n"] for (mk, b), m in metas.items() if b == "mmbench"), "—")

    lines = [
        "# Метрики оценки — GQA-ru + MMBench-ru (этап 05)",
        "",
        "Accuracy (%). Единый протокол: greedy-декодирование, свой процессор/chat-шаблон у каждой",
        "модели, общий текст задачи и метрика. Протокол и воспроизведение — `solution/eval/README.md`.",
        "",
        f"- **GQA-ru**: testdev_balanced, exact-match с нормализацией ответа. N = **{gqa_n}** вопросов.",
        f"- **MMBench-ru**: dev, single-choice, accuracy по распарсенной букве. N = **{mmb_n}** вопросов.",
        "",
        "GQA-ячейка: **extracted (exact)**. `extracted` — короткий ответ, извлечённый из",
        "(возможно многословного) ответа: для да/нет — первое слово-полярность, иначе gold как",
        "ведущее слово/среди первых трёх (D11). Применяется ко всем моделям одинаково; эталоны уже",
        "краткие → не меняются, многословные мои модели возвращают потерянные совпадения. `exact` —",
        "строгий стандарт GQA (штрафует многословие). `lenient` (доп. таблица) — потолок: gold где",
        "угодно в ответе.",
        "",
        "| Модель | GQA-ru extracted (exact) | MMBench-ru (acc) |",
        "|---|---|---|",
    ]
    for model in ROW_ORDER:
        lines.append(f"| {label(model)} | {cell(model,'gqa')} | {cell(model,'mmbench')} |")

    # вспомогательные показатели
    lines += ["", "## Доп. показатели", ""]
    lines.append("| Прогон | n | exact | extracted | lenient / доп |")
    lines.append("|---|---|---|---|---|")
    for model in ROW_ORDER:
        for bench in ("gqa", "mmbench"):
            m = metas.get((model, bench))
            if not m:
                continue
            if bench == "gqa":
                ext = f"{m.get('accuracy_extracted', m['accuracy'])*100:.2f}%"
                extra = f"lenient={m.get('accuracy_lenient', 0)*100:.2f}%"
            else:
                ext = "—"
                extra = f"letter_rate={m.get('letter_rate', 0)*100:.1f}%, no_letter={m.get('n_no_letter', 0)}"
            lines.append(f"| {m['label']} · {bench} | {m['n']} | "
                         f"{m['accuracy']*100:.2f}% | {ext} | {extra} |")

    out = REPO_ROOT / "solution" / "results" / "metrics.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[aggregate] {len(metas)} прогонов → {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
