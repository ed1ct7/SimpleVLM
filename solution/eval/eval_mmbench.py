#!/usr/bin/env python3
"""Оценка одной модели на MMBench-ru (single-choice A/B/C/D, accuracy по букве).

    python solution/eval/eval_mmbench.py --model mine-8b --n 0

Промпт ОБЯЗАН заставлять модель выдать букву (A/B/C/D), парсер берёт первую A–D из ответа.
`letter_rate` в .meta.json фиксирует долю ответов, где буква реально найдена (критерий приёмки:
проверить, что модель выдаёт букву, а не свободный текст).

Ключи модели: mine-8b, mine-2b, ref-saiga-8b, ref-gemma-2b (реестр в eval_common.MODELS).
Сырые предсказания → solution/results/raw/mmbench__<model>.jsonl (+ .meta.json).
"""
from __future__ import annotations

import argparse

import eval_common as E


def main() -> int:
    p = argparse.ArgumentParser(description="MMBench-ru eval (single model)")
    p.add_argument("--model", required=True, choices=list(E.MODELS))
    p.add_argument("--n", type=int, default=0, help="сабсет (0 = весь dev 3910)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--out-dir", default=str(E.REPO_ROOT / "solution" / "results" / "raw"))
    args = p.parse_args()

    items = E.load_mmbench(args.n, args.seed)
    print(f"[mmbench] вопросов: {len(items)} (n={args.n or 'all'}, seed={args.seed})", flush=True)
    vlm = E.load_model(args.model)
    out = f"{args.out_dir}/mmbench__{args.model}.jsonl"
    E.run_benchmark(vlm, "mmbench", items, out, max_new_tokens=args.max_new_tokens)
    E.free(vlm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
