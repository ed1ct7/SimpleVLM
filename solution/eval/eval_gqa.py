#!/usr/bin/env python3
"""Оценка одной модели на GQA-ru (открытые вопросы, exact-match accuracy).

    python solution/eval/eval_gqa.py --model mine-8b --n 1000 --seed 7

Ключи модели: mine-8b, mine-2b, ref-saiga-8b, ref-gemma-2b (реестр в eval_common.MODELS).
Сырые предсказания → solution/results/raw/gqa__<model>.jsonl (+ .meta.json со сводкой).
"""
from __future__ import annotations

import argparse

import eval_common as E


def main() -> int:
    p = argparse.ArgumentParser(description="GQA-ru eval (single model)")
    p.add_argument("--model", required=True, choices=list(E.MODELS))
    p.add_argument("--n", type=int, default=1000, help="сабсет (0 = весь testdev-balanced 12216)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--out-dir", default=str(E.REPO_ROOT / "solution" / "results" / "raw"))
    args = p.parse_args()

    items = E.load_gqa(args.n, args.seed)
    print(f"[gqa] вопросов: {len(items)} (n={args.n}, seed={args.seed})", flush=True)
    vlm = E.load_model(args.model)
    out = f"{args.out_dir}/gqa__{args.model}.jsonl"
    E.run_benchmark(vlm, "gqa", items, out, max_new_tokens=args.max_new_tokens)
    E.free(vlm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
