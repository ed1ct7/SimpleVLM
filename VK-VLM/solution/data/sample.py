#!/usr/bin/env python3
"""Проверка собранного набора VK-VLM (этап 02).

Берёт N случайных примеров из манифеста (`build_dataset.py`), печатает текст диалога и
подтверждает, что картинка реально открывается через PIL и её размеры > 0.

Код возврата 0 — все картинки открылись; иначе 1 (есть битые/отсутствующие).

Запуск:
    python solution/data/sample.py
    python solution/data/sample.py --n 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

from _common import render_messages_text, resolve_data_root

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("[error] нет Pillow. Установи: pip install pillow", file=sys.stderr)
    sys.exit(2)


def reservoir_sample(path: str, n: int, rng: random.Random) -> list[dict]:
    """Случайные n строк JSONL за один проход (reservoir), без загрузки всего файла в память."""
    chosen: list[dict] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(chosen) < n:
                chosen.append(json.loads(line))
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    chosen[j] = json.loads(line)
    return chosen


def main() -> int:
    p = argparse.ArgumentParser(description="Показать N примеров и проверить картинки (PIL)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--manifest", default=None, help="JSONL (деф. <data-root>/processed/...)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    data_root = resolve_data_root(args.data_root)
    manifest = args.manifest or os.path.join(data_root, "processed", "llava_instruct_ru.jsonl")
    if not os.path.exists(manifest):
        print(
            f"[error] нет манифеста {manifest}. Сначала запусти build_dataset.py.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    samples = reservoir_sample(manifest, args.n, rng)
    if not samples:
        print(f"[error] манифест {manifest} пуст.", file=sys.stderr)
        return 2

    ok = 0
    for i, rec in enumerate(samples, 1):
        print(f"\n========== пример {i}/{len(samples)} ==========")
        print(f"id={rec.get('id')}  type={rec.get('type')}")
        print(f"image={rec.get('image')}")
        print(render_messages_text(rec.get("messages") or []))

        img_rel = rec.get("image")
        img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(data_root, img_rel)
        try:
            with Image.open(img_abs) as im:
                im.load()  # реально декодировать (поймать обрезанные файлы)
                w, h = im.size
            if w > 0 and h > 0:
                print(f"[ok] картинка открыта: {w}x{h}, format={im.format}")
                ok += 1
            else:
                print(f"[FAIL] нулевые размеры: {img_abs}")
        except Exception as e:  # noqa: BLE001 — нужно отчитаться по любой картинке
            print(f"[FAIL] не открылась {img_abs}: {e}")

    print(f"\n=== ИТОГ: {ok}/{len(samples)} картинок открылись ===")
    return 0 if ok == len(samples) else 1


if __name__ == "__main__":
    sys.exit(main())
