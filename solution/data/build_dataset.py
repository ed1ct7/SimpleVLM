#!/usr/bin/env python3
"""Сборка обучающего набора VK-VLM (этап 02).

По полю `image` каждой записи `LLaVA-Instruct-ru` находит файл картинки COCO (резолвер по
12-значному id — устойчив к 2014/2017-неймингу), отбрасывает записи без файла (с логом),
приводит `conversations` к chat-формату и пишет манифест JSONL + статистику.

Манифест (одна запись на строку):
    {"id": "...", "type": "...", "image": "coco/train2014/COCO_train2014_<id>.jpg",
     "messages": [...]}
`image` хранится **относительно** data-root → манифест переносим.

Запуск:
    python solution/data/build_dataset.py
    python solution/data/build_dataset.py --template text --image-token "<image>"
    python solution/data/build_dataset.py --limit 100        # быстрый прогон
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from _common import (
    TEMPLATES,
    build_coco_index,
    extract_image_id,
    resolve_data_root,
    to_chat,
)


def load_records(data_root: str, dataset_dir: str | None, hf_repo: str, split: str):
    """Загрузить датасет: сперва с диска (save_to_disk из download.py), иначе из HF."""
    ds_dir = dataset_dir or os.path.join(data_root, "datasets", "llava-instruct-ru")
    os.environ.setdefault("HF_HOME", os.path.join(data_root, "hf-cache"))
    if os.path.isdir(ds_dir) and os.listdir(ds_dir):
        from datasets import load_from_disk

        print(f"[dataset] load_from_disk({ds_dir})")
        return load_from_disk(ds_dir)
    from datasets import load_dataset

    print(f"[dataset] {ds_dir} пуст — load_dataset({hf_repo!r}, split={split!r})")
    return load_dataset(hf_repo, split=split)


def main() -> int:
    p = argparse.ArgumentParser(description="Собрать (картинка, диалог) набор для VK-VLM")
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset-dir", default=None, help="каталог save_to_disk (деф. под data-root)")
    p.add_argument("--hf-repo", default="deepvk/LLaVA-Instruct-ru")
    p.add_argument("--split", default="train")
    p.add_argument("--coco-dir", default=None, help="каталог COCO (деф. <data-root>/coco)")
    p.add_argument("--out", default=None, help="выходной JSONL (деф. <data-root>/processed/...)")
    p.add_argument("--template", choices=TEMPLATES, default="messages", help="chat-шаблон")
    p.add_argument("--image-token", default="<image>", help="плейсхолдер картинки для --template text")
    p.add_argument("--limit", type=int, default=0, help="ограничить число записей (0 = все)")
    args = p.parse_args()

    data_root = resolve_data_root(args.data_root)
    coco_dir = args.coco_dir or os.path.join(data_root, "coco")
    out_path = args.out or os.path.join(data_root, "processed", "llava_instruct_ru.jsonl")
    print(f"[data-root] {data_root}")
    print(f"[coco-dir]  {coco_dir}")

    print("[index] сканирую COCO ...")
    index = build_coco_index(coco_dir, data_root=data_root)
    print(f"[index] {len(index)} картинок в индексе")
    if not index:
        print(
            f"[error] в {coco_dir} нет картинок. Сначала запусти download.py "
            f"(или укажи --coco-dir).",
            file=sys.stderr,
        )
        return 2

    records = load_records(data_root, args.dataset_dir, args.hf_repo, args.split)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = matched = dropped = 0
    dropped_examples: list[str] = []

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            if args.limit and total >= args.limit:
                break
            total += 1
            img_field = rec.get("image")
            img_id = extract_image_id(img_field)
            rel = index.get(img_id) if img_id else None
            if not rel:
                dropped += 1
                if len(dropped_examples) < 10:
                    dropped_examples.append(str(img_field))
                continue
            messages = to_chat(
                rec.get("conversations") or [],
                template=args.template,
                image_token=args.image_token,
            )
            f.write(
                json.dumps(
                    {
                        "id": rec.get("id"),
                        "type": rec.get("type"),
                        "image": rel,
                        "messages": messages,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            matched += 1

    stats = {
        "dataset": args.hf_repo,
        "split": args.split,
        "template": args.template,
        "total": total,
        "with_valid_image": matched,
        "dropped": dropped,
        "coco_indexed": len(index),
        "out": os.path.relpath(out_path, data_root).replace(os.sep, "/"),
    }
    stats_path = os.path.join(os.path.dirname(out_path), "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== СТАТИСТИКА ===")
    print(f"  всего записей      : {total}")
    print(f"  с валидной картинкой: {matched}")
    print(f"  отброшено (нет файла): {dropped}")
    if dropped_examples:
        print(f"  примеры отброшенных image: {dropped_examples}")
    print(f"  манифест: {out_path}")
    print(f"  статистика: {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
