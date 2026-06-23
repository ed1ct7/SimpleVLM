#!/usr/bin/env python3
"""Скачивание данных VK-VLM (этап 02).

Делает две вещи:
  1. тянет текстовый датасет `deepvk/LLaVA-Instruct-ru` через `datasets` и сохраняет на диск
     (`<data-root>/datasets/llava-instruct-ru`);
  2. качает и распаковывает изображения COCO в `<data-root>/coco/` (WSL-ФС, НЕ /mnt).

> Нюанс (проверено на данных): поле `image` указывает на `coco/train2017/<id>.jpg`
> (2017-нейминг), но это те же снимки, что лежат в COCO **train2014**. `train2014` (~13 ГБ)
> покрывает все id датасета, поэтому он и качается по умолчанию. Сопоставление имя↔id делает
> `build_dataset.py` по 12-значному id, так что фактический набор COCO (2014 или 2017) не важен —
> можно указать `--coco-splits val2014,train2017,...` при необходимости.

Запуск (в WSL2, активированном venv из этапа 01):
    python solution/data/download.py                      # датасет + COCO train2014
    python solution/data/download.py --coco-splits train2014,val2014
    python solution/data/download.py --skip-coco          # только текст
    python solution/data/download.py --skip-dataset --coco-splits val2014
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from _common import resolve_data_root

COCO_BASE_URL = "http://images.cocodataset.org/zips"
HF_REPO = "deepvk/LLaVA-Instruct-ru"


# --------------------------------------------------------------------------- helpers

def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def download_file(url: str, dest: str) -> None:
    """Скачать `url` в `dest` с докачкой. Сначала curl (-C - резюмирует), иначе urllib."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        print(f"[download] curl {url} -> {dest}")
        # --fail: ненулевой код на 4xx/5xx; -C -: докачка; -L: следовать редиректам
        rc = subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "-C", "-", "-o", dest, url]
        ).returncode
        if rc != 0:
            raise RuntimeError(f"curl завершился с кодом {rc} для {url}")
        return

    print(f"[download] urllib {url} -> {dest} (нет curl, докачка недоступна)")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (доверенный COCO URL)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {_human(done)}/{_human(total)} ({pct:4.1f}%)", end="", flush=True)
        print()
    os.replace(tmp, dest)


def _split_jpg_count(split_dir: str) -> int:
    if not os.path.isdir(split_dir):
        return 0
    n = 0
    with os.scandir(split_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".jpg"):
                n += 1
    return n


def extract_zip(zip_path: str, coco_dir: str, split: str) -> int:
    """Распаковать `split`.zip в `coco_dir` (zip содержит верхнюю папку `split/`)."""
    split_dir = os.path.join(coco_dir, split)
    print(f"[extract] {zip_path} -> {coco_dir}/")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        total = len(members)
        for i, m in enumerate(members, 1):
            zf.extract(m, coco_dir)
            if i % 5000 == 0 or i == total:
                print(f"\r  {i}/{total} файлов", end="", flush=True)
    print()
    return _split_jpg_count(split_dir)


def download_coco_split(coco_dir: str, split: str, base_url: str, keep_zip: bool) -> None:
    split_dir = os.path.join(coco_dir, split)
    have = _split_jpg_count(split_dir)
    if have > 0:
        print(f"[skip] COCO {split}: уже распакован ({have} jpg) -> {split_dir}")
        return
    zip_path = os.path.join(coco_dir, "_zips", f"{split}.zip")
    if not os.path.exists(zip_path):
        download_file(f"{base_url}/{split}.zip", zip_path)
    n = extract_zip(zip_path, coco_dir, split)
    print(f"[ok] COCO {split}: {n} jpg -> {split_dir}")
    if not keep_zip:
        try:
            os.remove(zip_path)
            print(f"[clean] удалён {zip_path}")
        except OSError:
            pass


# --------------------------------------------------------------------------- dataset

def download_dataset(data_root: str, repo: str, split: str) -> None:
    out_dir = os.path.join(data_root, "datasets", "llava-instruct-ru")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"[skip] датасет уже на диске -> {out_dir}")
        return
    # HF-кэш держим тоже в WSL-ФС рядом с данными
    os.environ.setdefault("HF_HOME", os.path.join(data_root, "hf-cache"))
    from datasets import load_dataset  # импорт тут, чтобы --skip-dataset не требовал datasets

    print(f"[dataset] load_dataset({repo!r}, split={split!r}) ...")
    ds = load_dataset(repo, split=split)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    ds.save_to_disk(out_dir)
    print(f"[ok] датасет: {len(ds)} записей -> {out_dir}")


# --------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description="Скачать LLaVA-Instruct-ru + COCO для VK-VLM")
    p.add_argument("--data-root", default=None, help="корень данных (деф. ~/vk-vlm-data)")
    p.add_argument("--hf-repo", default=HF_REPO, help="HF-репозиторий датасета")
    p.add_argument("--split", default="train", help="split датасета (деф. train)")
    p.add_argument(
        "--coco-splits",
        default="train2014",
        help="COCO-сплиты через запятую (деф. train2014; покрывает все id датасета)",
    )
    p.add_argument("--coco-base-url", default=COCO_BASE_URL, help="база URL для zip COCO")
    p.add_argument("--skip-dataset", action="store_true", help="не качать текстовый датасет")
    p.add_argument("--skip-coco", action="store_true", help="не качать изображения COCO")
    p.add_argument("--keep-zips", action="store_true", help="не удалять zip после распаковки")
    args = p.parse_args()

    data_root = resolve_data_root(args.data_root)
    os.makedirs(data_root, exist_ok=True)
    print(f"[data-root] {data_root}")

    if not args.skip_dataset:
        download_dataset(data_root, args.hf_repo, args.split)

    if not args.skip_coco:
        coco_dir = os.path.join(data_root, "coco")
        os.makedirs(coco_dir, exist_ok=True)
        for split in [s.strip() for s in args.coco_splits.split(",") if s.strip()]:
            download_coco_split(coco_dir, split, args.coco_base_url, args.keep_zips)

    print("[done] загрузка завершена. Следующий шаг: build_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
