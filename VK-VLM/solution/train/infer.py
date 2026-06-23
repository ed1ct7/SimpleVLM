#!/usr/bin/env python3
"""Инференс обученной VK-VLM: грузим базу + LoRA-адаптеры (+проектор) и отвечаем по картинке.

Критерий приёмки этапа 03: сохранённые адаптеры грузятся, модель даёт СВЯЗНЫЙ РУССКИЙ ответ
на вопрос по изображению.

    python solution/train/infer.py --adapters solution/model/2b-debug --n 3

Базовые id энкодера/LLM берутся из `training_meta.json` в каталоге адаптеров.
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", os.path.expanduser("~/vk-vlm-data/hf-cache"))

import torch
from PIL import Image
from transformers import AutoProcessor

from data_collator import flatten_messages
from model import IMAGE_TOKEN, build_vlm


def first_question(messages):
    """Оставить первый user-турн (с картинкой) — для чистого Q→A инференса."""
    flat = flatten_messages(messages)
    for m in flat:
        if m["role"] == "user":
            return [m]
    return flat[:1]


def main() -> int:
    p = argparse.ArgumentParser(description="Инференс VK-VLM (адаптеры + проектор)")
    p.add_argument("--adapters", default="solution/model/2b-debug")
    p.add_argument("--manifest", default=None)
    p.add_argument("--data-root", default="~/vk-vlm-data")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    data_root = os.path.expanduser(args.data_root)
    manifest = os.path.expanduser(
        args.manifest or os.path.join(data_root, "processed", "llava_instruct_ru.jsonl")
    )
    with open(os.path.join(args.adapters, "training_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # база: та же сборка, что на обучении (детерминированно) → имена модулей совпадут с адаптерами
    model, _, _ = build_vlm(
        meta["encoder_id"], meta["llm_id"],
        load_in_4bit=meta.get("load_in_4bit", True),
        image_size=meta.get("image_size"),
        compute_dtype=torch.bfloat16,
    )
    # процессор (с <image>-токеном и chat-шаблоном) — из каталога адаптеров
    processor = AutoProcessor.from_pretrained(args.adapters)

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapters)
    model.eval()

    # N случайных примеров
    rng = random.Random(args.seed)
    lines = []
    with open(manifest, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            lines.append(line)
    samples = [json.loads(x) for x in rng.sample(lines, args.n)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, rec in enumerate(samples, 1):
        img_path = os.path.join(data_root, rec["image"])
        image = Image.open(img_path).convert("RGB")
        q = first_question(rec["messages"])
        prompt = processor.apply_chat_template(q, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=[prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        answer = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

        question_text = q[0]["content"].replace(IMAGE_TOKEN, "").strip()
        print(f"\n========== пример {i}/{len(samples)} ==========")
        print(f"image: {rec['image']}")
        print(f"вопрос: {question_text}")
        print(f"ответ : {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
