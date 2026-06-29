#!/usr/bin/env python3
"""Инференс обученной VK-VLM: грузим базу + LoRA-адаптеры (+проектор) и отвечаем по картинке.

Критерий приёмки этапа 03: сохранённые адаптеры грузятся, модель даёт СВЯЗНЫЙ РУССКИЙ ответ
на вопрос по изображению.

Три режима:
  1. Сэмпл из манифеста (по умолчанию) — N случайных примеров из обучающих данных:
       python solution/train/infer.py --adapters solution/model/8b --n 3
  2. Разовый вопрос по своей картинке:
       python solution/train/infer.py --adapters solution/model/8b \
           --image ~/my.jpg --question "Что изображено на картинке?"
  3. Интерактивный чат (вводишь картинку и вопросы, 'exit' — выход):
       python solution/train/infer.py --adapters solution/model/8b --chat

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


def load_vlm(adapters: str):
    """Загрузить базу (детерминированно по training_meta) + адаптеры + процессор.

    Возвращает (model, processor, eos_ids, pad_id, device). eos_ids — все токены конца хода
    (иначе модель эмитит ответ → маркер хода → 'assistant'-заголовок как обычный текст).
    """
    with open(os.path.join(adapters, "training_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    # база: та же сборка, что на обучении → имена модулей совпадут с адаптерами
    model, _, _ = build_vlm(
        meta["encoder_id"], meta["llm_id"],
        load_in_4bit=meta.get("load_in_4bit", True),
        image_size=meta.get("image_size"),
        compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(adapters)  # с <image>-токеном и chat-шаблоном

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapters)
    model.eval()
    model.config.use_cache = True  # ускоряет генерацию (на обучении было False под grad-ckpt)

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            stops.add(i)
    eos_ids = [x for x in stops if x is not None]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, processor, eos_ids, pad_id, device


@torch.no_grad()
def answer(model, processor, eos_ids, pad_id, device, image, question, max_new_tokens=128):
    """Один ответ модели по картинке + русскому вопросу. question — без image-токена."""
    content = f"{IMAGE_TOKEN}\n{question}" if IMAGE_TOKEN not in question else question
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # BOS по-модельно: если chat-шаблон уже вставил BOS (Saiga/Llama-3) → add_special_tokens=False
    # (иначе двойной BOS); если нет (Gemma) → True. Qwen2 BOS не использует → True безвреден.
    bos = processor.tokenizer.bos_token
    add_special = not (bos and prompt.startswith(bos))
    inputs = processor(
        images=[image], text=[prompt], return_tensors="pt", add_special_tokens=add_special
    ).to(device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=pad_id, eos_token_id=eos_ids,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def _open_image(path: str) -> Image.Image:
    return Image.open(os.path.expanduser(path.strip().strip('"').strip("'"))).convert("RGB")


def run_sample(model, processor, eos_ids, pad_id, device, args):
    """N случайных примеров из манифеста (Q берётся из данных) — критерий приёмки."""
    data_root = os.path.expanduser(args.data_root)
    manifest = os.path.expanduser(
        args.manifest or os.path.join(data_root, "processed", "llava_instruct_ru.jsonl")
    )
    rng = random.Random(args.seed)
    lines = []
    with open(manifest, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            lines.append(line)
    samples = [json.loads(x) for x in rng.sample(lines, args.n)]
    for i, rec in enumerate(samples, 1):
        image = _open_image(os.path.join(data_root, rec["image"]))
        q = first_question(rec["messages"])
        question_text = q[0]["content"].replace(IMAGE_TOKEN, "").strip()
        ans = answer(model, processor, eos_ids, pad_id, device, image, question_text,
                     args.max_new_tokens)
        print(f"\n========== пример {i}/{len(samples)} ==========")
        print(f"image : {rec['image']}")
        print(f"вопрос: {question_text}")
        print(f"ответ : {ans}")


def run_chat(model, processor, eos_ids, pad_id, device, args):
    """Интерактив: картинку спрашиваем один раз, дальше — поток вопросов по ней.

    Команда ':img ПУТЬ' меняет картинку; 'exit'/'quit'/'q' — выход.
    """
    print("Интерактив. Картинка задаётся раз; дальше — вопросы по ней.")
    print("  ':img ПУТЬ' — сменить картинку, 'exit' — выход.")

    def ask_image(raw):
        try:
            return _open_image(raw), raw
        except Exception as e:  # noqa: BLE001 — интерактив: печатаем и просим снова
            print(f"  не открыть картинку: {e}")
            return None, None

    image = img_label = None
    while image is None:  # первая картинка обязательна
        raw = input("\nкартинка (путь): ").strip().strip('"').strip("'")
        if raw.lower() in ("exit", "quit", "q"):
            return
        if raw:
            image, img_label = ask_image(raw)

    while True:
        q = input(f"\n[{os.path.basename(img_label)}] вопрос (':img ПУТЬ' / exit): ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        if q.lower().startswith(":img"):  # сменить картинку
            raw = q[4:].strip().strip('"').strip("'")
            if not raw:
                print("  укажи путь: :img /mnt/c/...")
                continue
            new_img, new_label = ask_image(raw)
            if new_img is not None:
                image, img_label = new_img, new_label
                print(f"  картинка → {img_label}")
            continue
        ans = answer(model, processor, eos_ids, pad_id, device, image, q, args.max_new_tokens)
        print(f"  ответ: {ans}")


def main() -> int:
    p = argparse.ArgumentParser(description="Инференс VK-VLM (адаптеры + проектор)")
    p.add_argument("--adapters", default="solution/model/8b")
    p.add_argument("--image", default=None, help="путь к своей картинке (разовый вопрос)")
    p.add_argument("--question", default=None, help="свой вопрос к --image (по-русски)")
    p.add_argument("--chat", action="store_true", help="интерактивный режим (картинка+вопросы)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--data-root", default="~/vk-vlm-data")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    model, processor, eos_ids, pad_id, device = load_vlm(args.adapters)

    if args.image:  # разовый вопрос по своей картинке
        q = args.question or "Что изображено на картинке?"
        image = _open_image(args.image)
        ans = answer(model, processor, eos_ids, pad_id, device, image, q, args.max_new_tokens)
        print(f"image : {args.image}")
        print(f"вопрос: {q}")
        print(f"ответ : {ans}")
    elif args.chat:
        run_chat(model, processor, eos_ids, pad_id, device, args)
    else:  # сэмпл из манифеста (критерий приёмки)
        run_sample(model, processor, eos_ids, pad_id, device, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
