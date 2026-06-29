#!/usr/bin/env python3
"""Общая библиотека оценки (этап 05): GQA-ru + MMBench-ru, единый протокол для 4 моделей.

Один протокол на все модели (CONTEXT §6, честное сравнение):
  - своя 8B (`solution/model/8b`, PEFT-адаптеры поверх CLIP+Saiga), своя 2B (`2b-debug`),
  - эталоны deepvk: `llava-saiga-8b`, `llava-gemma-2b-lora` (оба — готовые
    `LlavaForConditionalGeneration`, грузятся напрямую `from_pretrained`).

Каждая модель рендерится СВОИМ процессором: свой `image_token` и свой chat-шаблон LLM
(Llama-3 / Gemma / Qwen2). Это корректно — навязывать чужой шаблон было бы нечестно. Едины:
текст задачи, набор/сабсет, параметры генерации (greedy), метрика и парсинг ответа.

Память (RTX 5080 16 ГБ): база 4-bit (nf4 + double-quant, compute bf16). Одна модель за раз —
изоляция по процессам (см. `run_eval.sh`): процесс на (модель × бенчмарк), выход = полная
очистка VRAM, без накопления фрагментации за длинный прогон.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# HF_HOME → WSL-ФС (веса ~22 ГБ; иначе C:, см. инцидент disk-full STATE 02). До импорта torch.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/vk-vlm-data/hf-cache"))
# Hub оставляем онлайн: бенчмарки берутся из кэша (быстрый HEAD-фолбэк), а веса эталонов при
# первом обращении докачиваются. HF_DATASETS_OFFLINE НЕ ставим — ломает резолв MMBench-ru (parquet).

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "solution" / "train"))  # build_vlm, IMAGE_TOKEN

# Скоринг/парсинг — в отдельном модуле без torch (общий с rescore_gqa.py).
from eval_score import gqa_correct, norm_answer, parse_letter  # noqa: E402,F401

# --------------------------------------------------------------------------- реестр моделей

MODELS = {
    "mine-8b":      {"kind": "peft",   "adapters": "solution/model/8b",         "label": "Моя 8B"},
    "mine-2b":      {"kind": "peft",   "adapters": "solution/model/2b-debug",   "label": "Моя 2B"},
    "ref-saiga-8b": {"kind": "native", "repo": "deepvk/llava-saiga-8b",         "label": "deepvk/llava-saiga-8b"},
    "ref-gemma-2b": {"kind": "native", "repo": "deepvk/llava-gemma-2b-lora",    "label": "deepvk/llava-gemma-2b-lora"},
}


def _bnb4bit(compute_dtype=torch.bfloat16):
    # skip_modules: НЕ квантовать визуальную башню и проектор — иначе 4-bit ломает image-фичи
    # (эталон выдаёт мусор). Наша build_vlm уже держит энкодер+проектор в bf16 — паритет.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
    )


def _native_from_pretrained(repo, **kw):
    """from_pretrained устойчив к torch_dtype→dtype (transformers 5.x)."""
    try:
        return LlavaForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16, **kw)
    except TypeError:
        return LlavaForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw)


class VLM:
    """Обёртка-адаптер: единый `.answer(image, user_text, max_new_tokens)` для любой из 4 моделей."""

    def __init__(self, model, processor, image_token, name):
        self.model = model
        self.processor = processor
        self.image_token = image_token
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        # Стоп-токены конца хода: иначе модель эмитит ответ → маркер хода → "assistant"-заголовок
        # как обычный текст (утечка в GQA: "...книги.assistant"). Собираем все, что есть в словаре.
        stops = {tok.eos_token_id}
        for t in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|end_of_text|>"):
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i != tok.unk_token_id:
                stops.add(i)
        self.eos_ids = [x for x in stops if x is not None]

    @torch.no_grad()
    def answer(self, image: Image.Image, user_text: str, max_new_tokens: int = 16) -> str:
        # image-токен в начале (как на обучении: flatten_messages префиксует <image>)
        content = f"{self.image_token}\n{user_text}"
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # BOS — по-модельно: если chat-шаблон уже вставил BOS (Llama-3/Saiga: `<|begin_of_text|>`),
        # add_special_tokens=False (иначе ДВОЙНОЙ BOS). Если шаблон BOS не ставит (Gemma:
        # начинается с `<start_of_turn>`) — add_special_tokens=True, иначе модель без BOS выдаёт
        # мусор. Qwen2 BOS не использует (bos_token=None) → True безвреден.
        bos = self.processor.tokenizer.bos_token
        add_special = not (bos and prompt.startswith(bos))
        inputs = self.processor(
            images=[image], text=[prompt], return_tensors="pt", add_special_tokens=add_special
        ).to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.pad_id, eos_token_id=self.eos_ids,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def _image_token_of(processor, model) -> str:
    """Строка image-токена данной модели (свой токен у каждого процессора)."""
    tok_str = getattr(processor, "image_token", None)
    if isinstance(tok_str, str) and tok_str:
        return tok_str
    idx = getattr(model.config, "image_token_index", None)
    if idx is not None:
        s = processor.tokenizer.convert_ids_to_tokens(int(idx))
        if s:
            return s
    return "<image>"


def load_model(key: str) -> VLM:
    """Загрузить одну из 4 моделей по ключу реестра. Возвращает VLM-обёртку."""
    spec = MODELS[key]
    if spec["kind"] == "native":
        repo = spec["repo"]
        model = _native_from_pretrained(
            repo, quantization_config=_bnb4bit(),
            device_map={"": 0} if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(repo)
        # deepvk-эталоны шипнуты с patch_size=None у процессора → LlavaProcessor падает на
        # подсчёте image-токенов. Берём patch из vision_config. num_additional_image_tokens=0 и
        # strategy=None оставляем как есть: (336//14)^2 + 0 = 576 = cfg.image_seq_length. НЕ
        # выставляем strategy="default" — в этой версии она включила бы вычет CLS (→575, mismatch).
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = model.config.vision_config.patch_size
        # Gemma масштабирует входные эмбеддинги на sqrt(hidden_size) (нормализатор), а LLaVA
        # вставляет image-фичи в inputs_embeds СЫРЫМИ (мимо нормализатора). В transformers 5.12
        # merge не домножает image-фичи → они ~×45 меньше текстовых → модель их игнорирует →
        # генерит мусор (китайский/тагальский). Llama-3/Saiga нормализатора не имеет → не страдает.
        # Лечим хуком: домножаем выход проектора на sqrt(hidden_size). Проверено: gemma выдаёт
        # связные ответы по картинке. Гейтим строго по text_config.model_type == gemma*.
        import math
        if str(getattr(model.config.text_config, "model_type", "")).startswith("gemma"):
            scale = math.sqrt(model.config.text_config.hidden_size)
            model.model.multi_modal_projector.register_forward_hook(
                lambda _m, _i, out: out * scale
            )
            print(f"[load] gemma-фикс: image-фичи ×sqrt(hidden)={scale:.2f}", flush=True)
    else:  # peft (своя модель): детерминированная пересборка базы + адаптеры (как infer.py)
        from model import build_vlm  # из solution/train
        adapters = str(REPO_ROOT / spec["adapters"])
        with open(os.path.join(adapters, "training_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        model, _, _ = build_vlm(
            meta["encoder_id"], meta["llm_id"],
            load_in_4bit=meta.get("load_in_4bit", True),
            image_size=meta.get("image_size"),
            compute_dtype=torch.bfloat16,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters)
        processor = AutoProcessor.from_pretrained(adapters)

    # Некоторые эталонные процессоры (gemma) сохранены без chat-шаблона → берём из токенайзера.
    if getattr(processor, "chat_template", None) is None:
        processor.chat_template = processor.tokenizer.chat_template
        print(f"[load] chat_template взят из токенайзера ({key})", flush=True)

    model.eval()
    model.config.use_cache = True  # ускоряет генерацию (на обучении было False под grad-ckpt)
    image_token = _image_token_of(processor, model)
    print(f"[load] {key} ({spec['label']}) | image_token={image_token!r}", flush=True)
    return VLM(model, processor, image_token, key)


def free(vlm: VLM):
    del vlm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- загрузка бенчмарков

def load_gqa(n: int, seed: int):
    """GQA-ru testdev_balanced: join инструкций (12216) и изображений (398) по imageId.

    Возвращает список dict: {id, imageId, question, answer, image(PIL)}. n<=0 → весь набор.
    """
    import random
    from datasets import load_dataset

    instr = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev")
    imgs = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev")
    img_by_id = {r["id"]: r["image"] for r in imgs}

    idx = list(range(len(instr)))
    if n and n > 0 and n < len(idx):
        idx = random.Random(seed).sample(idx, n)
        idx.sort()
    items, miss = [], 0
    for i in idx:
        r = instr[i]
        im = img_by_id.get(r["imageId"])
        if im is None:
            miss += 1
            continue
        items.append({"id": r["id"], "imageId": r["imageId"],
                      "question": r["question"], "answer": r["answer"], "image": im})
    if miss:
        print(f"[gqa][warn] {miss} вопросов без картинки (пропущены)", flush=True)
    return items


def load_mmbench(n: int, seed: int):
    """MMBench-ru dev (3910): single-choice. Возвращает list dict с полями вопроса+вариантов.

    n<=0 → весь dev. Варианты 'nan' отбрасываются (есть вопросы на 2–3 опции).
    """
    import random
    from datasets import load_dataset

    ds = load_dataset("deepvk/MMBench-ru", "default", split="dev")
    idx = list(range(len(ds)))
    if n and n > 0 and n < len(idx):
        idx = random.Random(seed).sample(idx, n)
        idx.sort()
    items = []
    for i in idx:
        r = ds[i]
        opts = [(L, r[L]) for L in ("A", "B", "C", "D")
                if r.get(L) is not None and str(r[L]).strip().lower() != "nan"]
        hint = r.get("hint")
        hint = None if (hint is None or str(hint).strip().lower() == "nan") else str(hint)
        items.append({"index": r["index"], "question": r["question"], "hint": hint,
                      "options": opts, "answer": r["answer"], "category": r.get("category"),
                      "image": r["image"]})
    return items


# --------------------------------------------------------------------------- промпты

GQA_INSTRUCTION = "Ответь на вопрос кратко — одним словом или короткой фразой."


def gqa_prompt(item) -> str:
    return f"{GQA_INSTRUCTION}\n{item['question']}"


def mmbench_prompt(item) -> str:
    lines = [f"{L}. {val}" for L, val in item["options"]]
    hint = f"{item['hint']}\n" if item["hint"] else ""
    return (f"{hint}{item['question']}\n" + "\n".join(lines) +
            "\nОтветь только буквой варианта (A, B, C или D).")


# --------------------------------------------------------------------------- прогон

def _key_of(rec):
    return rec.get("id", rec.get("index"))


def run_benchmark(vlm: VLM, benchmark: str, items, out_path: str, *, max_new_tokens: int):
    """Прогнать модель по бенчмарку, писать сырые предсказания построчно (jsonl). → meta dict.

    **Резюмируемо:** если out_path уже частично заполнен (прошлый прогон убит — WSL реапит
    отвязанные процессы, STATE 04 #1), читаем готовые строки, пропускаем их id и ДОПИСЫВАЕМ
    остаток. Метрика считается по всем строкам (готовым + новым). Повторный вызов = докатка.
    .meta.json пишется только когда обработаны ВСЕ items (признак завершения).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(items)
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done[_key_of(r)] = r
                except json.JSONDecodeError:
                    continue  # оборванная последняя строка от kill — игнор
        print(f"[resume] {vlm.name}|{benchmark}: уже готово {len(done)}/{n}", flush=True)

    n_correct = n_lenient = n_extracted = n_no_letter = 0
    t0 = time.time()
    mode = "a" if done else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for k, it in enumerate(items, 1):
            key = it["id"] if benchmark == "gqa" else it["index"]
            rec = done.get(key)
            if rec is None:
                img = it["image"].convert("RGB")
                if benchmark == "gqa":
                    pred = vlm.answer(img, gqa_prompt(it), max_new_tokens=max_new_tokens)
                    exact, lenient, extracted = gqa_correct(pred, it["answer"])
                    rec = {"id": it["id"], "imageId": it["imageId"], "question": it["question"],
                           "gold": it["answer"], "pred": pred, "correct": exact,
                           "correct_lenient": lenient, "correct_extracted": extracted}
                else:
                    pred = vlm.answer(img, mmbench_prompt(it), max_new_tokens=max_new_tokens)
                    letter = parse_letter(pred)
                    rec = {"index": it["index"], "question": it["question"],
                           "category": it["category"], "gold": it["answer"],
                           "pred_raw": pred, "pred_letter": letter,
                           "correct": (letter == it["answer"])}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()  # устойчивость к kill: прогресс на диске сразу
            if benchmark == "gqa":
                n_correct += int(rec["correct"])
                n_lenient += int(rec.get("correct_lenient", rec["correct"]))
                # старые записи (резюм прошлого прогона) могли не иметь extracted → пересчёт
                n_extracted += int(rec["correct_extracted"]) if "correct_extracted" in rec \
                    else int(gqa_correct(rec["pred"], rec["gold"])[2])
            else:
                n_correct += int(rec["correct"])
                if rec.get("pred_letter", "") == "":
                    n_no_letter += 1
            if k % 100 == 0 or k == n:
                acc = n_correct / k
                extra = f" no_letter={n_no_letter}" if benchmark == "mmbench" else ""
                print(f"[{vlm.name}|{benchmark}] {k}/{n} acc={acc:.4f}{extra} "
                      f"({(time.time()-t0)/max(k-len(done),1):.2f}s/it)", flush=True)

    meta = {
        "model": vlm.name, "label": MODELS[vlm.name]["label"], "benchmark": benchmark,
        "n": n, "n_correct": n_correct, "accuracy": round(n_correct / n, 4) if n else 0.0,
        "max_new_tokens": max_new_tokens, "seconds": round(time.time() - t0, 1),
        "raw": os.path.basename(out_path),
    }
    if benchmark == "gqa":
        meta["accuracy_lenient"] = round(n_lenient / n, 4) if n else 0.0
        meta["accuracy_extracted"] = round(n_extracted / n, 4) if n else 0.0
    else:
        meta["n_no_letter"] = n_no_letter
        meta["letter_rate"] = round((n - n_no_letter) / n, 4) if n else 0.0
    with open(out_path.replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] {vlm.name}|{benchmark}: acc={meta['accuracy']:.4f} "
          f"(n={n}, {meta['seconds']}s)", flush=True)
    return meta
