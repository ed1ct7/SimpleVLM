#!/usr/bin/env python3
"""Конфигурируемое QLoRA-обучение VK-VLM (этап 03 — отладочный end-to-end цикл).

Цель этапа — не метрика, а доказать рабочий цикл «данные → train → сохранение адаптеров»
без ошибок (OOM / sm_120). Гиперы — из YAML (`configs/2b-debug.yaml`), любой можно
переопределить флагом CLI.

Пример:
    python solution/train/train.py --config solution/train/configs/2b-debug.yaml
    python solution/train/train.py --config ... --smoke        # быстрый дым (8 примеров, 2 шага)
    python solution/train/train.py --config ... --subset 5000 --epochs 1

Стек/железо — CONTEXT §4-5 (WSL2, RTX 5080 sm_120, 16 ГБ; bf16 + 4-bit + grad checkpointing).
"""
from __future__ import annotations

import argparse
import json
import os
import random

# Кэш моделей/датасетов — в WSL-ФС (не на C:!). Иначе скачивание весов забьёт системный диск
# (см. инцидент disk-full в STATE, этап 02). Ставим ДО импорта transformers.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/vk-vlm-data/hf-cache"))
# Аллокатор с расширяемыми сегментами — меньше фрагментации у потолка 16 ГБ VRAM. Без него
# прогон при длинных seq уходил в near-OOM thrash (пейджинг paged-optimizer, ~7x замедление).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

from data_collator import VLMDataCollator
from model import apply_qlora, build_vlm, freeze_for_alignment, load_projector, save_projector


# --------------------------------------------------------------------------- config / data

def deep_get(cfg: dict, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_subset(manifest: str, subset: int, seed: int):
    """Reservoir-выборка `subset` строк JSONL (один проход, без загрузки всего файла)."""
    rng = random.Random(seed)
    chosen, seen = [], 0
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            rec = None
            if len(chosen) < subset:
                rec = json.loads(line)
                chosen.append(rec)
            else:
                j = rng.randint(0, seen - 1)
                if j < subset:
                    chosen[j] = json.loads(line)
    print(f"[data] манифест: {seen} строк → подвыборка {len(chosen)} (seed={seed})")
    return chosen


# --------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description="QLoRA-обучение VK-VLM (отладка 2B)")
    p.add_argument("--config", required=True)
    p.add_argument("--subset", type=int, default=None, help="число примеров (переопр. YAML)")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="ограничить число шагов")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--smoke", action="store_true", help="дым: subset=8, max_steps=2, без сохранения")
    p.add_argument("--stage", type=int, default=None, choices=[1, 2],
                   help="1=alignment (только проектор), 2=instruction tuning (QLoRA). Переопр. YAML")
    p.add_argument("--init-projector-from", default=None,
                   help="путь к projector.safetensors стадии 1 (для стадии 2)")
    p.add_argument("--resume", action="store_true", help="продолжить с последнего чекпойнта output_dir")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    encoder_id = cfg["encoder_id"]
    llm_id = cfg["llm_id"]
    image_size = cfg.get("image_size")
    load_in_4bit = cfg.get("load_in_4bit", True)
    seed = cfg.get("seed", 42)
    stage = args.stage if args.stage is not None else cfg.get("stage", 2)
    meta_init = cfg.get("meta_init", True)
    mask_prompt = deep_get(cfg, "train.mask_prompt", stage == 2)  # assistant-only по умолч. на стадии 2
    init_projector_from = (args.init_projector_from
                           or os.path.expanduser(cfg.get("init_projector_from") or "") or None)

    data_root = os.path.expanduser(args.data_root or deep_get(cfg, "data.data_root", "~/vk-vlm-data"))
    manifest = os.path.expanduser(
        args.manifest or deep_get(cfg, "data.manifest",
                                  os.path.join(data_root, "processed", "llava_instruct_ru.jsonl"))
    )
    output_dir = args.output_dir or cfg.get("output_dir", "solution/model/2b-debug")

    subset = args.subset or deep_get(cfg, "train.subset", 2000)
    epochs = args.epochs if args.epochs is not None else deep_get(cfg, "train.epochs", 1)
    max_steps = args.max_steps if args.max_steps is not None else deep_get(cfg, "train.max_steps", -1)
    if args.smoke:
        subset, max_steps, output_dir = 8, 2, output_dir.rstrip("/") + "-smoke"

    max_length = deep_get(cfg, "train.max_length", 1024)
    random.seed(seed)
    torch.manual_seed(seed)

    # 1) данные
    records = load_subset(manifest, subset, seed)
    from datasets import Dataset
    ds = Dataset.from_list([{"image": r["image"], "messages": r["messages"]} for r in records])

    # 2) модель + стадия обучения (1 = только проектор; 2 = QLoRA на LLM + проектор)
    print(f"[stage] стадия {stage}: "
          + ("alignment — обучается ТОЛЬКО проектор (энкодер+LLM заморожены)"
             if stage == 1 else "instruction tuning — QLoRA (LLM) + проектор"))
    model, processor, tokenizer = build_vlm(
        encoder_id, llm_id, load_in_4bit=load_in_4bit, image_size=image_size,
        compute_dtype=torch.bfloat16, attn_implementation=cfg.get("attn_implementation", "sdpa"),
        meta_init=meta_init,
    )
    if stage == 2 and init_projector_from and os.path.exists(init_projector_from):
        load_projector(model, init_projector_from)  # стартуем с выровненного проектора (стадия 1)
    elif stage == 2 and init_projector_from:
        print(f"[stage2][warn] проектор стадии 1 не найден: {init_projector_from} — "
              f"стартую со случайного проектора (ок только для дыма/отладки)")
    elif stage == 2 and not init_projector_from:
        print("[stage2][warn] init_projector_from не задан — проектор стартует случайным "
              "(без alignment-стадии). Рекомендуется сначала прогнать стадию 1.")

    if stage == 1:
        model = freeze_for_alignment(model, use_4bit_prep=load_in_4bit)
    else:
        model = apply_qlora(
            model,
            r=deep_get(cfg, "lora.r", 16),
            alpha=deep_get(cfg, "lora.alpha", 32),
            dropout=deep_get(cfg, "lora.dropout", 0.05),
            target_modules=deep_get(cfg, "lora.target_modules", None),
            train_projector=deep_get(cfg, "lora.train_projector", True),
            use_4bit_prep=load_in_4bit,
        )

    # 3) коллатор (assistant-only loss на финале — STATE 03 заметка #5)
    collator = VLMDataCollator(processor, data_root=data_root, max_length=max_length,
                               mask_prompt=mask_prompt)

    # 4) Trainer (TRL SFT)
    from trl import SFTConfig, SFTTrainer

    sft = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=deep_get(cfg, "train.per_device_batch_size", 1),
        gradient_accumulation_steps=deep_get(cfg, "train.grad_accum", 8),
        learning_rate=float(deep_get(cfg, "train.lr", 1e-4)),
        num_train_epochs=epochs,
        max_steps=max_steps,
        warmup_ratio=deep_get(cfg, "train.warmup_ratio", 0.03),
        lr_scheduler_type=deep_get(cfg, "train.lr_scheduler", "cosine"),
        logging_steps=deep_get(cfg, "train.logging_steps", 5),
        save_strategy=deep_get(cfg, "train.save_strategy", "epoch"),
        save_steps=deep_get(cfg, "train.save_steps", 500),
        save_total_limit=deep_get(cfg, "train.save_total_limit", 2),
        optim=deep_get(cfg, "train.optim", "paged_adamw_8bit"),
        bf16=True,
        gradient_checkpointing=deep_get(cfg, "train.grad_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        seed=seed,
        dataloader_num_workers=deep_get(cfg, "train.dataloader_workers", 2),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        data_collator=collator,
        train_dataset=ds,
        processing_class=processor,
    )

    # Стадия 1: модель НЕ PEFT-обёрнута → штатный чекпойнт Trainer сохранил бы весь 4-bit base
    # (~5 ГБ шардами, ~40 с). Вместо этого — лёгкий колбэк, кладущий ТОЛЬКО проектор (~20 МБ).
    if stage == 1 and not args.smoke:
        from transformers import TrainerCallback

        ckpt_every = deep_get(cfg, "train.save_steps", 0) or 0

        class ProjectorSaveCallback(TrainerCallback):
            def on_step_end(self, a, state, control, **kw):
                if ckpt_every and state.global_step > 0 and state.global_step % ckpt_every == 0:
                    save_projector(model, os.path.join(output_dir, "projector.safetensors"))
                return control

        if ckpt_every:
            os.makedirs(output_dir, exist_ok=True)
            trainer.add_callback(ProjectorSaveCallback())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_out = trainer.train(resume_from_checkpoint=args.resume or None)

    # 5) сохранение артефактов (стадия-зависимо) + процессор + мета
    if not args.smoke:
        os.makedirs(output_dir, exist_ok=True)
        if stage == 1:
            # стадия 1: сохраняем только проектор (вход для стадии 2)
            save_projector(model, os.path.join(output_dir, "projector.safetensors"))
        else:
            # стадия 2: LoRA-адаптеры + проектор (modules_to_save) одним save_model
            trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)   # tokenizer(+<image>) + image_processor + chat_template
        meta = {
            "encoder_id": encoder_id,
            "llm_id": llm_id,
            "image_size": image_size,
            "load_in_4bit": load_in_4bit,
            "stage": stage,
            "init_projector_from": init_projector_from,
            "mask_prompt": mask_prompt,
            "subset": subset,
            "epochs": epochs,
            "max_length": max_length,
            "lr": float(deep_get(cfg, "train.lr", 1e-4)),
            "grad_accum": deep_get(cfg, "train.grad_accum", 8),
            "train_loss": train_out.metrics.get("train_loss"),
            "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                             if torch.cuda.is_available() else None),
        }
        with open(os.path.join(output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # отчёт
    print("\n=== ИТОГ ОБУЧЕНИЯ ===")
    print(f"  шагов: {train_out.global_step}")
    print(f"  train_loss: {train_out.metrics.get('train_loss')}")
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if losses:
        print(f"  loss: первый {losses[0]:.4f} → последний {losses[-1]:.4f} "
              f"({'снижается ✅' if losses[-1] < losses[0] else 'НЕ снизился ⚠️'})")
    if torch.cuda.is_available():
        print(f"  пик VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} ГБ")
    if not args.smoke:
        what = "проектор (стадия 1)" if stage == 1 else "адаптеры + проектор (стадия 2)"
        print(f"  {what} сохранены: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
