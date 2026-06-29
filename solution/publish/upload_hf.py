#!/usr/bin/env python3
"""Публикация обученной VK-VLM на Hugging Face Hub: адаптеры + проектор + процессор + мета.

Авторизация — ВАША (скрипт не хранит токен). Перед запуском:
    huggingface-cli login          # либо export HF_TOKEN=hf_...

Запуск (из корня репозитория):
    python solution/publish/upload_hf.py --repo-id <HF_USERNAME>/vk-vlm-saiga8b-clip-lora

Что загружается из --model-dir (по умолчанию solution/model/8b):
    adapter_model.safetensors  — LoRA-адаптеры LLM + обученный проектор (~261 МБ)
    adapter_config.json        — конфиг PEFT
    training_meta.json         — id энкодера/LLM, гиперы, финальный loss, пик VRAM
    tokenizer*, processor_config.json, chat_template.jinja  — процессор
    README.md                  — карточка модели (из publish/MODEL_CARD.md)

НЕ загружаются промежуточные чекпойнты (checkpoint-*/) — только финальные адаптеры.
Базовые веса (CLIP, Saiga) не публикуются: тянутся с HF при загрузке.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Публикация VK-VLM на HF Hub")
    p.add_argument("--repo-id", required=True,
                   help="<username>/<model-name>, напр. ed1ct7/vk-vlm-saiga8b-clip-lora")
    p.add_argument("--model-dir", default="solution/model/8b",
                   help="каталог с адаптерами+процессором (по умолчанию целевая 8B)")
    p.add_argument("--card", default="solution/publish/MODEL_CARD.md",
                   help="карточка модели → загружается как README.md репозитория")
    p.add_argument("--private", action="store_true", help="создать приватный репозиторий")
    p.add_argument("--token", default=None,
                   help="HF-токен (иначе из HF_TOKEN / huggingface-cli login)")
    args = p.parse_args()

    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    model_dir = os.path.expanduser(args.model_dir)
    if not os.path.isfile(os.path.join(model_dir, "adapter_model.safetensors")):
        print(f"[!] нет adapter_model.safetensors в {model_dir} — обучите модель (task 04) "
              f"или укажите --model-dir", file=sys.stderr)
        return 1

    # 1) репозиторий (идемпотентно)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"[+] репозиторий: https://huggingface.co/{args.repo_id} (private={args.private})")

    # 2) карточка модели → README.md репозитория
    if os.path.isfile(args.card):
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="model")
        print(f"[+] карточка модели загружена ({args.card} → README.md)")

    # 3) адаптеры + процессор + мета; чекпойнты и кэш — мимо
    api.upload_folder(
        repo_id=args.repo_id, repo_type="model", folder_path=model_dir,
        ignore_patterns=["checkpoint-*", "checkpoint-*/*", "README.md",
                         "training_args.bin", "**/__pycache__/*"],
        commit_message="Upload VK-VLM adapters + projector + processor",
    )
    print(f"[+] веса+процессор загружены из {model_dir}")
    print(f"\n[done] вставьте ссылку в solution/README.md и solution/SOLUTION.md:")
    print(f"        https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
