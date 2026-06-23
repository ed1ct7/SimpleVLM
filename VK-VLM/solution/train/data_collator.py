#!/usr/bin/env python3
"""Кастомный data collator (текст + изображение) поверх манифеста этапа 02.

Вход коллатора — «сырые» строки манифеста `processed/llava_instruct_ru.jsonl`:
    {"id", "type", "image": "coco/train2014/...jpg", "messages": [...]}
`messages` — структурный chat-формат этапа 02 (`content` = список `{type:image|text}`).

Что делает на каждый батч:
  1. грузит картинки (PIL) по относительному пути от data-root;
  2. сводит структурные сообщения к плоскому тексту с ровно одним токеном `<image>`
     и применяет chat-шаблон LLM (`apply_chat_template`);
  3. прогоняет (текст+картинки) через `LlavaProcessor` → `input_ids`/`attention_mask`/
     `pixel_values` (процессор разворачивает `<image>` в N image-токенов);
  4. строит `labels`: маскирует паддинг (по attention_mask) и позиции image-токенов (-100).

> Отладочный режим: лосс считается по всей текстовой последовательности (паддинг и image-токены
> исключены). Маскирование промпта (assistant-only loss) — апгрейд для финального этапа.
"""
from __future__ import annotations

import os

from PIL import Image

from model import IMAGE_TOKEN


def flatten_messages(messages, image_token: str = IMAGE_TOKEN):
    """Структурный chat-content → плоский (роль, строка) с одним `<image>`-плейсхолдером."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        has_image = False
        parts = []
        for p in content or []:
            if p.get("type") == "image":
                has_image = True
            elif p.get("type") == "text":
                parts.append(p.get("text", ""))
        text = "\n".join(parts)
        if has_image:
            text = f"{image_token}\n{text}" if text else image_token
        out.append({"role": m["role"], "content": text})
    return out


class VLMDataCollator:
    """Собирает батч (картинка, диалог) для `SFTTrainer`/`LlavaForConditionalGeneration`.

    `mask_prompt=True` (финал, задача 04): **assistant-only loss** — лосс считается только по
    токенам ответов ассистента; промпт (system/user-турны, image-токены, паддинг) маскируется
    `-100`. Так модель учится отвечать, а не воспроизводить вопрос (STATE 03, заметка #5).
    `mask_prompt=False` (отладка 03): лосс по всей текстовой последовательности.
    """

    def __init__(self, processor, data_root: str, max_length: int = 1024,
                 image_token: str = IMAGE_TOKEN, mask_prompt: bool = False):
        self.processor = processor
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.max_length = max_length
        self.image_token = image_token
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
        self.mask_prompt = mask_prompt
        self._warned = False
        # Токен-маркеры для assistant-only loss. `asst_ids` — префикс генерации ассистента
        # (то, что добавляет add_generation_prompt); `boundary_id` — токен начала любого турна
        # (первый токен маркера, напр. <|start_header_id|> у Llama-3 / <|im_start|> у Qwen).
        self.asst_ids, self.boundary_id = (None, None)
        if mask_prompt:
            self.asst_ids, self.boundary_id = self._derive_assistant_marker()

    def _derive_assistant_marker(self):
        tok = self.processor.tokenizer
        try:
            base = [{"role": "user", "content": "x"}]
            with_gen = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
            no_gen = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=False)
            if not with_gen.startswith(no_gen):
                raise ValueError("add_generation_prompt не является суффиксом")
            asst_prefix = with_gen[len(no_gen):]
            asst_ids = tok(asst_prefix, add_special_tokens=False).input_ids
            if not asst_ids:
                raise ValueError("пустой assistant-префикс")
            print(f"[mask] assistant-only loss: префикс={asst_prefix!r} → {asst_ids} "
                  f"(boundary_id={asst_ids[0]})")
            return asst_ids, asst_ids[0]
        except Exception as e:  # noqa: BLE001 — деградируем к лоссу по всей последовательности
            print(f"[mask][warn] не удалось вывести assistant-маркер ({e}); "
                  f"использую лосс по всей последовательности")
            return None, None

    def _assistant_only_labels(self, ids_row):
        """labels по строке input_ids: размаскированы только спаны ответов ассистента."""
        ids = ids_row.tolist()
        L = len(ids)
        labels = [-100] * L
        k = len(self.asst_ids)
        found = False
        i = 0
        while i <= L - k:
            if ids[i:i + k] == self.asst_ids:
                start = i + k
                j = start
                while j < L and ids[j] != self.boundary_id:
                    j += 1
                for t in range(start, j):
                    labels[t] = ids[t]
                found = True
                i = j
            else:
                i += 1
        return labels, found

    def _load_image(self, rel_path: str) -> Image.Image:
        path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_root, rel_path)
        return Image.open(path).convert("RGB")

    def __call__(self, examples):
        images, texts = [], []
        for ex in examples:
            images.append(self._load_image(ex["image"]))
            flat = flatten_messages(ex["messages"], self.image_token)
            text = self.processor.apply_chat_template(
                flat, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        batch = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        if self.mask_prompt and self.asst_ids is not None:
            # assistant-only: маскируем всё, кроме спанов ответов ассистента.
            import torch
            fallback_rows = 0
            for r in range(labels.shape[0]):
                row_labels, found = self._assistant_only_labels(batch["input_ids"][r])
                if found:
                    labels[r] = torch.tensor(row_labels, dtype=labels.dtype)
                else:
                    fallback_rows += 1  # маркер не найден → оставляем полную последовательность
            if fallback_rows and not self._warned:
                print(f"[mask][warn] assistant-маркер не найден в {fallback_rows} строке(ах) "
                      f"батча → для них лосс по всей последовательности")
                self._warned = True
        labels[batch["attention_mask"] == 0] = -100      # паддинг
        labels[batch["input_ids"] == self.image_token_id] = -100  # image-токены
        batch["labels"] = labels
        return batch
