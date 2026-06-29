#!/usr/bin/env python3
"""Общие утилиты пайплайна данных VK-VLM (этап 02).

Содержит то, что нужно сразу нескольким скриптам (`download.py`, `build_dataset.py`,
`sample.py`):

- разрешение корня данных (`~/vk-vlm-data`, WSL-ФС, НЕ /mnt);
- индекс изображений COCO по 12-значному id (layout-agnostic: понимает и 2014-нейминг
  `COCO_train2014_<id>.jpg`, и 2017-нейминг `<id>.jpg`);
- приведение `conversations` LLaVA к chat-формату (роли user/assistant + плейсхолдер
  изображения), с выбором шаблона.

> Нюанс данных (проверено): в `deepvk/LLaVA-Instruct-ru` поле `image` =
> `coco/train2017/<id>.jpg` (2017-нейминг), хотя физически это изображения COCO **train2014**.
> Поэтому сопоставление идёт **по id**, а не по точному имени файла — см. `build_coco_index`.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any

DEFAULT_DATA_ROOT = "~/vk-vlm-data"
DATA_ROOT_ENV = "VK_VLM_DATA_ROOT"

# 12-значный id картинки COCO (есть и в `000000253464.jpg`, и в `COCO_train2014_000000253464.jpg`).
IMAGE_ID_RE = re.compile(r"(\d{12})")
# Плейсхолдер изображения в исходных репликах LLaVA.
IMAGE_TOKEN_RE = re.compile(r"<image>")

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}

TEMPLATES = ("messages", "text")


# --------------------------------------------------------------------------- paths

def resolve_data_root(arg: str | None = None, *, warn: bool = True) -> str:
    """Корень данных: аргумент CLI > env VK_VLM_DATA_ROOT > дефолт `~/vk-vlm-data`.

    Возвращает абсолютный путь. При `warn` ругается, если корень лежит на /mnt/* —
    это медленный I/O, данные тренировки держим в WSL-ФС (см. CONTEXT §4).
    """
    raw = arg or os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT
    root = os.path.abspath(os.path.expanduser(raw))
    if warn and (root.startswith("/mnt/") or re.match(r"^[A-Za-z]:[\\/]", root)):
        print(
            f"[warn] data-root '{root}' выглядит как Windows/`/mnt` путь — медленный I/O. "
            f"Держи данные в WSL-ФС (напр. {DEFAULT_DATA_ROOT}).",
            file=sys.stderr,
        )
    return root


# --------------------------------------------------------------------------- COCO index

def extract_image_id(path_or_name: str) -> str | None:
    """Вытащить 12-значный id картинки из пути/имени (`coco/train2017/000000253464.jpg`)."""
    if not path_or_name:
        return None
    m = IMAGE_ID_RE.search(os.path.basename(str(path_or_name)))
    return m.group(1) if m else None


def build_coco_index(coco_dir: str, data_root: str | None = None) -> dict[str, str]:
    """Просканировать каталог COCO и собрать индекс `id -> путь к .jpg`.

    Обходит сам `coco_dir` и его подкаталоги первого уровня (`train2014`, `val2014`,
    `train2017`, ... — любые, что есть на диске). Имя файла может быть в 2014- или
    2017-нейминге; ключ всегда — 12-значный id.

    Если задан `data_root`, путь в индексе хранится **относительно** него (для переносимого
    манифеста); иначе — абсолютный.
    """
    index: dict[str, str] = {}
    if not os.path.isdir(coco_dir):
        return index

    def add_dir(d: str) -> None:
        try:
            entries = os.scandir(d)
        except OSError:
            return
        with entries:
            for e in entries:
                if not e.is_file():
                    continue
                name = e.name
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_id = extract_image_id(name)
                if not img_id:
                    continue
                full = e.path
                rel = os.path.relpath(full, data_root) if data_root else full
                index.setdefault(img_id, rel.replace(os.sep, "/"))

    # файлы прямо в coco_dir (плоский layout) + подкаталоги-сплиты
    add_dir(coco_dir)
    try:
        subdirs = [e.path for e in os.scandir(coco_dir) if e.is_dir()]
    except OSError:
        subdirs = []
    for d in sorted(subdirs):
        add_dir(d)
    return index


# --------------------------------------------------------------------------- chat format

def _clean_text(value: Any) -> str:
    """Убрать токены `<image>` из реплики и подчистить пробелы/переводы строк."""
    text = "" if value is None else str(value)
    text = IMAGE_TOKEN_RE.sub("", text)
    # схлопнуть пустые строки, оставшиеся после удаления токена
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def to_chat(
    conversations: list[dict[str, Any]],
    *,
    template: str = "messages",
    image_token: str = "<image>",
) -> list[dict[str, Any]]:
    """Привести `conversations` LLaVA к chat-формату целевой LLM.

    Роли: human→user, gpt→assistant, system→system. Изображение крепится ровно один раз —
    к первой реплике, где был токен `<image>` (если нигде — к первой user-реплике).

    Шаблоны (`template`):
      - `messages` — структурный content (`[{type:image}, {type:text,text:...}]`); рекомендуется
        для TRL/processor, реальный chat-template применяется уже на обучении (этап 03);
      - `text`     — плоский content-строка; изображение рендерится как `image_token` в начале
        соответствующей реплики.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template '{template}', choose from {TEMPLATES}")

    # выбрать, к какой реплике прикрепить изображение
    img_idx = -1
    first_user = -1
    for i, turn in enumerate(conversations):
        role = ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
        if first_user < 0 and role == "user":
            first_user = i
        if img_idx < 0 and "<image>" in str(turn.get("value", "")):
            img_idx = i
    if img_idx < 0:
        img_idx = first_user if first_user >= 0 else 0

    messages: list[dict[str, Any]] = []
    for i, turn in enumerate(conversations):
        role = ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
        text = _clean_text(turn.get("value"))
        has_image = i == img_idx
        if template == "messages":
            content: list[dict[str, Any]] = []
            if has_image:
                content.append({"type": "image"})
            if text:
                content.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content})
        else:  # text
            if has_image and image_token:
                flat = f"{image_token}\n{text}" if text else image_token
            else:
                flat = text
            messages.append({"role": role, "content": flat})
    return messages


def render_messages_text(messages: list[dict[str, Any]]) -> str:
    """Человекочитаемый рендер chat-сообщений (для печати в sample.py)."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if p.get("type") == "image":
                    parts.append("<image>")
                elif p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
            rendered = "\n".join(parts)
        else:
            rendered = str(content)
        lines.append(f"[{role}] {rendered}")
    return "\n".join(lines)
