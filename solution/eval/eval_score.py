#!/usr/bin/env python3
"""Чистый скоринг/парсинг ответов (без torch/transformers) — общий для eval и rescore.

Вынесено из eval_common, чтобы пересчёт метрик из сырых предсказаний (`rescore_gqa.py`)
не тянул тяжёлый ML-стек и гонялся где угодно (в т.ч. на Windows-python без torch).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- нормализация GQA

_PUNCT = re.compile(r"[^0-9a-zа-я ]+")
_WS = re.compile(r"\s+")

YESNO = {"да", "нет"}


def norm_answer(s: str) -> str:
    """Нормализация для exact-match GQA: lower, ё→е, выкинуть пунктуацию, схлопнуть пробелы."""
    s = (s or "").strip().lower().replace("ё", "е")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def extract_answer(p_norm: str, g_norm: str) -> bool:
    """Извлечь короткий ответ из (возможно многословного) предсказания и сверить с gold.

    Объединяется с exact (через `gqa_correct` — никогда не ниже строгого совпадения). Эталоны
    уже краткие → метрика их не меняет; многословные ответы (мои модели, STATE 03 #5)
    возвращают потерянные на exact-match совпадения. Правила:
      - gold ∈ {да, нет}: решает первое встреченное слово-полярность («Да, серфер…» → «да»);
      - иначе: gold как ведущее слово ИЛИ среди первых трёх (фокусный ответ, не зарытый
        в длинном неверном — строже, чем lenient «gold где угодно»).
    Оба аргумента — уже НОРМАЛИЗОВАННЫЕ строки (см. `norm_answer`).
    """
    if g_norm == "":
        return False
    pw = p_norm.split()
    if p_norm == g_norm:
        return True
    if g_norm in YESNO:
        head = next((w for w in pw if w in YESNO), "")
        return head == g_norm
    fw = pw[0] if pw else ""
    return fw == g_norm or g_norm in pw[:3]


def gqa_correct(pred: str, gold: str):
    """(exact, lenient, extracted) для одного ответа GQA.

    - exact: строгое равенство нормализованных (стандарт GQA, штрафует многословие);
    - lenient: gold целым словом где угодно в pred (диагностический потолок, может пере-засчитать);
    - extracted: короткий ответ из многословного (см. extract_answer), честно между моделями
      разной краткости — основная метрика сравнения (D11).
    """
    p, g = norm_answer(pred), norm_answer(gold)
    exact = (p == g) and g != ""
    lenient = exact or (g != "" and g in p.split())
    extracted = exact or extract_answer(p, g)
    return exact, lenient, extracted


# --------------------------------------------------------------------------- парсинг MMBench

_LETTER = re.compile(r"[ABCD]")


def parse_letter(s: str) -> str:
    """Первая буква варианта A–D в ответе модели (пусто, если буквы нет → ложно низкая метрика)."""
    m = _LETTER.search((s or "").upper())
    return m.group(0) if m else ""
