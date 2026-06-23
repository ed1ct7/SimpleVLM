#!/usr/bin/env python3
"""Сборка VLM LLaVA-стиля из произвольных компонентов (этап 03).

    визуальный энкодер (CLIP / SigLIP)  ->  проектор (MLP)  ->  LLM (Qwen2 / Gemma)

Используется ГОТОВЫЙ класс `transformers.LlavaForConditionalGeneration` (VLM с нуля не пишем —
CONTEXT D1). Компоненты параметризованы (`encoder_id`, `llm_id`) → этот же код переиспользуется
в задаче 04 для целевого обучения 8B (Saiga-8b).

Сборка (`build_vlm`):
  1. грузим LLM (опц. 4-bit / QLoRA) + токенайзер; добавляем спец-токен `<image>` и ресайзим
     эмбеддинги под новый словарь;
  2. грузим визуальный энкодер (vision tower) в compute-dtype;
  3. строим `LlavaConfig` из двух под-конфигов, создаём `LlavaForConditionalGeneration` и
     **пересаживаем** реальные веса энкодера/LLM; проектор (`LlavaMultiModalProjector`,
     2-слойный MLP) инициализируется заново и обучается;
  4. `apply_qlora` — `prepare_model_for_kbit_training` + LoRA на LLM, проектор уходит в
     `modules_to_save` (обучается целиком и сохраняется вместе с адаптерами).

Замечания по железу (CONTEXT §4, RTX 5080 / sm_120 / 16 ГБ):
  - 4-bit база (bitsandbytes nf4) + bf16 compute + gradient checkpointing → влезает с запасом;
  - энкодер и проектор держим в bf16; `bf16=True` в Trainer включает autocast и снимает
    рассогласование dtype на пути энкодер→проектор→LLM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPVisionModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    SiglipVisionModel,
)

IMAGE_TOKEN = "<image>"
# Стандартные LoRA-цели для LLM трансформера (Qwen2/Gemma/LLaMA-семейство).
DEFAULT_LLM_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- helpers

def _from_pretrained(cls, model_id, dtype, **kw):
    """`from_pretrained` устойчиво к переименованию `torch_dtype`→`dtype` (transformers 5.x)."""
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kw)


def _materialize_projector(projector: nn.Module, dtype) -> None:
    """Материализовать проектор с meta на CPU и заново инициализировать (Linear: normal 0.02)."""
    projector.to_empty(device="cpu")
    for m in projector.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    projector.to(dtype)


def _vision_meta(encoder_id: str):
    """Достать (vision_config, patch, image_size, select_strategy, feature_layer, is_siglip)."""
    cfg = AutoConfig.from_pretrained(encoder_id)
    vcfg = getattr(cfg, "vision_config", cfg)
    mtype = (getattr(cfg, "model_type", "") or getattr(vcfg, "model_type", "")).lower()
    is_siglip = "siglip" in mtype
    # SigLIP: нет CLS-токена → стратегия 'full', берём последний слой.
    # CLIP (рецепт LLaVA): дропаем CLS ('default'), берём предпоследний слой (-2).
    strategy = "full" if is_siglip else "default"
    feature_layer = -1 if is_siglip else -2
    return vcfg, vcfg.patch_size, vcfg.image_size, strategy, feature_layer, is_siglip


# --------------------------------------------------------------------------- processor

def build_processor(encoder_id: str, llm_id: str, image_size: int | None = None,
                    image_token: str = IMAGE_TOKEN):
    """LlavaProcessor = image_processor(энкодер) + tokenizer(LLM) + спец-токен `<image>`.

    Возвращает (processor, tokenizer, num_added_tokens). `num_added_tokens` нужен вызывающему,
    чтобы понять, надо ли ресайзить эмбеддинги LLM.
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Картинка крепится к первому user-турну → image-токены лежат в начале последовательности.
    # Режем СПРАВА (хвост ответа), иначе обрезка может срезать image-плейсхолдеры и
    # рассинхронить число image-токенов с визуальными фичами (ошибка merge в LlavaForCG).
    tokenizer.truncation_side = "right"
    added = 0
    if image_token not in tokenizer.get_vocab():
        added = tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    _, patch, native_img, strategy, _, _ = _vision_meta(encoder_id)
    if image_size and image_size != native_img:
        print(f"[warn] image_size={image_size} != нативный размер энкодера {native_img}; "
              f"использую нативный {native_img} (смена размера ломает pos-emb энкодера)")

    # Согласование числа image-токенов с числом визуальных фич:
    # процессор считает (H/patch)*(W/patch) и при strategy="default" вычитает 1 (CLS),
    # т.е. для CLIP даёт num_patches-1=575. Но энкодер при "default" дропает CLS из 577 → 576
    # фич. Компенсируем +1 (num_additional), иначе merge падает: "image features ... do not match".
    num_additional = 1 if strategy == "default" else 0
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=patch,
        vision_feature_select_strategy=strategy,
        image_token=image_token,
        num_additional_image_tokens=num_additional,
    )
    if getattr(processor, "chat_template", None) is None:
        processor.chat_template = tokenizer.chat_template
    return processor, tokenizer, added


# --------------------------------------------------------------------------- model

def build_vlm(encoder_id: str, llm_id: str, *, load_in_4bit: bool = True,
              image_size: int | None = None, compute_dtype=torch.bfloat16,
              attn_implementation: str = "sdpa", device: str = "cuda",
              meta_init: bool = True):
    """Собрать LLaVA-VLM из энкодера + LLM. Возвращает (model, processor, tokenizer).

    Проектор инициализирован случайно (обучается). Энкодер/LLM — реальные предобученные веса.

    `meta_init=True` (по умолчанию для 8B): скелет `LlavaForConditionalGeneration(config)`
    создаётся на **meta-устройстве** (без аллокации), затем в него пересаживаются реальные
    модули, а проектор материализуется и инициализируется заново. Так избегаем транзиентного
    случайного 8B-тела на CPU (~32 ГБ RAM, STATE 03 заметка #3). Для 2B не критично.
    """
    processor, tokenizer, added = build_processor(encoder_id, llm_id, image_size)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # 1) LLM (опц. 4-bit QLoRA)
    bnb = None
    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    llm = _from_pretrained(
        AutoModelForCausalLM, llm_id, compute_dtype,
        quantization_config=bnb,
        device_map={"": 0} if (load_in_4bit and on_cuda) else None,
        attn_implementation=attn_implementation,
    )
    if added:
        llm.resize_token_embeddings(len(tokenizer))

    # 2) визуальный энкодер
    vcfg, patch, native_img, strategy, feature_layer, is_siglip = _vision_meta(encoder_id)
    num_patches = (native_img // patch) ** 2
    image_seq = num_patches + (1 if (strategy == "full" and not is_siglip) else 0)
    VisionModel = SiglipVisionModel if is_siglip else CLIPVisionModel
    vision = _from_pretrained(VisionModel, encoder_id, compute_dtype)

    # 3) сборка LlavaForConditionalGeneration + пересадка реальных весов
    config = LlavaConfig(
        vision_config=vision.config,
        text_config=llm.config,
        image_token_index=image_token_id,
        image_seq_length=image_seq,
        vision_feature_select_strategy=strategy,
        vision_feature_layer=feature_layer,
        projector_hidden_act="gelu",
    )
    config.text_config.vocab_size = len(tokenizer)
    if meta_init:
        # Скелет на meta (нулевая аллокация); проектор материализуем и инициализируем ниже.
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = LlavaForConditionalGeneration(config)
    else:
        model = LlavaForConditionalGeneration(config)
    model.model.vision_tower = vision          # энкодер
    model.model.language_model = llm.model     # тело LLM (без lm_head)
    model.lm_head = llm.lm_head                 # голова LLM
    if meta_init:
        # Единственный оставшийся на meta модуль — проектор: материализуем + инициализируем.
        _materialize_projector(model.model.multi_modal_projector, compute_dtype)
    model.config.image_token_index = image_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # 4) разложить по устройству: 4-bit LLM уже на GPU (device_map); энкодер+проектор → bf16/GPU
    if on_cuda:
        model.model.vision_tower.to(device, compute_dtype)
        model.model.multi_modal_projector.to(device, compute_dtype)
        if not load_in_4bit:
            model.model.language_model.to(device, compute_dtype)
            model.lm_head.to(device, compute_dtype)
    else:
        model.to(device)

    print(f"[model] энкодер={encoder_id} (patch={patch}, {native_img}px, {image_seq} img-токенов, "
          f"layer={feature_layer}, {strategy}) | LLM={llm_id} | 4bit={load_in_4bit}")
    return model, processor, tokenizer


def apply_qlora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                target_modules=None, train_projector: bool = True,
                use_4bit_prep: bool = True):
    """Навесить LoRA на LLM (+ проектор в modules_to_save). Возвращает PEFT-модель.

    Энкодер остаётся заморожен. Проектор обучается целиком (modules_to_save) — иначе случайно
    инициализированный проектор не научится мэппить визуальные фичи в пространство LLM.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit_prep:
        # gradient checkpointing включаем в Trainer (SFTConfig), здесь только подготовка слоёв.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()  # нужно для grad checkpointing через замороженные эмбеддинги

    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_LLM_TARGETS,
        modules_to_save=["multi_modal_projector"] if train_projector else None,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- 2-стадийность (LLaVA)

def freeze_for_alignment(model, *, use_4bit_prep: bool = True):
    """Стадия 1 (alignment): заморозить ВСЁ кроме проектора (CONTEXT §3, D1).

    Энкодер + LLM заморожены (LLM остаётся 4-bit ради памяти), обучается **только** проектор
    (`multi_modal_projector`). Цель — выровнять визуальные фичи с пространством эмбеддингов LLM,
    не трогая саму LLM. Возвращает ту же модель (НЕ PEFT — обучаем полноразмерный проектор).
    """
    from peft import prepare_model_for_kbit_training

    if use_4bit_prep:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)
    model.enable_input_require_grads()  # grad течёт через замороженные эмбеддинги к проектору

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[align] обучаемых (только проектор): {n_train/1e6:.2f}M / {n_total/1e6:.0f}M "
          f"({100*n_train/n_total:.3f}%)")
    return model


def save_projector(model, path: str) -> None:
    """Сохранить state_dict проектора (вход для стадии 2). Стадия 1 — это plain-модель.

    Примечание: у HF-моделей есть свойство `base_model`, поэтому проектор адресуем напрямую
    через `model.model.multi_modal_projector` (LlavaForConditionalGeneration → LlavaModel).
    """
    sd = model.model.multi_modal_projector.state_dict()
    from safetensors.torch import save_file
    sd = {k: v.to(torch.float16).contiguous().cpu() for k, v in sd.items()}
    save_file(sd, path)
    print(f"[align] проектор сохранён: {path} ({len(sd)} тензоров)")


def load_projector(model, path: str, dtype=torch.bfloat16) -> None:
    """Загрузить веса проектора (до `apply_qlora`). Стадия 2 стартует с выровненного проектора."""
    from safetensors.torch import load_file
    sd = load_file(path)
    proj = model.model.multi_modal_projector
    target_dtype = next(proj.parameters()).dtype
    sd = {k: v.to(target_dtype) for k, v in sd.items()}
    missing, unexpected = proj.load_state_dict(sd, strict=False)
    print(f"[stage2] проектор загружен из {path} "
          f"(missing={list(missing)}, unexpected={list(unexpected)})")
