# Задача 01: Подготовка окружения (WSL2 + CUDA + ML-стек)

## Контекст
Прочитай `CONTEXT.md` (источник правды) и `STATE.md` (что сделано). Особое внимание —
`CONTEXT.md §4` (оборудование, footgun'ы Blackwell/WSL2).

## Цель
Поднять рабочее ML-окружение в WSL2 на машине с RTX 5080 (Blackwell, sm_120), где
запускается PyTorch с поддержкой GPU.

## Вход
- Хост: Windows 11, RTX 5080 16 ГБ, 64 ГБ RAM. WSL2 может быть ещё не установлен.

## Что сделать
1. Убедиться, что установлен WSL2 + дистрибутив Ubuntu и свежий драйвер NVIDIA на Windows
   (с поддержкой CUDA 12.8+). В WSL отдельный CUDA-драйвер ставить НЕ нужно — он проброшен
   с Windows.
2. Создать изолированное окружение (conda или `python -m venv`) в WSL-ФС (`~/vk-vlm-env`),
   НЕ на `/mnt/*`.
3. Установить PyTorch со сборкой под CUDA 12.8 (cu128, версия 2.7+).
4. Установить ML-стек: `transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate`,
   `datasets`, `pillow`, `sentencepiece`. Версии — свежие (Blackwell-совместимые).
5. Написать скрипт проверки `solution/env/check_gpu.py`: печатает версию torch, CUDA,
   `torch.cuda.is_available()`, имя GPU, и прогоняет тест-матмул на GPU + крошечный
   bitsandbytes 4-bit тест (чтобы поймать sm_120-несовместимость СРАЗУ).
6. Зафиксировать версии в `solution/env/requirements.txt` (или `environment.yml`) +
   краткий `solution/env/README.md` с командами установки.

## Выход (артефакты)
- `solution/env/check_gpu.py` — скрипт проверки.
- `solution/env/requirements.txt` (или `environment.yml`) — зафиксированные версии.
- `solution/env/README.md` — пошаговая установка.

## Критерий приёмки
- [ ] `python solution/env/check_gpu.py` печатает `CUDA available: True` и `NVIDIA GeForce RTX 5080`.
- [ ] Тест-матмул на GPU проходит без `no kernel image is available`.
- [ ] bitsandbytes 4-bit тест проходит без ошибки sm_120.
- [ ] Версии зафиксированы в requirements.

## Footgun'ы (из CONTEXT §4)
- Старые wheels падают на Blackwell → ставить cu128 / свежий bitsandbytes.
- Окружение и данные — в WSL-ФС, не на `/mnt/c|d` (медленно).

## В конце
Допиши `STATE.md`: статус 01 → ✅, список созданных артефактов, отметь, что разблокирована
задача 02. Если поймал блокер по стеку — запиши точную ошибку и пометь ⛔.
