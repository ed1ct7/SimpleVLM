# Окружение VK-VLM (WSL2 + CUDA 12.8 + Blackwell)

Рабочее ML-окружение для RTX 5080 (**Blackwell, sm_120**, 16 ГБ VRAM) в **WSL2 / Ubuntu 24.04**.
Источник правды по железу — `../../CONTEXT.md §4`.

> **Главное правило Blackwell:** старые wheels падают с `no kernel image is available`.
> Нужны **CUDA 12.8+**, **PyTorch 2.7+ (cu128)** и **свежий bitsandbytes** с ядрами sm_120.
> Всё ставится **в WSL2**, окружение и данные — в WSL-ФС (`~/...`), **не** на `/mnt/c|d`.

---

## 0. Предусловия на Windows (хост)

1. **Драйвер NVIDIA** с поддержкой CUDA 12.8+ (проброс в WSL идёт с Windows-драйвера —
   отдельный CUDA-драйвер внутри WSL ставить НЕ нужно).
   Проверка в PowerShell:
   ```powershell
   nvidia-smi      # Driver 570+/590+, CUDA Version 12.8 и выше, видно "RTX 5080"
   ```
2. **WSL2** установлен (`wsl --version` показывает версию 2).

## 1. Установка Ubuntu 24.04 в WSL2

```powershell
wsl --install -d Ubuntu-24.04 --no-launch     # --no-launch: без интерактивного OOBE
```

Проверка проброса GPU внутрь WSL (должен быть виден RTX 5080):

```bash
wsl -d Ubuntu-24.04 -- nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 5080 (UUID: ...)
```

## 2. Системные пакеты (внутри WSL)

```bash
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip python3-dev build-essential git
python3 --version        # Ubuntu 24.04 -> Python 3.12.x
```

## 3. Изолированное окружение в WSL-ФС

```bash
cd ~                                  # домашняя папка WSL, НЕ /mnt/*
python3 -m venv ~/vk-vlm-env
source ~/vk-vlm-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. PyTorch под CUDA 12.8 (cu128)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 5. ML-стек (свежие, Blackwell-совместимые версии)

```bash
pip install transformers trl peft bitsandbytes accelerate datasets pillow sentencepiece
```

Либо одной командой из зафиксированного списка:

```bash
pip install -r requirements.txt        # + сначала torch из шага 4 (cu128 index)
```

> `requirements.txt` НЕ содержит индекс cu128 — torch/torchvision ставьте **строго** командой
> из шага 4, иначе подтянется CPU-сборка или сборка без ядер sm_120.

## 6. Проверка (обязательно)

```bash
source ~/vk-vlm-env/bin/activate
python solution/env/check_gpu.py
```

Ожидаемо: `CUDA available: True`, `NVIDIA GeForce RTX 5080`, проходят матмул на GPU и
bitsandbytes 4-bit тест, в конце — `ALL CHECKS PASSED`.

---

## Footgun'ы (не игнорировать)

| Симптом | Причина | Фикс |
|---|---|---|
| `no kernel image is available for execution` | wheel без ядер sm_120 | ставить torch из cu128-индекса, свежий bitsandbytes |
| `torch.cuda.is_available() == False` | CPU-сборка torch / нет проброса GPU | переустановить cu128; проверить `nvidia-smi` в WSL |
| bitsandbytes падает на 4-bit | старая сборка без Blackwell-ядер | `pip install -U bitsandbytes` |
| Дикий тормоз I/O | окружение/данные на `/mnt/c|d` | держать в `~/` (ext4 WSL-ФС) |

## Файлы

- `check_gpu.py` — скрипт проверки GPU/CUDA/bitsandbytes (sm_120).
- `requirements.txt` — зафиксированные версии стека (без torch-индекса, см. шаг 4).
- `README.md` — этот файл.
