#!/usr/bin/env python3
"""GPU/CUDA/Blackwell sanity check for the VK-VLM environment.

Run inside the WSL2 venv:

    source ~/vk-vlm-env/bin/activate
    python solution/env/check_gpu.py

What it verifies (and why it matters on RTX 5080 / Blackwell / sm_120):
  1. torch + CUDA build version, torch.cuda.is_available(), GPU name.
  2. The compiled CUDA arch list actually contains sm_120 (Blackwell).
  3. A real matmul ON the GPU — catches `no kernel image is available`
     (old wheels built without sm_120 kernels fail exactly here).
  4. A tiny bitsandbytes 4-bit quant/dequant + 4-bit linear on GPU —
     catches sm_120-incompatible bitsandbytes immediately, before any
     QLoRA training run wastes time.

Exit code 0 = all required checks passed. Non-zero = something is broken;
read the printed error. The bitsandbytes check is treated as required
because the whole project plan depends on 4-bit QLoRA for the 8B model.
"""

import sys
import traceback

EXPECTED_GPU_SUBSTR = "RTX 5080"
EXPECTED_ARCH = "sm_120"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    failures = []

    # ---- 1. torch / CUDA basics ---------------------------------------
    section("1. torch / CUDA")
    try:
        import torch
    except Exception:
        print("FATAL: cannot import torch")
        traceback.print_exc()
        return 1

    print(f"torch version      : {torch.__version__}")
    print(f"torch CUDA build   : {torch.version.cuda}")
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    print(f"cuDNN              : {cudnn}")

    available = torch.cuda.is_available()
    print(f"CUDA available     : {available}")
    if not available:
        print("FATAL: torch.cuda.is_available() is False.")
        print("  - Check `nvidia-smi` works inside WSL.")
        print("  - Check the wheel is a cu128 build, not CPU-only.")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU name           : {name}")
    print(f"Compute capability : sm_{cap[0]}{cap[1]}")
    print(f"Device count       : {torch.cuda.device_count()}")

    if EXPECTED_GPU_SUBSTR not in name:
        failures.append(f"expected '{EXPECTED_GPU_SUBSTR}' in GPU name, got '{name}'")

    # ---- 2. compiled arch list includes Blackwell ---------------------
    section("2. Compiled CUDA arch list (must include sm_120)")
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []
    print(f"arch list          : {arch_list}")
    if EXPECTED_ARCH not in arch_list:
        # Not fatal on its own — the live matmul below is the real test —
        # but warn loudly because it usually precedes a kernel-image error.
        print(f"WARNING: '{EXPECTED_ARCH}' not in compiled arch list. "
              f"Blackwell kernels may be missing.")

    # ---- 3. real matmul on the GPU ------------------------------------
    section("3. GPU matmul (catches 'no kernel image is available')")
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        checksum = float(c.float().abs().mean().item())
        print(f"matmul fp16 OK     : 2048x2048, mean|C|={checksum:.4f}")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        failures.append("GPU matmul failed")
        print("FAIL: GPU matmul raised an exception:")
        traceback.print_exc()
        if "no kernel image" in str(e):
            print("  -> classic Blackwell mismatch: wheel lacks sm_120 kernels. "
                  "Reinstall torch from the cu128 index.")

    # ---- 4. bitsandbytes 4-bit on the GPU -----------------------------
    section("4. bitsandbytes 4-bit (catches sm_120 incompatibility)")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes ver   : {bnb.__version__}")
        import torch
        import bitsandbytes.functional as F

        # 4-bit quantize -> dequantize round-trip on the GPU.
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        qx, state = F.quantize_4bit(x, quant_type="nf4")
        xdq = F.dequantize_4bit(qx, state)
        torch.cuda.synchronize()
        err = float((x - xdq).float().abs().mean().item())
        print(f"nf4 round-trip OK  : mean abs err={err:.4f}")

        # 4-bit linear layer forward — exercises the actual QLoRA kernel path.
        lin = bnb.nn.Linear4bit(64, 32, bias=False,
                                compute_dtype=torch.float16,
                                quant_type="nf4").cuda()
        out = lin(x)
        torch.cuda.synchronize()
        print(f"Linear4bit fwd OK  : out shape={tuple(out.shape)}")
    except Exception as e:
        failures.append("bitsandbytes 4-bit failed")
        print("FAIL: bitsandbytes 4-bit test raised an exception:")
        traceback.print_exc()
        if "no kernel image" in str(e) or "120" in str(e):
            print("  -> bitsandbytes lacks sm_120 (Blackwell) kernels. "
                  "Upgrade to a recent bitsandbytes build.")

    # ---- verdict ------------------------------------------------------
    section("RESULT")
    if failures:
        print("FAILED checks:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — environment is Blackwell/sm_120 ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
