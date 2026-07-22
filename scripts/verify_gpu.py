"""Verify the cellquorum-gpu env can actually use the GPU for scVI/scArches.

Usage:
    python scripts/verify_gpu.py
Exits non-zero (and explains) if PyTorch cannot see CUDA — the condition that
would make scVI/scArches fail under compute.backend: gpu.
"""

from __future__ import annotations


def main() -> int:
    try:
        import torch
    except Exception as e:  # pragma: no cover - env probe
        print("FAIL: could not import torch:", repr(e))
        return 1

    cuda_build = torch.version.cuda
    built = torch.backends.cuda.is_built()
    avail = torch.cuda.is_available()
    print(
        f"torch {torch.__version__}  cuda_build={cuda_build}  "
        f"is_built={built}  is_available={avail}"
    )

    if cuda_build is None or not built:
        print("FAIL: this is a CPU-only PyTorch build (torch.version.cuda is None).")
        print("      scVI/scArches will fail under compute.backend: gpu.")
        return 1
    if not avail:
        print("FAIL: CUDA build present but torch.cuda.is_available() is False.")
        print("      Check the driver/WSL CUDA runtime is visible to this shell.")
        return 1

    # Prove a real device op works.
    try:
        x = torch.zeros(8, device="cuda")
        _ = (x + 1).sum().item()
        print("OK:", torch.cuda.get_device_name(0), "- CUDA tensor op succeeded.")
    except Exception as e:  # pragma: no cover - env probe
        print("FAIL: CUDA reported available but a device op failed:", repr(e))
        return 1

    # scvi import sanity (the actual consumer).
    try:
        import scvi  # noqa: F401

        print("OK: scvi-tools importable.")
    except Exception as e:  # pragma: no cover - env probe
        print("WARN: scvi-tools import failed:", repr(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
