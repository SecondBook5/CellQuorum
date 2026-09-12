"""Verify GPU operations for the selected execution profile.

Usage:
    python scripts/verify_gpu.py --profile rapids
    python scripts/verify_gpu.py --profile scvi

These environment probes do not replace installed-workflow acceptance tests.
"""

from __future__ import annotations

import argparse


def verify_rapids() -> int:
    try:
        import cupy as cp
        import rapids_singlecell

        value = (cp.arange(8, dtype=cp.float32) + 1).sum().item()
        if value != 36:
            raise RuntimeError(f"Unexpected device result: {value}")
        cp.cuda.get_current_stream().synchronize()
        print(f"OK: CuPy {cp.__version__}, RAPIDS single-cell {rapids_singlecell.__version__}")
        print("OK: CUDA allocation, kernel compilation, and reduction succeeded.")
    except Exception as exc:
        print("FAIL: RAPIDS GPU probe:", repr(exc))
        return 1
    return 0


def verify_scvi() -> int:
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
        print("FAIL: scvi-tools import failed:", repr(e))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("scvi", "rapids", "all"), default="scvi")
    args = parser.parse_args()
    results = []
    if args.profile in ("rapids", "all"):
        results.append(verify_rapids())
    if args.profile in ("scvi", "all"):
        results.append(verify_scvi())
    return int(any(results))


if __name__ == "__main__":
    raise SystemExit(main())
