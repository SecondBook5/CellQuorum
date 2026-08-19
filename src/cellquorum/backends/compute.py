"""Decide whether a stage should run on GPU (rapids-singlecell) or CPU (scanpy).

The decision combines two things:
  1. real capability — can rapids_singlecell + cupy actually import and see a
     CUDA device? (An NVIDIA device being visible is NOT enough: the CPU env has
     a device but no RAPIDS, so it must route to CPU.)
  2. config preference — compute.backend / prefer_gpu, with backend="cpu" as a
     hard escape hatch that forces CPU regardless of hardware.

This is the single place that answer lives, so every stage routes consistently.
"""

from __future__ import annotations

# Cache the capability probe result for the process (it cannot change mid-run).
_GPU_AVAILABLE: bool | None = None


def gpu_compute_available() -> bool:
    """
    Return True iff rapids-singlecell + cupy import and a CUDA device is present.

    Never raises: any import error or CUDA failure is treated as "no GPU".

    Returns:
        Whether GPU compute (rapids-singlecell) is usable in this process.
    """

    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    # Probe once. rapids_singlecell import + a real cupy device count.
    available = False
    try:
        import cupy  # noqa: F401
        import rapids_singlecell  # noqa: F401

        available = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        available = False

    _GPU_AVAILABLE = available
    return available


def _compute_settings(context: object) -> tuple[str, bool, bool]:
    """Extract (backend, prefer_gpu, fallback_to_cpu) from a context, with defaults."""

    # ComputeConfig defaults: backend="auto", prefer_gpu=True, fallback_to_cpu=True.
    config = getattr(context, "config", None)
    compute = getattr(config, "compute", None) if config is not None else None
    if compute is None:
        return "auto", True, True

    # Support both pydantic ComputeConfig and a plain dict.
    if isinstance(compute, dict):
        return (
            compute.get("backend", "auto"),
            compute.get("prefer_gpu", True),
            compute.get("fallback_to_cpu", True),
        )
    return (
        getattr(compute, "backend", "auto"),
        getattr(compute, "prefer_gpu", True),
        getattr(compute, "fallback_to_cpu", True),
    )


def should_use_gpu(context: object) -> bool:
    """
    Return whether a stage should route to GPU compute for this run.

    Args:
        context: Pipeline context exposing config.compute (ComputeConfig/dict/None).

    Returns:
        True iff the config permits GPU AND GPU compute is actually available.
    """

    # Resolve the config preference.
    backend, prefer_gpu, _ = _compute_settings(context)

    # Escape hatch: an explicit CPU request forces CPU regardless of hardware.
    if backend == "cpu":
        return False

    # Decide whether the config *wants* GPU.
    wants_gpu = backend in {"gpu", "rapids"} or (backend == "auto" and bool(prefer_gpu))

    # Only use GPU when both wanted AND actually available.
    return wants_gpu and gpu_compute_available()


def resolve_compute(context: object) -> dict:
    """
    Return the routing decision for a method as a small dict.

    Args:
        context: Pipeline context.

    Returns:
        {"use_gpu": bool, "fallback_to_cpu": bool}.
    """

    # Combine the decision with the fallback policy.
    _, _, fallback = _compute_settings(context)
    return {"use_gpu": should_use_gpu(context), "fallback_to_cpu": bool(fallback)}


__all__ = ["gpu_compute_available", "resolve_compute", "should_use_gpu"]
