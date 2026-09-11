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

from cellquorum.core.exceptions import CellQuorumBackendError

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
    """Resolve GPU use while enforcing the configured fallback policy."""
    return resolve_compute(context)["use_gpu"]


def resolve_compute(context: object) -> dict:
    """Resolve compute, raising when required GPU support is unavailable.

    Explicit GPU requests that permit CPU fallback carry a human-readable reason.
    Automatic selection may choose CPU without treating that as a failed request.
    """
    backend, prefer_gpu, fallback = _compute_settings(context)
    wants_gpu = backend in {"gpu", "rapids"} or (backend == "auto" and prefer_gpu)
    use_gpu = wants_gpu and gpu_compute_available()
    result = {"use_gpu": bool(use_gpu), "fallback_to_cpu": bool(fallback)}
    if wants_gpu and not use_gpu:
        reason = "CuPy, rapids-singlecell, or a working CUDA device is unavailable."
        if not fallback:
            raise CellQuorumBackendError(f"GPU compute is required: {reason}")
        if backend in {"gpu", "rapids"}:
            result["fallback_reason"] = f"Requested GPU compute fell back to CPU: {reason}"
    return result


__all__ = ["gpu_compute_available", "resolve_compute", "should_use_gpu"]
