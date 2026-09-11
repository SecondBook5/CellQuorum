"""Shared GPU self-gating for the VAE-based integration methods (scVI, scANVI).

Both methods are GPU-oriented and must fail with a clear, actionable error rather than
crash deep inside training when no GPU backend is available. Extracted so the two cannot
drift on the check or the message.
"""

from __future__ import annotations

from cellquorum.core.exceptions import CellQuorumStageError


def require_gpu(context: object, *, method_name: str) -> None:
    """Raise unless the context's backend registry reports a GPU as available.

    Args:
        context: The stage context; read via ``backend_registry.available("gpu")``
            when present. A missing or malformed registry is treated as "no GPU".
        method_name: "scVI" or "scANVI", for the message.

    Raises:
        CellQuorumStageError: If no GPU backend is available.
    """
    registry = getattr(context, "backend_registry", None)
    gpu_ok = False
    if registry is not None and hasattr(registry, "available"):
        try:
            gpu_ok = bool(registry.available("gpu"))
        except Exception:
            gpu_ok = False
    if not gpu_ok:
        raise CellQuorumStageError(
            "integration",
            f"{method_name} integration requires a GPU backend, which is unavailable. "
            "Use method='harmony' for CPU integration.",
        )


__all__ = ["require_gpu"]
