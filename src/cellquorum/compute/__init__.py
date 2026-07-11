"""GPU/CPU compute routing for CellQuorum stages."""

from __future__ import annotations

from cellquorum.compute.router import (
    gpu_compute_available,
    resolve_compute,
    should_use_gpu,
)

__all__ = ["gpu_compute_available", "resolve_compute", "should_use_gpu"]
