"""Trajectory stage package (spec #1: loom I/O + RNA velocity)."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.cellrank_method import CellRankMethod
from cellquorum.trajectory.config import (
    CellRankConfig,
    CytoTraceConfig,
    TrajectoryConfig,
    VelocityConfig,
    VelocityGenerationConfig,
)
from cellquorum.trajectory.cytotrace_method import CytoTraceMethod
from cellquorum.trajectory.dpt_method import DptMethod
from cellquorum.trajectory.palantir_method import PalantirMethod
from cellquorum.trajectory.velocity_method import VelocityMethod

for _method in (VelocityMethod, CellRankMethod, DptMethod, PalantirMethod, CytoTraceMethod):
    if not METHOD_REGISTRY.has("trajectory", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "CellRankConfig",
    "CellRankMethod",
    "CytoTraceConfig",
    "CytoTraceMethod",
    "DptMethod",
    "PalantirMethod",
    "TrajectoryConfig",
    "VelocityConfig",
    "VelocityGenerationConfig",
    "VelocityMethod",
]
