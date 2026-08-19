"""Trajectory visualization stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.viz.config import TrajectoryVizConfig
from cellquorum.trajectory.viz.driver_viz import DriverVizMethod
from cellquorum.trajectory.viz.fate_viz import FateVizMethod
from cellquorum.trajectory.viz.gene_trend_viz import GeneTrendVizMethod
from cellquorum.trajectory.viz.macrostate_viz import MacrostateVizMethod
from cellquorum.trajectory.viz.pseudotime_heatmap_viz import PseudotimeHeatmapVizMethod
from cellquorum.trajectory.viz.pseudotime_viz import PseudotimeVizMethod
from cellquorum.trajectory.viz.velocity_viz import VelocityVizMethod

# Method classes are appended to this tuple as each figure family lands (Tasks 4-9).
_METHODS: tuple = (
    PseudotimeVizMethod,
    FateVizMethod,
    DriverVizMethod,
    GeneTrendVizMethod,
    MacrostateVizMethod,
    VelocityVizMethod,
    PseudotimeHeatmapVizMethod,
)

for _method in _METHODS:
    if not METHOD_REGISTRY.has("trajectory_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "TrajectoryVizConfig",
    "PseudotimeVizMethod",
    "FateVizMethod",
    "DriverVizMethod",
    "GeneTrendVizMethod",
    "MacrostateVizMethod",
    "VelocityVizMethod",
    "PseudotimeHeatmapVizMethod",
]
