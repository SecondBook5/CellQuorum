"""Trajectory visualization stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.viz import (
    _helpers as inputs,  # re-export as 'inputs' for backward compat
)
from cellquorum.trajectory.viz import _helpers as plots  # re-export as 'plots' for backward compat
from cellquorum.trajectory.viz._kernel_plots import (
    DriverVizMethod,
    FateVizMethod,
    MacrostateVizMethod,
    VelocityVizMethod,
)
from cellquorum.trajectory.viz._pseudotime_plots import (
    GeneTrendVizMethod,
    PseudotimeHeatmapVizMethod,
    PseudotimeVizMethod,
)
from cellquorum.trajectory.viz.config import TrajectoryVizConfig

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
    "inputs",  # backward compat alias for _helpers
    "plots",  # backward compat alias for _helpers
]
