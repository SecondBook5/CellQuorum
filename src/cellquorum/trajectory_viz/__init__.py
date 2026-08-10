"""Trajectory visualization stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory_viz.config import TrajectoryVizConfig
from cellquorum.trajectory_viz.driver_viz import DriverVizMethod
from cellquorum.trajectory_viz.fate_viz import FateVizMethod
from cellquorum.trajectory_viz.macrostate_viz import MacrostateVizMethod
from cellquorum.trajectory_viz.pseudotime_viz import PseudotimeVizMethod

# Method classes are appended to this tuple as each figure family lands (Tasks 4-9).
_METHODS: tuple = (PseudotimeVizMethod, FateVizMethod, DriverVizMethod, MacrostateVizMethod)

for _method in _METHODS:
    if not METHOD_REGISTRY.has("trajectory_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "TrajectoryVizConfig",
    "PseudotimeVizMethod",
    "FateVizMethod",
    "DriverVizMethod",
    "MacrostateVizMethod",
]
