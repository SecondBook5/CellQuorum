"""Differential-expression visualization stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.comparative.differential_expression.viz.config import DeVizConfig
from cellquorum.stages.comparative.differential_expression.viz.volcano_viz import VolcanoVizMethod

for _method in (VolcanoVizMethod,):
    if not METHOD_REGISTRY.has("de_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = ["DeVizConfig", "VolcanoVizMethod"]
