"""Differential-expression visualization stage package."""

from __future__ import annotations

from cellquorum.differential_expression.viz.config import DeVizConfig
from cellquorum.differential_expression.viz.volcano_viz import VolcanoVizMethod
from cellquorum.methods.registry import METHOD_REGISTRY

for _method in (VolcanoVizMethod,):
    if not METHOD_REGISTRY.has("de_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = ["DeVizConfig", "VolcanoVizMethod"]
