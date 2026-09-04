"""Module remodeling — condition effects on per-cell module activity, with figures.

The statistics live in :mod:`cellquorum.stats` so a notebook can call them
directly; this package is the stage that runs them inside a pipeline, names the
group axis, and draws the panels. ``state_scoring`` scores the modules and emits
no figures, which is the gap this closes.
"""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.comparative.module_remodeling.config import ModuleRemodelingConfig
from cellquorum.stages.comparative.module_remodeling.remodeling_method import (
    ModuleRemodelingMethod,
)

for _method in (ModuleRemodelingMethod,):
    if not METHOD_REGISTRY.has("module_remodeling", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "ModuleRemodelingConfig",
    "ModuleRemodelingMethod",
]
