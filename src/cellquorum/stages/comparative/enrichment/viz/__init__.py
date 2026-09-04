"""Enrichment visualization stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.comparative.enrichment.viz.config import EnrichmentVizConfig
from cellquorum.stages.comparative.enrichment.viz.viz_methods import (
    ActivityVizMethod,
    GseaVizMethod,
    GsvaVizMethod,
    OraVizMethod,
)

for _method in (GseaVizMethod, OraVizMethod, GsvaVizMethod, ActivityVizMethod):
    if not METHOD_REGISTRY.has("enrichment_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "ActivityVizMethod",
    "EnrichmentVizConfig",
    "GseaVizMethod",
    "GsvaVizMethod",
    "OraVizMethod",
]
