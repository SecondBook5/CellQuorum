"""Enrichment visualization stage package."""

from __future__ import annotations

from cellquorum.enrichment_viz.activity_viz import ActivityVizMethod
from cellquorum.enrichment_viz.config import EnrichmentVizConfig
from cellquorum.enrichment_viz.gsea_viz import GseaVizMethod
from cellquorum.enrichment_viz.gsva_viz import GsvaVizMethod
from cellquorum.enrichment_viz.ora_viz import OraVizMethod
from cellquorum.methods.registry import METHOD_REGISTRY

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
