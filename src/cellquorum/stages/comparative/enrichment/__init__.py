"""Enrichment / pathway-activity stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.comparative.enrichment.activity_method import ActivityMethod
from cellquorum.stages.comparative.enrichment.config import EnrichmentConfig
from cellquorum.stages.comparative.enrichment.gsea_method import GseaMethod
from cellquorum.stages.comparative.enrichment.gsva_method import GsvaMethod
from cellquorum.stages.comparative.enrichment.ora_method import OraMethod

for _method in (GseaMethod, OraMethod, GsvaMethod, ActivityMethod):
    if not METHOD_REGISTRY.has("enrichment", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "ActivityMethod",
    "EnrichmentConfig",
    "GseaMethod",
    "GsvaMethod",
    "OraMethod",
]
