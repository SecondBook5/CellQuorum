"""Enrichment / pathway-activity stage package."""

from __future__ import annotations

from cellquorum.comparative.enrichment.activity_method import ActivityMethod
from cellquorum.comparative.enrichment.config import EnrichmentConfig
from cellquorum.comparative.enrichment.gsea_method import GseaMethod
from cellquorum.comparative.enrichment.gsva_method import GsvaMethod
from cellquorum.comparative.enrichment.ora_method import OraMethod
from cellquorum.methods.registry import METHOD_REGISTRY

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
