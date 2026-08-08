"""Enrichment / pathway-activity stage package."""

from __future__ import annotations

from cellquorum.enrichment.activity_method import ActivityMethod
from cellquorum.enrichment.config import EnrichmentConfig
from cellquorum.enrichment.gsea_method import GseaMethod
from cellquorum.enrichment.gsva_method import GsvaMethod
from cellquorum.enrichment.ora_method import OraMethod
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
