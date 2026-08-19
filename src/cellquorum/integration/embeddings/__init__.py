"""Embeddings stage package."""

from __future__ import annotations

from cellquorum.integration.embeddings.categorical_method import CategoricalEmbeddingMethod
from cellquorum.integration.embeddings.config import EmbeddingsConfig, MagicConfig, OverlayConfig
from cellquorum.integration.embeddings.overlay_method import ContinuousOverlayMethod
from cellquorum.integration.embeddings.paga_method import PagaMethod
from cellquorum.integration.embeddings.phate_method import PhateMethod
from cellquorum.integration.embeddings.umap_method import UmapMethod
from cellquorum.methods.registry import METHOD_REGISTRY

for _method in (
    UmapMethod,
    PhateMethod,
    PagaMethod,
    CategoricalEmbeddingMethod,
    ContinuousOverlayMethod,
):
    if not METHOD_REGISTRY.has("embeddings", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "CategoricalEmbeddingMethod",
    "ContinuousOverlayMethod",
    "EmbeddingsConfig",
    "MagicConfig",
    "OverlayConfig",
    "PagaMethod",
    "PhateMethod",
    "UmapMethod",
]
