"""Embeddings stage package."""

from __future__ import annotations

from cellquorum.embeddings.categorical_method import CategoricalEmbeddingMethod
from cellquorum.embeddings.config import EmbeddingsConfig, MagicConfig, OverlayConfig
from cellquorum.embeddings.overlay_method import ContinuousOverlayMethod
from cellquorum.embeddings.paga_method import PagaMethod
from cellquorum.embeddings.phate_method import PhateMethod
from cellquorum.embeddings.umap_method import UmapMethod
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
