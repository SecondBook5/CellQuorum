"""Embeddings stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.integration.embeddings.config import (
    EmbeddingsConfig,
    MagicConfig,
    OverlayConfig,
)
from cellquorum.stages.integration.embeddings.methods import (
    CategoricalEmbeddingMethod,
    ContinuousOverlayMethod,
    PagaMethod,
    PhateMethod,
    UmapMethod,
)

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
