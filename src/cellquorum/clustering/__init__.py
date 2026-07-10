"""Clustering stage (kNN neighbors graph + Leiden community detection)."""

from __future__ import annotations

from cellquorum.clustering.neighbors_leiden import LeidenMethod
from cellquorum.clustering.stage import ClusteringStage
from cellquorum.methods.registry import METHOD_REGISTRY

# Self-register the Leiden method. Guard against double registration.
if not METHOD_REGISTRY.has("clustering", "leiden"):
    METHOD_REGISTRY.register(LeidenMethod)

__all__ = ["ClusteringStage", "LeidenMethod"]
