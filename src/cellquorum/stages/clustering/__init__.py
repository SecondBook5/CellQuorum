"""Clustering stage (kNN neighbors graph + Leiden community detection)."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.clustering.neighbors_leiden import LeidenMethod
from cellquorum.stages.clustering.stage import ClusteringStage

# Self-register the Leiden method. Guard against double registration.
if not METHOD_REGISTRY.has("clustering", "leiden"):
    METHOD_REGISTRY.register(LeidenMethod)

__all__ = ["ClusteringStage", "LeidenMethod"]
