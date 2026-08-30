"""Dimensionality-reduction stage (PCA + scree/elbow + auto component selection)."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.preprocessing.dimensionality.pca import PCAMethod
from cellquorum.stages.preprocessing.dimensionality.stage import DimensionalityStage

# Self-register the PCA method so config selection can resolve it. Guard against
# double registration when the module is imported more than once.
if not METHOD_REGISTRY.has("dimensionality", "pca"):
    METHOD_REGISTRY.register(PCAMethod)

__all__ = ["DimensionalityStage", "PCAMethod"]
