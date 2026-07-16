"""Batch-integration stage (Harmony CPU; scVI/scANVI GPU-gated)."""

from __future__ import annotations

from cellquorum.integration.harmony import HarmonyMethod
from cellquorum.integration.scanvi_methods import ScANVIMethod
from cellquorum.integration.scvi_methods import ScVIMethod
from cellquorum.integration.stage import IntegrationStage
from cellquorum.methods.registry import METHOD_REGISTRY

# Self-register integration methods (guarded against double registration).
if not METHOD_REGISTRY.has("integration", "harmony"):
    METHOD_REGISTRY.register(HarmonyMethod)
if not METHOD_REGISTRY.has("integration", "scvi"):
    METHOD_REGISTRY.register(ScVIMethod)
if not METHOD_REGISTRY.has("integration", "scanvi"):
    METHOD_REGISTRY.register(ScANVIMethod)

__all__ = ["HarmonyMethod", "IntegrationStage", "ScANVIMethod", "ScVIMethod"]
