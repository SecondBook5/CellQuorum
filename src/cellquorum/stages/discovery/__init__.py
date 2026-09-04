"""De-novo program discovery stage package (consensus NMF)."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.discovery.config import DiscoveryConfig
from cellquorum.stages.discovery.nmf_method import NmfMethod

if not METHOD_REGISTRY.has("discovery", NmfMethod.name):
    METHOD_REGISTRY.register(NmfMethod)

__all__ = [
    "DiscoveryConfig",
    "NmfMethod",
]
