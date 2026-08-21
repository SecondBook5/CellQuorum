"""De-novo program discovery stage package (consensus NMF)."""

from __future__ import annotations

from cellquorum.discovery.config import DiscoveryConfig
from cellquorum.discovery.nmf_method import NmfMethod
from cellquorum.methods.registry import METHOD_REGISTRY

if not METHOD_REGISTRY.has("discovery", NmfMethod.name):
    METHOD_REGISTRY.register(NmfMethod)

__all__ = [
    "DiscoveryConfig",
    "NmfMethod",
]
