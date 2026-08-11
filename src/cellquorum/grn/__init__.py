"""Gene-regulatory network inference stage (classic pySCENIC)."""

from __future__ import annotations

from cellquorum.grn.config import GrnConfig
from cellquorum.grn.pyscenic_method import PyscenicMethod
from cellquorum.methods.registry import METHOD_REGISTRY

if not METHOD_REGISTRY.has("grn", "pyscenic"):
    METHOD_REGISTRY.register(PyscenicMethod)

__all__ = ["GrnConfig", "PyscenicMethod"]
