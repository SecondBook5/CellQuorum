"""Gene-regulatory network inference stage (classic pySCENIC)."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.gene_regulation.grn.config import GrnConfig
from cellquorum.stages.gene_regulation.grn.pyscenic_method import PyscenicMethod

if not METHOD_REGISTRY.has("grn", "pyscenic"):
    METHOD_REGISTRY.register(PyscenicMethod)

__all__ = ["GrnConfig", "PyscenicMethod"]
