"""Co-expression network analysis stage.

Produces gene co-expression modules via hdWGCNA and publication-grade figure
primitives for module UMAPs and network visualizations. Method registration
added in Task 6.
"""

from __future__ import annotations

from cellquorum.gene_regulation.coexpression.config import CoexpressionConfig
from cellquorum.gene_regulation.coexpression.hdwgcna_method import HdwgcnaMethod
from cellquorum.methods.registry import METHOD_REGISTRY

if not METHOD_REGISTRY.has("coexpression", "hdwgcna"):
    METHOD_REGISTRY.register(HdwgcnaMethod)

__all__ = ["CoexpressionConfig", "HdwgcnaMethod"]
