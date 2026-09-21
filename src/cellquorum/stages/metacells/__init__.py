"""Metacell (MetaCell2) partitioning stage."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.metacells.config import MetacellsConfig
from cellquorum.stages.metacells.metacells_method import MetacellsMethod

if not METHOD_REGISTRY.has("metacells", "metacells"):
    METHOD_REGISTRY.register(MetacellsMethod)

__all__ = ["MetacellsConfig", "MetacellsMethod"]
