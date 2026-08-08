"""Differential-abundance stage package."""

from __future__ import annotations

from cellquorum.differential_abundance.config import DifferentialAbundanceConfig
from cellquorum.differential_abundance.propeller_method import PropellerMethod
from cellquorum.methods.registry import METHOD_REGISTRY

# Register the propeller method as an import side effect (mirrors differential_expression).
if not METHOD_REGISTRY.has("differential_abundance", "propeller"):
    METHOD_REGISTRY.register(PropellerMethod)

__all__ = ["DifferentialAbundanceConfig", "PropellerMethod"]
