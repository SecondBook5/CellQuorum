"""Differential-abundance stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.comparative.differential_abundance.config import DifferentialAbundanceConfig
from cellquorum.stages.comparative.differential_abundance.milo_method import MiloMethod
from cellquorum.stages.comparative.differential_abundance.propeller_method import PropellerMethod
from cellquorum.stages.comparative.differential_abundance.proportion_ttest_method import (
    ProportionTTestMethod,
)
from cellquorum.stages.comparative.differential_abundance.sccoda_method import SccodaMethod

# Register the propeller method as an import side effect (mirrors differential_expression).
if not METHOD_REGISTRY.has("differential_abundance", "propeller"):
    METHOD_REGISTRY.register(PropellerMethod)

# Register the milo method.
if not METHOD_REGISTRY.has("differential_abundance", "milo"):
    METHOD_REGISTRY.register(MiloMethod)

# Register the sccoda method.
if not METHOD_REGISTRY.has("differential_abundance", "sccoda"):
    METHOD_REGISTRY.register(SccodaMethod)

# Register the proportion_ttest method.
if not METHOD_REGISTRY.has("differential_abundance", "proportion_ttest"):
    METHOD_REGISTRY.register(ProportionTTestMethod)

__all__ = [
    "DifferentialAbundanceConfig",
    "MiloMethod",
    "PropellerMethod",
    "ProportionTTestMethod",
    "SccodaMethod",
]
