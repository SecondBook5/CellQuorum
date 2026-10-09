"""In-silico perturbation stage with TF and non-TF gene knockout methods."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.gene_regulation.perturbation.celloracle_method import (
    CellOracleMethod,
)
from cellquorum.stages.gene_regulation.perturbation.config import PerturbationConfig
from cellquorum.stages.gene_regulation.perturbation.sctenifoldknk_method import ScTenifoldKnkMethod

if not METHOD_REGISTRY.has("perturbation", "celloracle"):
    METHOD_REGISTRY.register(CellOracleMethod)

if not METHOD_REGISTRY.has("perturbation", "sctenifoldknk"):
    METHOD_REGISTRY.register(ScTenifoldKnkMethod)

__all__ = ["CellOracleMethod", "PerturbationConfig", "ScTenifoldKnkMethod"]
