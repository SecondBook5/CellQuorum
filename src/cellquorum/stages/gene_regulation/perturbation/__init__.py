"""In-silico perturbation (CellOracle) stage."""

from __future__ import annotations

from cellquorum.stages.gene_regulation.perturbation.celloracle_method import (
    CellOracleMethod,
)
from cellquorum.stages.gene_regulation.perturbation.config import PerturbationConfig
from cellquorum.methods.registry import METHOD_REGISTRY

if not METHOD_REGISTRY.has("perturbation", "celloracle"):
    METHOD_REGISTRY.register(CellOracleMethod)

__all__ = ["CellOracleMethod", "PerturbationConfig"]
