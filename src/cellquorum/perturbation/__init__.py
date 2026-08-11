"""In-silico perturbation (CellOracle) stage."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.perturbation.celloracle_method import CellOracleMethod
from cellquorum.perturbation.config import PerturbationConfig

if not METHOD_REGISTRY.has("perturbation", "celloracle"):
    METHOD_REGISTRY.register(CellOracleMethod)

__all__ = ["CellOracleMethod", "PerturbationConfig"]
