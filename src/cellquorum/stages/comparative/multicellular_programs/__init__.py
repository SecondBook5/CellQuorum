"""Multicellular programs (DIALOGUE) — cross-cell-type coordinated programs."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.comparative.multicellular_programs.config import MulticellularProgramsConfig
from cellquorum.stages.comparative.multicellular_programs.dialogue_method import (
    MulticellularProgramsMethod,
)

for _method in (MulticellularProgramsMethod,):
    if not METHOD_REGISTRY.has("multicellular_programs", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "MulticellularProgramsConfig",
    "MulticellularProgramsMethod",
]
