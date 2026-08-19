"""Strategy-based analysis method hierarchy for CellQuorum stages."""

from __future__ import annotations

from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.methods.registry import METHOD_REGISTRY, MethodRegistry
from cellquorum.methods.stage_base import MethodDispatchStage

__all__ = [
    "METHOD_REGISTRY",
    "AnalysisMethod",
    "MethodDispatchStage",
    "MethodRegistry",
    "MethodSkip",
    "RAnalysisMethod",
]
