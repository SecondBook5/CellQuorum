"""Differential-expression stage package."""

from __future__ import annotations

from cellquorum.comparative.differential_expression.config import DifferentialExpressionConfig
from cellquorum.comparative.differential_expression.pseudobulk_edger_method import (
    PseudobulkEdgeRMethod,
)
from cellquorum.methods.registry import METHOD_REGISTRY

# Register the pseudobulk edgeR method as an import side effect (mirrors annotation_diagnostics).
if not METHOD_REGISTRY.has("differential_expression", "pseudobulk_edger"):
    METHOD_REGISTRY.register(PseudobulkEdgeRMethod)

__all__ = ["DifferentialExpressionConfig", "PseudobulkEdgeRMethod"]
