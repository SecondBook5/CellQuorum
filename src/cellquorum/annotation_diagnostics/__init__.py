"""Annotation-diagnostics stage package."""

from __future__ import annotations

from cellquorum.annotation_diagnostics.scdiagnostics_method import (
    ScdiagnosticsMethod,
)
from cellquorum.annotation_diagnostics.stage import AnnotationDiagnosticsStage
from cellquorum.methods.registry import METHOD_REGISTRY

# Register the scDiagnostics method.
if not METHOD_REGISTRY.has("annotation_diagnostics", "scdiagnostics"):
    METHOD_REGISTRY.register(ScdiagnosticsMethod)

__all__ = ["AnnotationDiagnosticsStage", "ScdiagnosticsMethod"]
