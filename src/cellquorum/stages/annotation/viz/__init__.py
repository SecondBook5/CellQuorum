"""Annotation figure stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.annotation.viz.config import AnnotationVizConfig
from cellquorum.stages.annotation.viz.methods import MarkerDotplotMethod

for _method in (MarkerDotplotMethod,):
    if not METHOD_REGISTRY.has("annotation_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = ["AnnotationVizConfig", "MarkerDotplotMethod"]
