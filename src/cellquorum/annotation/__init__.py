"""Annotation stage (marker-vote CPU default; CellTypist optional/future)."""

from __future__ import annotations

from cellquorum.annotation.marker_vote import MarkerVoteMethod
from cellquorum.annotation.stage import AnnotationStage
from cellquorum.methods.registry import METHOD_REGISTRY

# Self-register the marker-vote method (guarded against double registration).
if not METHOD_REGISTRY.has("annotation", "marker_vote"):
    METHOD_REGISTRY.register(MarkerVoteMethod)

__all__ = ["AnnotationStage", "MarkerVoteMethod"]
