"""Annotation stage (marker-vote CPU default; CellTypist optional/future)."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.annotation.celltypist_method import CellTypistMethod
from cellquorum.stages.annotation.marker_vote import MarkerVoteMethod
from cellquorum.stages.annotation.passthrough import PassthroughAnnotationMethod
from cellquorum.stages.annotation.stage import AnnotationStage

# Self-register the marker-vote method (guarded against double registration).
if not METHOD_REGISTRY.has("annotation", "marker_vote"):
    METHOD_REGISTRY.register(MarkerVoteMethod)

# Self-register the celltypist method (guarded against double registration).
if not METHOD_REGISTRY.has("annotation", "celltypist"):
    METHOD_REGISTRY.register(CellTypistMethod)

# Self-register the passthrough method (guarded against double registration).
if not METHOD_REGISTRY.has("annotation", "passthrough"):
    METHOD_REGISTRY.register(PassthroughAnnotationMethod)

__all__ = [
    "AnnotationStage",
    "MarkerVoteMethod",
    "CellTypistMethod",
    "PassthroughAnnotationMethod",
]
