"""Configuration for the annotation stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AnnotationConfig(StrictBaseModel):
    """Annotation settings."""

    # Whether the annotation stage may run.
    enabled: bool = True

    # Annotation method registry key (marker_vote | celltypist).
    method: str = "marker_vote"

    # obs column holding cluster labels to annotate.
    cluster_key: str = "leiden"

    # Celltype -> marker gene list used by marker-vote scoring.
    marker_panels: dict[str, list[str]] = {}

    # Layer to score on (must be log-normalized).
    score_layer: str = "cellquorum_normalized"

    # obs column that receives the assigned cell-type label.
    key_added: str = "cell_type"

    # Random seed for deterministic scoring.
    random_state: int = 0

    # CellTypist model name or path (required when method == 'celltypist').
    model: str | None = None

    # Counts layer CellTypist normalizes to CP10k-log internally.
    counts_layer: str = "counts"

    # Whether CellTypist majority-voting over-clustering refinement is applied.
    majority_voting: bool = True


__all__ = ["AnnotationConfig"]
