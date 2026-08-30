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

    # Source obs column for the 'passthrough' method (preserve an already-trusted
    # label). None means read the trusted label directly from key_added.
    source_key: str | None = None

    # Random seed for deterministic scoring.
    random_state: int = 0

    # CellTypist model name or path (required when method == 'celltypist').
    model: str | None = None

    # Counts layer CellTypist normalizes to CP10k-log internally.
    counts_layer: str = "counts"

    # Whether CellTypist majority-voting over-clustering refinement is applied.
    majority_voting: bool = True

    # Multi-method dispatch: list of per-method sub-configs (each entry is a full
    # method config with its own `method`, `key_added`, etc.). An empty list (the
    # default) means use the scalar `method:` path; only a non-empty list triggers
    # multi-method dispatch, running each entry in order against the same AnnData.
    methods: list[dict] = []


__all__ = ["AnnotationConfig"]
