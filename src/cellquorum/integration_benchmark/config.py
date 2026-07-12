"""Configuration for the integration-benchmark evaluation stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class IntegrationBenchmarkConfig(StrictBaseModel):
    """Integration-quality evaluation via scib-metrics.

    Opt-in stage (off by default). Measures batch-correction quality
    (iLISI/kBET/pcr) and biological-structure preservation (cLISI/silhouette/
    graph-connectivity/NMI) over multiple integration embeddings. READ-ONLY:
    never modifies obsm/obs. Returns ranking + per-embedding metrics as
    StageResult.metrics only.

    Attributes:
        enabled: Whether the stage runs (opt-in; off by default).
        method: Benchmark method registry key (scib_benchmark).
        batch_key: obs column identifying batches to correct.
        label_key: obs column identifying cell types (biological labels).
        label_key_fallback: Fallback label column if label_key is missing.
        pre_embedding: obsm key for the pre-integration embedding (X_pca).
        embeddings: List of integration embedding obsm keys to evaluate.
        n_neighbors: Number of neighbors for kNN-based metrics.
        mode: Evaluation mode (full=batch+bio; batch_only=batch metrics only).
        batch_weight: Weight for batch-correction metrics in aggregate score.
        bio_weight: Weight for bio-preservation metrics in aggregate score.
    """

    # Whether the integration-benchmark stage runs (opt-in; off by default).
    enabled: bool = False

    # Benchmark method registry key (scib_benchmark | fallback).
    method: str = "scib_benchmark"

    # obs column identifying batches to correct for.
    batch_key: str = "batch"

    # obs column identifying cell types (biological labels).
    label_key: str = "cell_type"

    # Fallback label column if label_key is missing (batch-only mode if both absent).
    label_key_fallback: str | None = None

    # obsm key for the pre-integration embedding (e.g. X_pca).
    pre_embedding: str = "X_pca"

    # List of integration embedding obsm keys to evaluate.
    embeddings: list[str] = ["X_pca_harmony", "X_pca_scanorama"]

    # Number of neighbors for kNN-based metrics (ilisi/clisi/kbet/connectivity/nmi).
    n_neighbors: int = 90

    # Evaluation mode: full (batch+bio) or batch_only (batch metrics only).
    mode: str = "full"

    # Weight for batch-correction metrics in aggregate score (0-1).
    batch_weight: float = 0.4

    # Weight for bio-preservation metrics in aggregate score (0-1).
    bio_weight: float = 0.6


__all__ = ["IntegrationBenchmarkConfig"]
