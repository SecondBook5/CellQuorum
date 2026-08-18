"""Configuration for the trajectory-visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel
from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP


class TrajectoryVizConfig(StrictBaseModel):
    """Publication figures rendered from the trajectory producers' outputs.

    All biology comes from producer-written keys/files; this config carries only
    rendering controls and optional name filters (user-supplied, never defaulted
    to any biological value).
    """

    enabled: bool = True
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    top_k: int = 15
    embedding_basis: str | None = None
    pseudotime_keys: list[str] | None = None
    lineages: list[str] | None = None
    genes: list[str] | None = None
    cluster_key: str | None = None

    # Condition-split pseudotime heatmap controls (all optional; biology via config).
    heatmap_genes: list[str] | None = None
    heatmap_score_key: str | None = None
    heatmap_state_key: str | None = None
    heatmap_n_bins: int = 100
    heatmap_max_genes: int = 60
    heatmap_corr_cut: float = 0.1
    heatmap_expr_cmap: str = SEQUENTIAL_CMAP


__all__ = ["TrajectoryVizConfig"]
