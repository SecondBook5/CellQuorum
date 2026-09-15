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

    # Optional cluster/subcluster id -> human-readable identity label, e.g.
    # {"0": "Perivascular/FRC-like, Inflammatory/chemokine"}. CellRank names a
    # macrostate/lineage after the dominant cluster id it was seeded from
    # (optionally suffixed "_1"/"_2" when one cluster splits into more than
    # one macrostate), so driver_viz strips that suffix and looks the base id
    # up here, appending the label to the bare numeric title/axis tick rather
    # than replacing it. None (default) leaves every title as the bare id --
    # this is user-declared biology, never defaulted, so the engine stays
    # biology-agnostic when a manifest doesn't supply it.
    fate_labels: dict[str, str] | None = None

    # Condition-split pseudotime heatmap controls (all optional; biology via config).
    heatmap_genes: list[str] | None = None
    heatmap_score_key: str | None = None
    heatmap_state_key: str | None = None
    heatmap_n_bins: int = 100
    heatmap_max_genes: int = 60
    heatmap_corr_cut: float = 0.1
    heatmap_expr_cmap: str = SEQUENTIAL_CMAP

    # Activity-along-pseudotime cascade controls (all optional; nets via config).
    # Which prior-knowledge nets to score per cell and cascade along pseudotime
    # (e.g. ["progeny", "collectri", "hallmark", "dorothea"]); None → method default.
    activity_resources: list[str] | None = None
    # Per-net decoupler method override, e.g. {"progeny": "mlm"}; None → method default.
    activity_methods: dict[str, str] | None = None
    # Per-net cap on top-|rho| sources shown, e.g. {"hallmark": 10}; None → method default.
    cascade_top: dict[str, int | None] | None = None
    cascade_n_bins: int = 20
    # Optional pseudotime-axis annotation, e.g. "basal → terminal".
    cascade_xlab: str | None = None


__all__ = ["TrajectoryVizConfig"]
