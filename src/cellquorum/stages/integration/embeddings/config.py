"""Configuration for the embeddings stage (compute + render + overlay + MAGIC)."""

from __future__ import annotations

from pydantic import Field

from cellquorum.config.base import StrictBaseModel


class MagicConfig(StrictBaseModel):
    """Opt-in MAGIC imputation, scoped to the overlay gene set (viz only).

    Attributes:
        enabled: Whether to run MAGIC on the overlay gene set.
        knn: MAGIC nearest-neighbor count.
        solver: MAGIC solver ('approximate' or 'exact').
        random_state: Seed for reproducible imputation.
    """

    enabled: bool = False
    knn: int = 15
    solver: str = "approximate"
    random_state: int = 0


class OverlayConfig(StrictBaseModel):
    """What features to paint on an embedding. All biology is user-supplied.

    Attributes:
        genes: Gene symbols to color by (one figure each).
        programs: Program name -> gene list; scored via score_genes.
        obs_columns: Existing per-cell obs columns to color by.
        cell_cycle: If true, score cell cycle (requires s_genes + g2m_genes).
        s_genes: S-phase gene list (config-supplied, never defaulted).
        g2m_genes: G2M-phase gene list (config-supplied, never defaulted).
        layer: Expression layer the gene values and program scores are read
            from. This is the same default every other scoring stage in the
            engine declares, and it is a default rather than ``None`` for a
            measured reason: the overlay used to read ``adata.X``, which in this
            engine is raw counts, so a program score written to ``obs`` was
            ``score_genes`` over counts. On the LEC arm that score ran from
            -4.4 to 195.3 in count units and its Spearman with library depth was
            0.23 against 0.07 for the same panel scored on the normalized layer.
            The scores do not stay in the figure — they land in ``obs``, where any
            stage or driver can pick them up as "the capillary score".
    """

    genes: list[str] = []
    programs: dict[str, list[str]] = {}
    obs_columns: list[str] = []
    cell_cycle: bool = False
    s_genes: list[str] = []
    g2m_genes: list[str] = []
    layer: str | None = "cellquorum_normalized"


class EmbeddingsConfig(StrictBaseModel):
    """Compute + render controls for the embeddings stage.

    Carries only structural keys and rendering controls — zero biological
    defaults. Gene/program/label specifics live in ``overlay`` and come from
    the user.

    Attributes:
        enabled: Whether the stage runs.
        use_rep: Representation for PHATE and neighbors fallback.
        umap_min_dist: UMAP min_dist.
        phate_knn: PHATE knn.
        phate_decay: PHATE decay (alpha).
        paga_groupby: obs column for PAGA groups; None -> cell_type else leiden.
        paga_threshold: Minimum connectivity for a drawn PAGA edge.
        random_state: Seed threaded into UMAP/PHATE.
        embeddings: Which bases to render figures for.
        figure_formats: File formats per figure.
        dpi: Raster resolution.
        overlay: Feature-overlay specification.
        magic: Opt-in scoped MAGIC configuration.
    """

    enabled: bool = True
    use_rep: str = "X_pca_harmony"
    umap_min_dist: float = 0.3
    phate_knn: int = 15
    phate_decay: int = 40
    paga_groupby: str | None = None
    paga_threshold: float = 0.2
    random_state: int = 0
    embeddings: list[str] = ["umap", "phate"]
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    magic: MagicConfig = Field(default_factory=MagicConfig)


__all__ = ["EmbeddingsConfig", "OverlayConfig", "MagicConfig"]
