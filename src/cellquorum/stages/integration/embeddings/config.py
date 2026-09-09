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
        min_label_frac: A group holding less than this fraction of cells is drawn
            but not named, and gets no PAGA node. Guards against a handful of cells
            being labelled with the same authority as a major lineage, with its
            name landing on top of a cluster it is not.
        legend: Draw a side legend listing every group with its cell count. Covers
            the small groups ``min_label_frac`` leaves unnamed on the plot.
        figure_title: Optional title for the categorical panels. Empty means none.
        color_by: obs columns to colour the atlas by, one panel each. Its own switch:
            empty FOLLOWS the PAGA grouping rather than restating it, but setting it
            stops ``paga_groupby`` from doubling as the atlas's colour decision.
        qc_state_column: obs column holding the QC state (core/borderline/quarantine).
        atlas_states: QC states the atlas panel is restricted to, e.g. ``[core]``. Empty
            draws every cell (the historical behaviour). When set, BOTH the restricted
            panel and an ``_allcells`` panel are written, so what the restriction removed
            is visible rather than merely absent.
        exclude_multiplets: Also drop probable multiplets from the restricted panel.
        multiplet_column: obs column holding the probable-multiplet flag.
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
    min_label_frac: float = 0.001
    legend: bool = True
    figure_title: str = ""
    color_by: list[str] = []
    qc_state_column: str = "qc_state_initial"
    atlas_states: list[str] = []
    # Drop probable multiplets from the restricted atlas panel (they stay in _allcells).
    # A flagged doublet is not one biological cell; it pools in the central mixing zone
    # where unrelated lineages meet and paints it salt-and-pepper.
    exclude_multiplets: bool = True
    multiplet_column: str = "qc_probable_multiplet"
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    magic: MagicConfig = Field(default_factory=MagicConfig)


__all__ = ["EmbeddingsConfig", "OverlayConfig", "MagicConfig"]
