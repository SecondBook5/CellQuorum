"""Configuration for the metacells (MetaCell2) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class MetacellsConfig(StrictBaseModel):
    """MetaCell2 partitioning into near-homogeneous metacells, via an isolated env.

    Partitions the annotated object into metacells and an explicit outlier
    bucket, feeding a family of atlas-level QC/identity figures (a metacell
    similarity network, a gene x metacell heatmap with cell-type composition,
    unclassified-cell QC scatters, and optional per-gene marker traces).
    Off by default: this is a diagnostic, not a required pipeline step.

    Attributes:
        enabled: Whether the stage runs (off by default).
        method: Metacell method registry key (metacells).
        rep_key: obsm key holding the PCA/kNN representation MetaCell2 builds
            its similarity graph from.
        layer: Layer holding raw counts.
        group_by: obs column used as the color-bar/purity label for figures.
            Defaults to the lymphedema atlas's coarse reference-mapped label,
            fully overridable per project.
        target_metacell_size: MetaCell2's own tunable target cells/metacell.
        min_cells: Minimum total cells required to attempt partitioning.
        marker_genes: Optional gene list; drives the per-gene marker-trace
            figures only. Empty list means that figure is not rendered.
        seed: Random seed for reproducibility.
        env_name: Name of the isolated micromamba environment.
        launcher: Environment launcher (micromamba).
        timeout_seconds: MetaCell2 execution timeout in seconds.
    """

    enabled: bool = False
    method: str = "metacells"
    rep_key: str = "X_pca_harmony"
    layer: str = "counts"
    group_by: str = "ref_cell_type_granular_coarse"
    target_metacell_size: int = 96
    min_cells: int = 5000
    marker_genes: list[str] = []
    seed: int = 0
    env_name: str = "metacells_env"
    launcher: str = "micromamba"
    timeout_seconds: int = 3600


__all__ = ["MetacellsConfig"]
