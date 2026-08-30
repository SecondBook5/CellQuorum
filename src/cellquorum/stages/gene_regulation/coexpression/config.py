"""Configuration for the co-expression (hdWGCNA) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class CoexpressionConfig(StrictBaseModel):
    """hdWGCNA co-expression module discovery via an isolated R environment.

    Builds metacells, detects co-expression modules, and renders a module-UMAP
    figure. Runs hdWGCNA in an isolated micromamba env; skips cleanly when the
    env or required R packages are unavailable.

    Attributes:
        enabled: Whether the stage runs (enabled by default).
        method: Co-expression method registry key (hdwgcna).
        layer: Layer holding raw counts for hdWGCNA.
        group_by: Optional grouping variable for stratified analysis.
        condition_col: Optional condition column for design matrix.
        n_hvg: Number of highly variable genes to use.
        k: Number of neighbors for metacell construction.
        min_cells: Minimum cells per metacell.
        min_cells_total: Minimum total cells required for analysis.
        soft_power: Optional soft power threshold for network construction.
        seed: Random seed for reproducibility.
        env_name: Name of the isolated micromamba environment.
        launcher: Environment launcher (micromamba).
        timeout_seconds: R execution timeout in seconds.
        r_packages: List of R packages required for hdWGCNA.
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected co-expression method (registry key under stage_category 'coexpression').
    method: str = "hdwgcna"

    # Layer holding raw counts for hdWGCNA.
    layer: str = "counts"

    # Optional grouping variable for stratified analysis.
    group_by: str | None = None

    # Optional condition column for design matrix.
    condition_col: str | None = None

    # Number of highly variable genes to use.
    n_hvg: int = 3000

    # Number of neighbors for metacell construction.
    k: int = 25

    # Minimum cells per metacell.
    min_cells: int = 50

    # Minimum total cells required for analysis.
    min_cells_total: int = 100

    # Optional soft power threshold for network construction.
    soft_power: int | None = None

    # Random seed for reproducibility.
    seed: int = 0

    # Name of the isolated micromamba environment.
    env_name: str = "hdwgcna_env"

    # Environment launcher (micromamba).
    launcher: str = "micromamba"

    # R execution timeout (seconds).
    timeout_seconds: int = 3600

    # R packages required for hdWGCNA.
    r_packages: list[str] = ["hdWGCNA", "Seurat", "WGCNA", "zellkonverter"]


__all__ = ["CoexpressionConfig"]
