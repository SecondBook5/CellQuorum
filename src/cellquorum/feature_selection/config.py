"""Configuration for the feature-selection (HVG) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class FeatureSelectionConfig(StrictBaseModel):
    """Highly-variable-gene selection settings."""

    # Whether the feature-selection stage may run.
    enabled: bool = True

    # HVG method registry key (seurat | seurat_v3 | pearson_residuals).
    method: str = "seurat"

    # Number of top HVGs to flag.
    n_top_genes: int = 2000

    # Counts layer for count-based flavors (seurat_v3 / pearson_residuals).
    counts_layer: str = "counts"

    # Log-normalized layer for the seurat (v1) flavor.
    lognorm_layer: str = "cellquorum_normalized"

    # Optional batch key for batch-aware HVG selection.
    batch_key: str | None = None

    # var_name regex patterns to exclude from HVG (e.g. MT-/ribo/hb/sex-linked).
    exclude_gene_patterns: list[str] = []

    # Flag var["highly_variable"] only; never subset the object.
    flag_only: bool = True


__all__ = ["FeatureSelectionConfig"]
