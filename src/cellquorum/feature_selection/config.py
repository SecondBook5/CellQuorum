"""Configuration for the feature-selection (HVG) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class FeatureSelectionConfig(StrictBaseModel):
    """Highly-variable-gene selection settings.

    Opt-in stage: off by default. When enabled, it flags var['highly_variable']
    but never subsets the object. To consume the HVGs, also set
    dimensionality.use_highly_variable: true in the config.
    """

    # Whether the feature-selection stage runs (opt-in; off by default).
    enabled: bool = False

    # HVG method registry key (seurat_v3 | pearson_residuals | seurat).
    # seurat_v3 (default) operates on raw counts; seurat (v1) operates on lognorm.
    method: str = "seurat_v3"

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


__all__ = ["FeatureSelectionConfig"]
