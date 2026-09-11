"""Configuration for the feature-selection (HVG) stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class FeatureSelectionConfig(StrictBaseModel):
    """Highly-variable-gene selection settings.

    Opt-in stage: off by default. When enabled it flags var['highly_variable'] but never
    subsets the object; PCA and scVI read the flag and restrict themselves to it.

    Turned on in ONE place -- ``stages.feature_selection: true``. Consumers used to need
    their own repeat of the decision, which is how a run reached this stage at position 4
    of 36, skipped it, and then built both the PCA basis and the scVI latent space from all
    ~33,000 genes without a word.
    """

    # Whether the feature-selection stage runs. Kept in step with `stages.feature_selection`
    # by CellQuorumConfig.reconcile_stage_switches; declare it there, not here.
    enabled: bool = False

    # HVG method registry key (seurat_v3 | pearson_residuals | seurat).
    # seurat_v3 (default) operates on raw counts; seurat (v1) operates on lognorm.
    method: str = "seurat_v3"

    # Number of top HVGs to flag. 3,000 rather than scvi-tools' tutorial default of 2,000:
    # still squarely standard practice, with more margin against dropping a biologically
    # relevant gene that scores as low-variance.
    n_top_genes: int = 3000

    # Counts layer for count-based flavors (seurat_v3 / pearson_residuals).
    counts_layer: str = "counts"

    # Log-normalized layer for the seurat (v1) flavor.
    lognorm_layer: str = "cellquorum_normalized"

    # Optional batch key for batch-aware HVG selection.
    batch_key: str | None = None

    # var_name regex patterns to exclude from HVG (e.g. MT-/ribo/hb/sex-linked).
    exclude_gene_patterns: list[str] = []

    # Whether to write the mean-vs-dispersion HVG diagnostic figure.
    write_figures: bool = True


__all__ = ["FeatureSelectionConfig"]
