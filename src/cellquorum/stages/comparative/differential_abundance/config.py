"""Configuration for the differential-abundance stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DifferentialAbundanceConfig(StrictBaseModel):
    """Cell-type abundance differential-abundance analysis.

    Tests for statistically significant changes in cell-type proportions
    across experimental groups. Produces group-level abundance statistics,
    effect sizes, and confidence intervals.

    Attributes:
        enabled: Whether the stage runs (enabled by default).
        methods: DA method registry keys (empty for defaults selected per dataset).
        cell_type_col: Observation metadata column containing cell-type labels.
        use_rep: Embedding representation to use for DA computations.
        k: Number of neighbors for compositional DA methods.
        prop: Proportion threshold for abundance filtering.
        spatial_fdr: FDR threshold for spatial DA tests.
        reference_celltype: Reference cell type for compositional analysis (None for
            dataset default).
        seed: Random seed for reproducibility.
        num_iterations: Number of iterations for DA fitting algorithms.
        inclusion_prob_threshold: Threshold for including samples in DA analysis.
        transform: Transformation applied to abundance data (e.g., asin for proportions).
        fdr: FDR threshold recorded in outputs.
        timeout_seconds: Execution timeout in seconds.
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected DA methods (registry keys under stage_category 'differential_abundance').
    methods: list[dict] = []

    # Observation column containing cell-type labels.
    cell_type_col: str = "cell_type"

    # Embedding representation for DA computations.
    use_rep: str = "X_pca_harmony"

    # Number of neighbors for compositional DA methods.
    k: int = 30

    # Proportion threshold for abundance filtering.
    prop: float = 0.1

    # FDR threshold for spatial DA tests.
    spatial_fdr: float = 0.1

    # Reference cell type for compositional analysis (None for dataset default).
    reference_celltype: str | None = None

    # Random seed for reproducibility.
    seed: int = 0

    # Number of iterations for DA fitting algorithms.
    num_iterations: int = 20000

    # Threshold for including samples in DA analysis.
    inclusion_prob_threshold: float = 0.8

    # Transformation applied to abundance data.
    transform: str = "asin"

    # FDR threshold recorded in outputs.
    fdr: float = 0.05

    # Execution timeout (seconds).
    timeout_seconds: int = 1800


__all__ = ["DifferentialAbundanceConfig"]
