"""Configuration for the differential-abundance stage."""

from __future__ import annotations

from typing import Literal

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
        reference_celltype: Reference cell type for compositional analysis (None to
            let the engine choose it).
        select_reference: Whether the engine picks the compositional reference on
            centred-log-ratio variance. Disabling it hands the choice back to
            scCODA's own criterion, which ranks cell types largely by rarity.
        min_reference_abundance: Minimum mean relative abundance for a cell type to
            serve as the compositional reference.
        pair_by_donor: Whether donor enters the compositional model formula --
            'auto' (paired when enough donors span both arms), 'always', or 'never'.
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

    # Reference cell type for compositional analysis (None -> engine selection).
    reference_celltype: str | None = None

    # Let the engine choose the compositional reference on centred-log-ratio
    # variance. scCODA's own "automatic" criterion minimises var(p)/mean(p), which
    # is cv**2 * mean and so scales with abundance: it ranks cell types largely by
    # rarity, and the reference is the denominator of every reported effect.
    select_reference: bool = True

    # Minimum mean relative abundance for a compositional reference. Below a few
    # percent the reference's own counting noise propagates into every effect.
    min_reference_abundance: float = 0.05

    # Whether donor enters the compositional model formula. 'auto' pairs when
    # enough donors span both arms, which is a property of the cohort rather than a
    # preference; 'never' reproduces an unpaired fit; 'always' forces pairing where
    # the design permits it at all.
    pair_by_donor: Literal["auto", "always", "never"] = "auto"

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
