"""Configuration for the pseudobulk differential-expression stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DifferentialExpressionConfig(StrictBaseModel):
    """Pseudobulk differential-expression analysis via edgeR.

    Aggregates single-cell counts to pseudobulk samples per group,
    then performs robust statistical testing via edgeR (R). Produces
    gene-level statistics and diagnostic plots.

    Attributes:
        enabled: Whether the stage runs (enabled by default).
        method: DE method registry key (pseudobulk_edger).
        layer: Layer holding raw counts for pseudobulk aggregation.
        covariates: Optional covariates added to the design matrix.
        interactions: Optional two-way interaction terms. Each is a ``[a, b]``
            pair of factor columns (each the condition column or a declared
            covariate). When set, the fit tests the interaction (a
            difference-of-differences F-test) instead of the case-vs-control
            main effect.
        min_count: edgeR filterByExpr minimum count threshold.
        min_total_count: edgeR filterByExpr minimum total count threshold.
        fdr: FDR threshold recorded in outputs.
        timeout_seconds: R execution timeout in seconds.
        r_package: R package name for backend status checks (edgeR).
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected DE method (registry key under stage_category 'differential_expression').
    method: str = "pseudobulk_edger"

    # Layer holding raw counts for pseudobulk aggregation.
    layer: str = "counts"

    # Optional covariates added to the design matrix (schema-driven; validated
    # against obs only when non-empty). Empty for datasets without clinical metadata.
    covariates: list[str] = []

    # Optional two-way interaction terms as [factor_a, factor_b] pairs. Each member
    # must be the condition column or a declared covariate. When non-empty the fit
    # tests the interaction (difference-of-differences) rather than the main effect.
    interactions: list[list[str]] = []

    # edgeR filterByExpr thresholds.
    min_count: int = 10
    min_total_count: int = 15

    # FDR threshold recorded in outputs.
    fdr: float = 0.05

    # R execution timeout (seconds).
    timeout_seconds: int = 1800

    # R package required for the fit.
    r_package: str = "edgeR"


__all__ = ["DifferentialExpressionConfig"]
