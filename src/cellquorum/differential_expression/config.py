"""Configuration for the pseudobulk differential-expression stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DifferentialExpressionConfig(StrictBaseModel):
    """Config for the pseudobulk differential-expression stage."""

    # Whether this stage runs.
    enabled: bool = True

    # Selected DE method (registry key under stage_category 'differential_expression').
    method: str = "pseudobulk_edger"

    # Layer holding raw counts for pseudobulk aggregation.
    layer: str = "counts"

    # Optional covariates added to the design matrix (schema-driven; validated
    # against obs only when non-empty). Empty for datasets without clinical metadata.
    covariates: list[str] = []

    # edgeR filterByExpr thresholds.
    min_count: int = 10
    min_total_count: int = 15

    # FDR threshold recorded in outputs.
    fdr: float = 0.05

    # R execution timeout (seconds).
    timeout_seconds: int = 1800

    # R package required for the fit.
    r_package: str = "edgeR"
