"""Configuration for the pseudobulk differential-expression stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AmbientSpecificityFilterConfig(StrictBaseModel):
    """Optional cross-compartment ambient-gene filter, applied before pseudobulk
    aggregation (see ``cellquorum.stats.ambient_specificity``).

    Off by default: SoupX (``ambient_correction``) already runs per library before
    any cell-type label exists, so this is a second, later-stage correction for the
    contamination it structurally cannot see (a gene that peaks in one cell type and
    leaks into another at low, non-specific levels). It requires a reference profile
    built from a WIDER cohort than the one being tested — a single-lineage run has no
    other lineage's expression to compare against — so enabling it without a
    ``reference_path``/``target_group`` is a configuration error, not a silent no-op.

    Attributes:
        enabled: Whether the filter runs. Off by default.
        reference_path: Path to a genes x groups mean-expression CSV/parquet, built
            from a cohort wider than the one this DE run analyzes.
        target_group: The group (as named in ``reference_path``'s columns) this DE
            run's cells belong to.
        spec_threshold: A foreign-peaking gene is ambient only when its value here is
            strictly below this fraction of its peak elsewhere.
        group_map: Optional raw-group -> coarse-group mapping so a gene peaking in a
            biologically related group is never treated as foreign. No default
            grouping is provided; which groups are "related" is a domain judgment.
    """

    enabled: bool = False
    reference_path: str | None = None
    group_col: str = "cell_type"
    target_group: str | None = None
    spec_threshold: float = 0.25
    group_map: dict[str, str] = {}


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
        cell_type_col: Optional obs column to stratify the comparison by (e.g.
            a subcluster label). None (default) runs one comparison pooling
            every cell, exactly as before. Set, the stage ALSO runs the
            comparison separately within each group and writes a combined
            long-format table alongside the pooled one -- a pooled-only
            comparison cannot distinguish "this gene changes within a cell
            state" from "the MIX of states shifted between conditions",
            which every other comparative stage in this pipeline
            (composition, differential_abundance, coexpression, grn,
            enrichment) already accounts for by running per group. A group
            without enough donor replication for its own fit is skipped with
            a warning, not fatal to the others or to the pooled result.
    """

    # Whether this stage runs.
    enabled: bool = True

    # Selected DE method (registry key under stage_category 'differential_expression').
    method: str = "pseudobulk_edger"

    # Optional obs column to additionally stratify the comparison by.
    cell_type_col: str | None = None

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

    # Optional cross-compartment ambient-gene filter (off by default).
    ambient_filter: AmbientSpecificityFilterConfig = AmbientSpecificityFilterConfig()


__all__ = ["AmbientSpecificityFilterConfig", "DifferentialExpressionConfig"]
