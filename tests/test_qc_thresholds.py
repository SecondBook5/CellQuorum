"""Tests for CellQuorum QC threshold construction utilities."""

from __future__ import annotations

# Import NumPy for deterministic numeric threshold assertions.
import numpy as np

# Import pandas for metric table construction.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import QC configuration models used by threshold construction.
from cellquorum.stages.qc.config import QCConfig

# Import QC threshold utilities under test.
from cellquorum.stages.qc.thresholds import (
    MadBounds,
    QCThreshold,
    QCThresholdError,
    QCThresholdResult,
    build_fixed_cell_thresholds,
    build_fixed_gene_thresholds,
    build_global_mad_threshold_for_metric,
    build_grouped_mad_thresholds_for_metric,
    build_mad_cell_thresholds,
    build_mad_thresholds_for_metric,
    build_qc_thresholds,
    calculate_mad_bounds,
    format_group_value,
    format_threshold_context,
    normalize_group_values,
    require_metric_columns,
    validate_metric_table,
)


def make_cell_metrics() -> pd.DataFrame:
    """
    Build a small cell-level QC metric table for threshold tests.

    The table contains fixed-threshold metrics, default MAD metrics, the
    mitochondrial MAD metric, and groupby metadata columns.

    Returns:
        Cell-level QC metric table.
    """

    # Return deterministic cell-level metric values.
    return pd.DataFrame(
        {
            "total_counts": [100.0, 200.0, 300.0, 400.0, 1000.0],
            "n_genes_by_counts": [10.0, 20.0, 30.0, 40.0, 100.0],
            "log1p_total_counts": [1.0, 2.0, 3.0, 4.0, 10.0],
            "log1p_n_genes_by_counts": [1.0, 2.0, 3.0, 4.0, 10.0],
            "pct_counts_in_top_20_genes": [10.0, 20.0, 30.0, 40.0, 90.0],
            "pct_counts_mito": [1.0, 2.0, 3.0, 4.0, 20.0],
            "pct_counts_ribo": [5.0, 6.0, 7.0, 8.0, 9.0],
            "pct_counts_hemoglobin": [0.0, 0.0, 1.0, 0.0, 5.0],
            "sample_id": ["sample_1", "sample_1", "sample_2", "sample_2", "sample_2"],
            "batch": ["batch_a", "batch_a", "batch_b", "batch_b", "batch_c"],
        },
        index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
    )


def make_gene_metrics() -> pd.DataFrame:
    """
    Build a small gene-level QC metric table for threshold tests.

    Returns:
        Gene-level QC metric table.
    """

    # Return deterministic gene-level metric values.
    return pd.DataFrame(
        {
            "n_cells_by_counts": [0.0, 1.0, 5.0],
            "total_counts": [0.0, 10.0, 100.0],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )


def make_fixed_config() -> QCConfig:
    """
    Build a fixed-threshold-only QC configuration.

    Returns:
        QCConfig with all fixed thresholds populated.
    """

    # Return a fixed-only QC configuration.
    return QCConfig(
        threshold_strategy="fixed",
        mad={"enabled": False},
        basic={
            "min_genes_per_cell": 10,
            "max_genes_per_cell": 5000,
            "min_counts_per_cell": 100,
            "max_counts_per_cell": 50000,
            "min_cells_per_gene": 3,
            "max_mito_percent": 8.0,
            "max_ribo_percent": 90.0,
            "max_hemoglobin_percent": 10.0,
        },
    )


def test_qc_threshold_serializes_and_reports_group_specific_status() -> None:
    """
    Verify individual threshold records serialize cleanly.

    Threshold records are later used in decision tables, reports, and provenance,
    so their dictionary form should be JSON-friendly and stable.
    """

    # Build a group-specific threshold record.
    threshold = QCThreshold(
        axis="cell",
        metric="pct_counts_mito",
        rule_name="mad_mito_pct_counts_mito",
        lower=1.0,
        upper=5.0,
        source="mad_mito",
        groupby_columns=("sample_id",),
        group_values=("sample_1",),
        n_observations=3,
    )

    # Confirm the threshold is group-specific.
    assert threshold.is_group_specific() is True

    # Confirm dictionary serialization is stable.
    assert threshold.to_dict() == {
        "axis": "cell",
        "metric": "pct_counts_mito",
        "rule_name": "mad_mito_pct_counts_mito",
        "lower": 1.0,
        "upper": 5.0,
        "source": "mad_mito",
        "groupby_columns": ["sample_id"],
        "group_values": ["sample_1"],
        "n_observations": 3,
    }


def test_qc_threshold_reports_global_status() -> None:
    """
    Verify global threshold records are not marked group-specific.

    Global thresholds should have no groupby columns or group values.
    """

    # Build a global threshold record.
    threshold = QCThreshold(
        axis="cell",
        metric="total_counts",
        rule_name="fixed_min_counts_per_cell",
        lower=100.0,
        upper=None,
        source="fixed",
    )

    # Confirm the threshold is not group-specific.
    assert threshold.is_group_specific() is False


def test_qc_threshold_result_filters_axes_and_serializes_to_dataframe() -> None:
    """
    Verify threshold results can filter cell/gene thresholds and build a table.

    The threshold table is the primary auditable output before decisions are
    applied.
    """

    # Build one cell threshold.
    cell_threshold = QCThreshold(
        axis="cell",
        metric="total_counts",
        rule_name="fixed_min_counts_per_cell",
        lower=100.0,
        upper=None,
        source="fixed",
    )

    # Build one gene threshold.
    gene_threshold = QCThreshold(
        axis="gene",
        metric="n_cells_by_counts",
        rule_name="fixed_min_cells_per_gene",
        lower=3.0,
        upper=None,
        source="fixed",
    )

    # Build a threshold result.
    result = QCThresholdResult(
        thresholds=[cell_threshold, gene_threshold],
        warnings=["example warning"],
    )

    # Confirm cell thresholds are filtered correctly.
    assert result.cell_thresholds() == [cell_threshold]

    # Confirm gene thresholds are filtered correctly.
    assert result.gene_thresholds() == [gene_threshold]

    # Convert thresholds to a DataFrame.
    table = result.to_dataframe()

    # Confirm one row per threshold.
    assert table.shape[0] == 2

    # Confirm table columns include core threshold fields.
    assert set(table.columns) == {
        "axis",
        "metric",
        "rule_name",
        "lower",
        "upper",
        "source",
        "groupby_columns",
        "group_values",
        "n_observations",
    }

    # Confirm summary metadata is correct.
    assert result.to_summary_dict() == {
        "n_thresholds": 2,
        "n_cell_thresholds": 1,
        "n_gene_thresholds": 1,
        "warnings": ["example warning"],
    }


def test_empty_qc_threshold_result_has_schema_aware_dataframe() -> None:
    """
    Verify empty threshold results still return a schema-aware DataFrame.

    This keeps artifact writing stable even when no thresholds are configured.
    """

    # Build an empty threshold result.
    result = QCThresholdResult(thresholds=[])

    # Convert the empty result to a DataFrame.
    table = result.to_dataframe()

    # Confirm the table is empty.
    assert table.empty is True

    # Confirm the threshold schema is still present.
    assert list(table.columns) == [
        "axis",
        "metric",
        "rule_name",
        "lower",
        "upper",
        "source",
        "groupby_columns",
        "group_values",
        "n_observations",
    ]


def test_build_fixed_cell_thresholds_creates_expected_rules() -> None:
    """
    Verify fixed cell-level thresholds are created from QCConfig.

    Fixed thresholds should be explicit records rather than hidden logic inside
    the decision layer.
    """

    # Build fixed cell-level thresholds.
    thresholds = build_fixed_cell_thresholds(make_fixed_config())

    # Build a lookup by rule name.
    by_rule = {threshold.rule_name: threshold for threshold in thresholds}

    # Confirm all expected fixed cell threshold rules exist.
    assert set(by_rule) == {
        "fixed_min_genes_per_cell",
        "fixed_max_genes_per_cell",
        "fixed_min_counts_per_cell",
        "fixed_max_counts_per_cell",
        "fixed_max_mito_percent",
        "fixed_max_ribo_percent",
        "fixed_max_hemoglobin_percent",
    }

    # Confirm a lower-bound threshold.
    assert by_rule["fixed_min_genes_per_cell"].metric == "n_genes_by_counts"
    assert by_rule["fixed_min_genes_per_cell"].lower == 10.0
    assert by_rule["fixed_min_genes_per_cell"].upper is None

    # Confirm an upper-bound threshold.
    assert by_rule["fixed_max_mito_percent"].metric == "pct_counts_mito"
    assert by_rule["fixed_max_mito_percent"].lower is None
    assert by_rule["fixed_max_mito_percent"].upper == 8.0

    # Confirm all fixed cell thresholds use the cell axis.
    assert {threshold.axis for threshold in thresholds} == {"cell"}

    # Confirm all fixed cell thresholds are marked as fixed source.
    assert {threshold.source for threshold in thresholds} == {"fixed"}


def test_build_fixed_gene_thresholds_creates_expected_rules() -> None:
    """
    Verify fixed gene-level thresholds are created from QCConfig.

    The current fixed gene threshold is minimum cells per gene.
    """

    # Build fixed gene-level thresholds.
    thresholds = build_fixed_gene_thresholds(make_fixed_config())

    # Confirm exactly one gene threshold is created.
    assert len(thresholds) == 1

    # Extract the threshold.
    threshold = thresholds[0]

    # Confirm the threshold fields.
    assert threshold.axis == "gene"
    assert threshold.metric == "n_cells_by_counts"
    assert threshold.rule_name == "fixed_min_cells_per_gene"
    assert threshold.lower == 3.0
    assert threshold.upper is None
    assert threshold.source == "fixed"


def test_build_qc_thresholds_fixed_only_combines_cell_and_gene_rules() -> None:
    """
    Verify fixed-only threshold construction combines cell and gene records.

    A fixed-only strategy should not emit MAD thresholds or MAD warnings.
    """

    # Build all thresholds using fixed-only configuration.
    result = build_qc_thresholds(
        cell_metrics=make_cell_metrics(),
        gene_metrics=make_gene_metrics(),
        config=make_fixed_config(),
    )

    # Confirm all expected fixed thresholds were created.
    assert len(result.thresholds) == 8

    # Confirm seven cell thresholds were created.
    assert len(result.cell_thresholds()) == 7

    # Confirm one gene threshold was created.
    assert len(result.gene_thresholds()) == 1

    # Confirm no warnings were emitted.
    assert result.warnings == []

    # Confirm every threshold has fixed source.
    assert {threshold.source for threshold in result.thresholds} == {"fixed"}


def test_build_qc_thresholds_mad_only_creates_default_mad_rules() -> None:
    """
    Verify MAD-only threshold construction creates default adaptive rules.

    Default MAD configuration should build thresholds for three general metrics
    and one mitochondrial-specific metric.
    """

    # Build MAD-only configuration.
    config = QCConfig(threshold_strategy="mad")

    # Build thresholds.
    result = build_qc_thresholds(
        cell_metrics=make_cell_metrics(),
        gene_metrics=make_gene_metrics(),
        config=config,
    )

    # Confirm four MAD thresholds were created.
    assert len(result.thresholds) == 4

    # Confirm all thresholds apply to cells.
    assert len(result.cell_thresholds()) == 4
    assert result.gene_thresholds() == []

    # Confirm default general MAD rule names exist.
    assert {threshold.rule_name for threshold in result.thresholds} == {
        "mad_log1p_total_counts",
        "mad_log1p_n_genes_by_counts",
        "mad_pct_counts_in_top_20_genes",
        "mad_mito_pct_counts_mito",
    }

    # Confirm both general and mitochondrial MAD sources exist.
    assert {threshold.source for threshold in result.thresholds} == {"mad", "mad_mito"}


def test_build_qc_thresholds_fixed_and_mad_combines_all_thresholds() -> None:
    """
    Verify fixed-and-MAD strategy combines fixed and adaptive thresholds.

    The default project behavior should produce auditable fixed thresholds and
    adaptive MAD thresholds together.
    """

    # Build a fixed-and-MAD configuration with default basic thresholds.
    config = QCConfig(threshold_strategy="fixed_and_mad")

    # Build thresholds.
    result = build_qc_thresholds(
        cell_metrics=make_cell_metrics(),
        gene_metrics=make_gene_metrics(),
        config=config,
    )

    # Confirm default fixed thresholds plus four MAD thresholds are present.
    assert len(result.thresholds) == 7

    # Confirm default fixed thresholds include two cell rules and one gene rule.
    assert len([threshold for threshold in result.thresholds if threshold.source == "fixed"]) == 3

    # Confirm four MAD thresholds are present.
    assert len([threshold for threshold in result.thresholds if threshold.source == "mad"]) == 3
    assert (
        len([threshold for threshold in result.thresholds if threshold.source == "mad_mito"]) == 1
    )


def test_build_qc_thresholds_rejects_invalid_config_type() -> None:
    """
    Verify threshold construction rejects non-QCConfig config objects.

    This catches accidental dictionary usage before threshold logic accesses
    nested QC configuration fields.
    """

    # Confirm invalid config objects fail clearly.
    with pytest.raises(QCThresholdError, match="QCConfig object"):
        build_qc_thresholds(
            cell_metrics=make_cell_metrics(),
            gene_metrics=make_gene_metrics(),
            config={"threshold_strategy": "fixed"},  # type: ignore[arg-type]
        )


def test_build_qc_thresholds_rejects_invalid_metric_tables() -> None:
    """
    Verify threshold construction validates metric table inputs.

    Both cell and gene metrics must be non-empty pandas DataFrames.
    """

    # Confirm invalid cell metrics fail clearly.
    with pytest.raises(QCThresholdError, match="cell_metrics must be a pandas DataFrame"):
        build_qc_thresholds(
            cell_metrics={"not": "dataframe"},  # type: ignore[arg-type]
            gene_metrics=make_gene_metrics(),
            config=make_fixed_config(),
        )

    # Confirm invalid gene metrics fail clearly.
    with pytest.raises(QCThresholdError, match="gene_metrics must be a pandas DataFrame"):
        build_qc_thresholds(
            cell_metrics=make_cell_metrics(),
            gene_metrics={"not": "dataframe"},  # type: ignore[arg-type]
            config=make_fixed_config(),
        )


def test_build_mad_cell_thresholds_creates_global_thresholds() -> None:
    """
    Verify MAD cell-threshold construction creates global threshold records.

    With no groupby columns, one threshold should be emitted for each configured
    MAD metric plus the mitochondrial-specific MAD metric.
    """

    # Build cell metrics.
    cell_metrics = make_cell_metrics()

    # Build a focused MAD config.
    config = QCConfig(
        threshold_strategy="mad",
        mad={
            "n_mads": 2.0,
            "metrics": ["log1p_total_counts"],
            "mito_metric": "pct_counts_mito",
            "mito_n_mads": 3.0,
        },
    ).mad

    # Build MAD thresholds.
    result = build_mad_cell_thresholds(cell_metrics, config)

    # Confirm one general and one mitochondrial threshold were emitted.
    assert len(result.thresholds) == 2

    # Build a lookup by rule name.
    by_rule = {threshold.rule_name: threshold for threshold in result.thresholds}

    # Confirm general MAD bounds from values [1, 2, 3, 4, 10].
    assert by_rule["mad_log1p_total_counts"].lower == 1.0
    assert by_rule["mad_log1p_total_counts"].upper == 5.0

    # Confirm mitochondrial MAD bounds from values [1, 2, 3, 4, 20] with 3 MADs.
    assert by_rule["mad_mito_pct_counts_mito"].lower == 0.0
    assert by_rule["mad_mito_pct_counts_mito"].upper == 6.0


def test_build_mad_cell_thresholds_rejects_missing_metric_columns() -> None:
    """
    Verify MAD threshold construction rejects missing metric columns.

    MAD thresholds cannot be estimated when configured metric columns are absent.
    """

    # Build cell metrics without the requested metric.
    cell_metrics = make_cell_metrics().drop(columns=["log1p_total_counts"])

    # Build a MAD config requiring the dropped metric.
    config = QCConfig(threshold_strategy="mad").mad

    # Confirm missing metric columns fail clearly.
    with pytest.raises(QCThresholdError, match="missing required column"):
        build_mad_cell_thresholds(cell_metrics, config)


def test_build_mad_cell_thresholds_rejects_missing_groupby_columns() -> None:
    """
    Verify MAD threshold construction rejects missing groupby columns.

    Group-wise thresholding requires requested metadata columns in the metric
    table.
    """

    # Build cell metrics.
    cell_metrics = make_cell_metrics()

    # Build a MAD config requesting a missing groupby column.
    config = QCConfig(
        threshold_strategy="mad",
        mad={"groupby": ["donor_id"]},
    ).mad

    # Confirm missing groupby columns fail clearly.
    with pytest.raises(QCThresholdError, match="missing required column"):
        build_mad_cell_thresholds(cell_metrics, config)


def test_build_global_mad_threshold_for_metric_calculates_expected_bounds() -> None:
    """
    Verify global MAD threshold calculation for one metric.

    Values [1, 2, 3, 4, 10] have median 3 and MAD 1, so two MADs produce
    lower=1 and upper=5.
    """

    # Build cell metrics.
    cell_metrics = make_cell_metrics()

    # Build the global MAD threshold.
    result = build_global_mad_threshold_for_metric(
        cell_metrics,
        metric="log1p_total_counts",
        n_mads=2.0,
        source="mad",
        rule_name="mad_log1p_total_counts",
        skip_zero_mad=True,
    )

    # Confirm one threshold was created.
    assert len(result.thresholds) == 1

    # Extract the threshold.
    threshold = result.thresholds[0]

    # Confirm threshold fields.
    assert threshold.axis == "cell"
    assert threshold.metric == "log1p_total_counts"
    assert threshold.rule_name == "mad_log1p_total_counts"
    assert threshold.lower == 1.0
    assert threshold.upper == 5.0
    assert threshold.source == "mad"
    assert threshold.n_observations == 5

    # Confirm no warnings were emitted.
    assert result.warnings == []


def test_build_mad_thresholds_for_metric_dispatches_global_threshold() -> None:
    """
    Verify the MAD metric dispatcher builds global thresholds without groupby.

    This function routes to global or grouped threshold construction depending
    on whether groupby columns were supplied.
    """

    # Build thresholds through the dispatcher without groupby columns.
    result = build_mad_thresholds_for_metric(
        make_cell_metrics(),
        metric="log1p_total_counts",
        n_mads=2.0,
        source="mad",
        rule_name="mad_log1p_total_counts",
        groupby=[],
        skip_zero_mad=True,
    )

    # Confirm a global threshold was created.
    assert len(result.thresholds) == 1
    assert result.thresholds[0].is_group_specific() is False


def test_build_grouped_mad_thresholds_for_metric_calculates_group_specific_bounds() -> None:
    """
    Verify grouped MAD threshold calculation creates per-group records.

    The grouped threshold records should include groupby metadata and group
    values so later decision logic can apply the correct threshold per row.
    """

    # Build cell metrics with non-zero MAD in each sample group.
    cell_metrics = pd.DataFrame(
        {
            "metric": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
            "sample_id": ["a", "a", "a", "b", "b", "b"],
        },
        index=[f"cell_{index}" for index in range(6)],
    )

    # Build grouped MAD thresholds.
    result = build_grouped_mad_thresholds_for_metric(
        cell_metrics,
        metric="metric",
        n_mads=1.0,
        source="mad",
        rule_name="mad_metric",
        groupby=["sample_id"],
        skip_zero_mad=True,
    )

    # Confirm one threshold per group.
    assert len(result.thresholds) == 2

    # Build a lookup by group value.
    by_group = {threshold.group_values[0]: threshold for threshold in result.thresholds}

    # Confirm group a bounds.
    assert by_group["a"].lower == 1.0
    assert by_group["a"].upper == 3.0

    # Confirm group b bounds.
    assert by_group["b"].lower == 10.0
    assert by_group["b"].upper == 14.0

    # Confirm thresholds are group-specific.
    assert all(threshold.is_group_specific() for threshold in result.thresholds)

    # Confirm groupby columns are recorded.
    assert {threshold.groupby_columns for threshold in result.thresholds} == {("sample_id",)}


def test_build_mad_thresholds_for_metric_dispatches_grouped_thresholds() -> None:
    """
    Verify the MAD metric dispatcher builds grouped thresholds with groupby.

    Supplying groupby columns should produce group-specific threshold records.
    """

    # Build cell metrics with grouped values.
    cell_metrics = pd.DataFrame(
        {
            "metric": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
            "sample_id": ["a", "a", "a", "b", "b", "b"],
        }
    )

    # Build thresholds through the dispatcher with groupby columns.
    result = build_mad_thresholds_for_metric(
        cell_metrics,
        metric="metric",
        n_mads=1.0,
        source="mad",
        rule_name="mad_metric",
        groupby=["sample_id"],
        skip_zero_mad=True,
    )

    # Confirm grouped thresholds were created.
    assert len(result.thresholds) == 2

    # Confirm thresholds are group-specific.
    assert all(threshold.is_group_specific() for threshold in result.thresholds)


def test_build_grouped_mad_thresholds_for_metric_skips_zero_mad_groups() -> None:
    """
    Verify grouped MAD thresholding skips zero-MAD groups when configured.

    This prevents impossible or overly strict thresholds on groups with no
    variability.
    """

    # Build grouped metrics with one zero-MAD group.
    cell_metrics = pd.DataFrame(
        {
            "metric": [1.0, 1.0, 1.0, 10.0, 12.0, 14.0],
            "sample_id": ["a", "a", "a", "b", "b", "b"],
        }
    )

    # Build grouped MAD thresholds.
    result = build_grouped_mad_thresholds_for_metric(
        cell_metrics,
        metric="metric",
        n_mads=1.0,
        source="mad",
        rule_name="mad_metric",
        groupby=["sample_id"],
        skip_zero_mad=True,
    )

    # Confirm only the non-zero-MAD group emitted a threshold.
    assert len(result.thresholds) == 1
    assert result.thresholds[0].group_values == ("b",)

    # Confirm a zero-MAD warning was emitted.
    assert result.warnings == ["Skipped MAD threshold for metric (sample_id=a): MAD is zero."]


def test_calculate_mad_bounds_returns_expected_bounds() -> None:
    """
    Verify MAD bound calculation uses median +/- n_mads * MAD.

    Values [1, 2, 3, 4, 10] have median 3 and MAD 1.
    """

    # Calculate MAD bounds.
    bounds = calculate_mad_bounds(
        pd.Series([1.0, 2.0, 3.0, 4.0, 10.0]),
        n_mads=2.0,
        metric="metric",
        skip_zero_mad=True,
    )

    # Confirm a MadBounds record was returned.
    assert isinstance(bounds, MadBounds)

    # Confirm lower and upper bounds.
    assert bounds.lower == 1.0
    assert bounds.upper == 5.0

    # Confirm finite observation count.
    assert bounds.n_observations == 5

    # Confirm no warnings were emitted.
    assert bounds.warnings == []


def test_calculate_mad_bounds_ignores_non_finite_values() -> None:
    """
    Verify MAD bound calculation ignores NaN and infinite values.

    Threshold estimation should be based only on finite metric values.
    """

    # Calculate MAD bounds with non-finite values present.
    bounds = calculate_mad_bounds(
        pd.Series([1.0, 2.0, np.nan, np.inf, 3.0]),
        n_mads=2.0,
        metric="metric",
        skip_zero_mad=True,
    )

    # Confirm finite values [1, 2, 3] were used.
    assert bounds.n_observations == 3
    assert bounds.lower == 0.0
    assert bounds.upper == 4.0


def test_calculate_mad_bounds_skips_no_finite_values() -> None:
    """
    Verify MAD bound calculation skips metrics with no finite values.

    Metrics with no finite data cannot produce meaningful thresholds.
    """

    # Calculate MAD bounds for all-non-finite values.
    bounds = calculate_mad_bounds(
        pd.Series([np.nan, np.inf, -np.inf]),
        n_mads=2.0,
        metric="metric",
        skip_zero_mad=True,
    )

    # Confirm no bounds were produced.
    assert bounds.lower is None
    assert bounds.upper is None

    # Confirm zero observations were used.
    assert bounds.n_observations == 0

    # Confirm a warning was emitted.
    assert bounds.warnings == ["Skipped MAD threshold for metric: no finite values."]


def test_calculate_mad_bounds_skips_zero_mad_when_configured() -> None:
    """
    Verify zero-MAD metrics are skipped when skip_zero_mad is true.

    This is the default behavior because zero-MAD thresholds can be too strict.
    """

    # Calculate MAD bounds for constant values.
    bounds = calculate_mad_bounds(
        pd.Series([5.0, 5.0, 5.0]),
        n_mads=2.0,
        metric="metric",
        skip_zero_mad=True,
    )

    # Confirm no bounds were produced.
    assert bounds.lower is None
    assert bounds.upper is None

    # Confirm a warning was emitted.
    assert bounds.warnings == ["Skipped MAD threshold for metric: MAD is zero."]


def test_calculate_mad_bounds_uses_median_bounds_when_zero_mad_not_skipped() -> None:
    """
    Verify zero-MAD metrics can produce exact median bounds when requested.

    This behavior is available for users who want deterministic exact-match
    thresholds rather than skipping constant metrics.
    """

    # Calculate MAD bounds for constant values without skipping zero MAD.
    bounds = calculate_mad_bounds(
        pd.Series([5.0, 5.0, 5.0]),
        n_mads=2.0,
        metric="metric",
        skip_zero_mad=False,
    )

    # Confirm exact median bounds were produced.
    assert bounds.lower == 5.0
    assert bounds.upper == 5.0

    # Confirm a warning was emitted.
    assert bounds.warnings == ["MAD threshold for metric has MAD zero; using median bounds."]


def test_build_global_mad_threshold_for_metric_skips_zero_mad_threshold() -> None:
    """
    Verify global MAD threshold construction returns warnings when skipped.

    The threshold result should contain no thresholds and preserve warning
    context.
    """

    # Build a constant metric table.
    cell_metrics = pd.DataFrame({"metric": [1.0, 1.0, 1.0]})

    # Build a global MAD threshold.
    result = build_global_mad_threshold_for_metric(
        cell_metrics,
        metric="metric",
        n_mads=2.0,
        source="mad",
        rule_name="mad_metric",
        skip_zero_mad=True,
    )

    # Confirm no thresholds were emitted.
    assert result.thresholds == []

    # Confirm warning context was preserved.
    assert result.warnings == ["Skipped MAD threshold for metric: MAD is zero."]


def test_validate_metric_table_accepts_non_empty_dataframe() -> None:
    """
    Verify metric table validation accepts non-empty DataFrames.

    Valid metric tables should pass silently.
    """

    # Confirm non-empty DataFrame validation does not raise.
    validate_metric_table(pd.DataFrame({"metric": [1.0]}), table_name="cell_metrics")


def test_validate_metric_table_rejects_non_dataframe() -> None:
    """
    Verify metric table validation rejects non-DataFrame inputs.

    Threshold construction should fail before trying to access DataFrame methods.
    """

    # Confirm non-DataFrame inputs fail clearly.
    with pytest.raises(QCThresholdError, match="must be a pandas DataFrame"):
        validate_metric_table({"metric": [1.0]}, table_name="cell_metrics")  # type: ignore[arg-type]


def test_validate_metric_table_rejects_empty_dataframe() -> None:
    """
    Verify metric table validation rejects empty DataFrames.

    Threshold construction needs actual metric values.
    """

    # Confirm empty DataFrames fail clearly.
    with pytest.raises(QCThresholdError, match="at least one row and one column"):
        validate_metric_table(pd.DataFrame(), table_name="cell_metrics")


def test_require_metric_columns_accepts_existing_columns() -> None:
    """
    Verify required metric-column validation accepts existing columns.

    This helper guards threshold construction from missing metric fields.
    """

    # Build a metric table.
    table = pd.DataFrame({"a": [1.0], "b": [2.0]})

    # Confirm existing columns pass.
    require_metric_columns(table, ["a", "b"], table_name="cell_metrics")


def test_require_metric_columns_rejects_single_string_argument() -> None:
    """
    Verify required metric-column validation rejects single strings.

    Strings are iterable, so callers must provide an explicit sequence of column
    names.
    """

    # Build a metric table.
    table = pd.DataFrame({"a": [1.0]})

    # Confirm single-string columns fail clearly.
    with pytest.raises(QCThresholdError, match="not a string"):
        require_metric_columns(table, "a", table_name="cell_metrics")  # type: ignore[arg-type]


def test_require_metric_columns_rejects_missing_columns() -> None:
    """
    Verify required metric-column validation rejects missing columns.

    Missing columns should be reported together for easier configuration
    debugging.
    """

    # Build a metric table.
    table = pd.DataFrame({"a": [1.0]})

    # Confirm missing columns fail clearly.
    with pytest.raises(QCThresholdError, match="missing required column"):
        require_metric_columns(table, ["a", "b", "c"], table_name="cell_metrics")


def test_normalize_group_values_handles_scalars_and_tuples() -> None:
    """
    Verify groupby key normalization handles scalar and tuple keys.

    Pandas may return scalar keys for one groupby column and tuple keys for
    multiple columns.
    """

    # Confirm scalar group values become one-element tuples.
    assert normalize_group_values("sample_1") == ("sample_1",)

    # Confirm tuple group values are preserved as string tuples.
    assert normalize_group_values(("sample_1", 3)) == ("sample_1", "3")


def test_format_group_value_handles_missing_values() -> None:
    """
    Verify group values get stable string formatting.

    Missing values should be represented consistently in threshold records and
    warnings.
    """

    # Confirm normal values are stringified.
    assert format_group_value(3) == "3"

    # Confirm missing values get a stable placeholder.
    assert format_group_value(np.nan) == "<NA>"


def test_format_threshold_context_handles_global_and_grouped_contexts() -> None:
    """
    Verify warning contexts are readable for global and grouped thresholds.

    Clear context strings make skipped-threshold warnings actionable.
    """

    # Confirm global context contains only the metric name.
    assert format_threshold_context("metric", (), ()) == "metric"

    # Confirm grouped context includes groupby key-value pairs.
    assert (
        format_threshold_context(
            "metric",
            ("sample_id", "batch"),
            ("sample_1", "batch_a"),
        )
        == "metric (sample_id=sample_1, batch=batch_a)"
    )
