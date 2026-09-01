"""Tests for CellQuorum QC decision-table construction utilities."""

from __future__ import annotations

# Import NumPy for non-finite metric tests.
import numpy as np

# Import pandas for metric and decision table construction.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import QC configuration used by decision construction.
from cellquorum.stages.qc.config import QCConfig

# Import QC decision utilities under test.
from cellquorum.stages.qc.decisions import (
    QCDecisionError,
    QCDecisionResult,
    apply_threshold_to_decision_table,
    apply_thresholds_to_metric_table,
    build_decision_summary,
    build_failed_rule_strings,
    build_qc_decisions,
    build_qc_report_table,
    build_threshold_row_selector,
    count_failures_by_rule,
    evaluate_threshold_failures,
    finalize_decision_table,
    initialize_decision_table,
    normalize_threshold_records,
    require_decision_metric_columns,
    unique_rule_names,
    validate_decision_metric_table,
    validate_non_empty_filtered_result,
)

# Import threshold records used by decision construction.
from cellquorum.stages.qc.thresholds import QCThreshold, QCThresholdResult


def make_cell_metrics() -> pd.DataFrame:
    """
    Build a small cell-level QC metric table for decision tests.

    Returns:
        Cell-level metric table with fixed-threshold and grouped-threshold fields.
    """

    # Return deterministic cell-level metrics.
    return pd.DataFrame(
        {
            "total_counts": [50.0, 100.0, 500.0, 1000.0],
            "n_genes_by_counts": [5.0, 10.0, 50.0, 100.0],
            "pct_counts_mito": [2.0, 8.0, 12.0, 30.0],
            "log1p_total_counts": [1.0, 2.0, 3.0, 10.0],
            "sample_id": ["sample_1", "sample_1", "sample_2", "sample_2"],
            "batch": ["batch_a", "batch_a", "batch_b", "batch_b"],
        },
        index=["cell_1", "cell_2", "cell_3", "cell_4"],
    )


def make_gene_metrics() -> pd.DataFrame:
    """
    Build a small gene-level QC metric table for decision tests.

    Returns:
        Gene-level metric table.
    """

    # Return deterministic gene-level metrics.
    return pd.DataFrame(
        {
            "n_cells_by_counts": [0.0, 1.0, 5.0],
            "total_counts": [0.0, 10.0, 100.0],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )


def make_cell_thresholds() -> list[QCThreshold]:
    """
    Build representative cell-level threshold records.

    Returns:
        List of cell-level threshold records.
    """

    # Return deterministic cell threshold rules.
    return [
        QCThreshold(
            axis="cell",
            metric="n_genes_by_counts",
            rule_name="fixed_min_genes_per_cell",
            lower=10.0,
            upper=None,
            source="fixed",
        ),
        QCThreshold(
            axis="cell",
            metric="pct_counts_mito",
            rule_name="fixed_max_mito_percent",
            lower=None,
            upper=10.0,
            source="fixed",
        ),
        QCThreshold(
            axis="cell",
            metric="log1p_total_counts",
            rule_name="mad_log1p_total_counts",
            lower=1.5,
            upper=5.0,
            source="mad",
        ),
    ]


def make_gene_thresholds() -> list[QCThreshold]:
    """
    Build representative gene-level threshold records.

    Returns:
        List of gene-level threshold records.
    """

    # Return deterministic gene threshold rules.
    return [
        QCThreshold(
            axis="gene",
            metric="n_cells_by_counts",
            rule_name="fixed_min_cells_per_gene",
            lower=1.0,
            upper=None,
            source="fixed",
        )
    ]


def test_normalize_threshold_records_accepts_threshold_result() -> None:
    """
    Verify threshold records can be extracted from QCThresholdResult.

    Decision construction should accept the output of threshold construction
    directly.
    """

    # Build threshold records.
    thresholds = make_cell_thresholds()

    # Wrap records in a threshold result.
    result = QCThresholdResult(thresholds=thresholds)

    # Normalize the threshold result.
    normalized = normalize_threshold_records(result)

    # Confirm the original records were returned.
    assert normalized == thresholds


def test_normalize_threshold_records_accepts_threshold_sequence() -> None:
    """
    Verify threshold records can be supplied as a plain sequence.

    This keeps lower-level tests and plugin integrations simple.
    """

    # Build threshold records.
    thresholds = make_cell_thresholds()

    # Normalize the threshold sequence.
    normalized = normalize_threshold_records(thresholds)

    # Confirm the original records were returned.
    assert normalized == thresholds


def test_normalize_threshold_records_rejects_invalid_inputs() -> None:
    """
    Verify invalid threshold containers fail clearly.

    Threshold inputs should be explicit QCThreshold records, not strings,
    dictionaries, or arbitrary objects.
    """

    # Confirm a string input is rejected.
    with pytest.raises(QCDecisionError, match="not a string"):
        normalize_threshold_records("bad thresholds")

    # Confirm a non-sequence input is rejected.
    with pytest.raises(QCDecisionError, match="QCThresholdResult or a sequence"):
        normalize_threshold_records(123)  # type: ignore[arg-type]

    # Confirm a sequence containing non-threshold entries is rejected.
    with pytest.raises(QCDecisionError, match="only QCThreshold records"):
        normalize_threshold_records([{"not": "threshold"}])  # type: ignore[list-item]


def test_unique_rule_names_preserves_first_seen_order() -> None:
    """
    Verify unique rule-name extraction preserves threshold order.

    Grouped thresholds may repeat a rule name. The decision table should contain
    one rule column in first-seen order.
    """

    # Build threshold records with a duplicated rule name.
    thresholds = [
        QCThreshold(
            axis="cell",
            metric="metric_a",
            rule_name="rule_a",
            lower=0.0,
            upper=None,
            source="fixed",
        ),
        QCThreshold(
            axis="cell",
            metric="metric_b",
            rule_name="rule_b",
            lower=0.0,
            upper=None,
            source="fixed",
        ),
        QCThreshold(
            axis="cell",
            metric="metric_a",
            rule_name="rule_a",
            lower=1.0,
            upper=None,
            source="fixed",
        ),
    ]

    # Confirm unique rule names preserve first-seen order.
    assert unique_rule_names(thresholds) == ["rule_a", "rule_b"]


def test_initialize_decision_table_adds_rule_columns() -> None:
    """
    Verify decision-table initialization adds one column per unique rule.

    Initial rule columns should all be False before thresholds are applied.
    """

    # Build metrics and thresholds.
    metrics = make_cell_metrics()
    thresholds = make_cell_thresholds()

    # Initialize the decision table.
    decisions = initialize_decision_table(metrics.index, thresholds)

    # Confirm rule columns were added.
    assert list(decisions.columns) == [
        "fixed_min_genes_per_cell",
        "fixed_max_mito_percent",
        "mad_log1p_total_counts",
    ]

    # Confirm all initial decisions are False.
    assert decisions.to_numpy(dtype=bool).sum() == 0


def test_build_threshold_row_selector_global_selects_all_rows() -> None:
    """
    Verify global thresholds apply to all rows.

    Thresholds without group metadata should not restrict rows.
    """

    # Build cell metrics.
    metrics = make_cell_metrics()

    # Build a global threshold.
    threshold = make_cell_thresholds()[0]

    # Build the row selector.
    selector = build_threshold_row_selector(metrics, threshold)

    # Confirm every row is selected.
    assert selector.tolist() == [True, True, True, True]


def test_build_threshold_row_selector_group_specific_selects_matching_rows() -> None:
    """
    Verify group-specific thresholds apply only to matching rows.

    Grouped MAD thresholds should be applied only to the rows used to estimate
    the corresponding group-specific bound.
    """

    # Build cell metrics.
    metrics = make_cell_metrics()

    # Build a group-specific threshold.
    threshold = QCThreshold(
        axis="cell",
        metric="log1p_total_counts",
        rule_name="mad_log1p_total_counts",
        lower=1.5,
        upper=5.0,
        source="mad",
        groupby_columns=("sample_id",),
        group_values=("sample_2",),
    )

    # Build the row selector.
    selector = build_threshold_row_selector(metrics, threshold)

    # Confirm only sample_2 rows are selected.
    assert selector.tolist() == [False, False, True, True]


def test_build_threshold_row_selector_group_specific_selects_missing_values() -> None:
    """
    Verify group-specific thresholds can select missing group values.

    Threshold construction stores missing group values as <NA>, so decision logic
    should match rows where the groupby column is missing.
    """

    # Build cell metrics with a missing group value.
    metrics = make_cell_metrics()
    metrics.loc["cell_2", "sample_id"] = np.nan

    # Build a group-specific threshold for missing sample_id.
    threshold = QCThreshold(
        axis="cell",
        metric="log1p_total_counts",
        rule_name="mad_log1p_total_counts",
        lower=1.5,
        upper=5.0,
        source="mad",
        groupby_columns=("sample_id",),
        group_values=("<NA>",),
    )

    # Build the row selector.
    selector = build_threshold_row_selector(metrics, threshold)

    # Confirm only the missing group row is selected.
    assert selector.tolist() == [False, True, False, False]


def test_evaluate_threshold_failures_applies_lower_and_upper_bounds() -> None:
    """
    Verify threshold failure evaluation handles lower and upper bounds.

    Values below the lower bound or above the upper bound should fail.
    """

    # Build cell metrics.
    metrics = make_cell_metrics()

    # Build a bounded threshold.
    threshold = QCThreshold(
        axis="cell",
        metric="log1p_total_counts",
        rule_name="mad_log1p_total_counts",
        lower=1.5,
        upper=5.0,
        source="mad",
    )

    # Select all rows.
    selected_rows = pd.Series(True, index=metrics.index, dtype=bool)

    # Evaluate failures.
    failures, warnings = evaluate_threshold_failures(
        metrics=metrics,
        threshold=threshold,
        selected_rows=selected_rows,
    )

    # Confirm values outside [1.5, 5.0] failed.
    assert failures.tolist() == [True, False, False, True]

    # Confirm no warnings were emitted.
    assert warnings == []


def test_evaluate_threshold_failures_marks_non_finite_values_as_failures() -> None:
    """
    Verify non-finite metric values are marked as failures.

    QC should not silently pass NaN or infinite metric values.
    """

    # Build cell metrics with a non-finite value.
    metrics = make_cell_metrics()
    metrics.loc["cell_2", "pct_counts_mito"] = np.nan

    # Build an upper-bound threshold.
    threshold = QCThreshold(
        axis="cell",
        metric="pct_counts_mito",
        rule_name="fixed_max_mito_percent",
        lower=None,
        upper=10.0,
        source="fixed",
    )

    # Select all rows.
    selected_rows = pd.Series(True, index=metrics.index, dtype=bool)

    # Evaluate failures.
    failures, warnings = evaluate_threshold_failures(
        metrics=metrics,
        threshold=threshold,
        selected_rows=selected_rows,
    )

    # Confirm the NaN row and above-threshold rows fail.
    assert failures.tolist() == [False, True, True, True]

    # Confirm a warning was emitted.
    assert warnings == [
        "Threshold 'fixed_max_mito_percent' for metric 'pct_counts_mito' "
        "encountered 1 non-finite value(s), marked as failures."
    ]


def test_evaluate_threshold_failures_warns_when_group_matches_zero_rows() -> None:
    """
    Verify zero-row threshold matches emit a warning.

    This can happen when applying stale grouped thresholds to a different metric
    table.
    """

    # Build cell metrics.
    metrics = make_cell_metrics()

    # Build a threshold.
    threshold = make_cell_thresholds()[0]

    # Select no rows.
    selected_rows = pd.Series(False, index=metrics.index, dtype=bool)

    # Evaluate failures.
    failures, warnings = evaluate_threshold_failures(
        metrics=metrics,
        threshold=threshold,
        selected_rows=selected_rows,
    )

    # Confirm no rows failed.
    assert failures.tolist() == [False, False, False, False]

    # Confirm warning was emitted.
    assert warnings == [
        "Threshold 'fixed_min_genes_per_cell' for metric 'n_genes_by_counts' " "matched zero rows."
    ]


def test_apply_threshold_to_decision_table_updates_rule_column() -> None:
    """
    Verify one threshold updates its decision-table rule column.

    The function should OR failures into the rule column so repeated grouped
    thresholds with the same rule name can share one output column.
    """

    # Build metrics and threshold.
    metrics = make_cell_metrics()
    threshold = make_cell_thresholds()[1]

    # Initialize the decision table.
    decisions = initialize_decision_table(metrics.index, [threshold])

    # Apply the threshold.
    warnings = apply_threshold_to_decision_table(
        metrics=metrics,
        decisions=decisions,
        threshold=threshold,
    )

    # Confirm above-threshold mitochondrial rows failed.
    assert decisions["fixed_max_mito_percent"].tolist() == [False, False, True, True]

    # Confirm no warnings were emitted.
    assert warnings == []


def test_apply_threshold_to_decision_table_combines_grouped_thresholds_with_same_rule() -> None:
    """
    Verify repeated grouped thresholds share one rule column.

    Group-specific thresholds for the same metric/rule should OR their failures
    into the same decision column.
    """

    # Build cell metrics.
    metrics = make_cell_metrics()

    # Build two grouped thresholds with the same rule name.
    threshold_sample_1 = QCThreshold(
        axis="cell",
        metric="log1p_total_counts",
        rule_name="mad_log1p_total_counts",
        lower=1.5,
        upper=2.5,
        source="mad",
        groupby_columns=("sample_id",),
        group_values=("sample_1",),
    )
    threshold_sample_2 = QCThreshold(
        axis="cell",
        metric="log1p_total_counts",
        rule_name="mad_log1p_total_counts",
        lower=2.5,
        upper=5.0,
        source="mad",
        groupby_columns=("sample_id",),
        group_values=("sample_2",),
    )

    # Initialize the decision table.
    decisions = initialize_decision_table(
        metrics.index,
        [threshold_sample_1, threshold_sample_2],
    )

    # Apply both thresholds.
    apply_threshold_to_decision_table(
        metrics=metrics,
        decisions=decisions,
        threshold=threshold_sample_1,
    )
    apply_threshold_to_decision_table(
        metrics=metrics,
        decisions=decisions,
        threshold=threshold_sample_2,
    )

    # Confirm sample_1 low cell and sample_2 high cell failed the shared rule.
    assert decisions["mad_log1p_total_counts"].tolist() == [True, False, False, True]


def test_apply_threshold_to_decision_table_rejects_threshold_without_bounds() -> None:
    """
    Verify thresholds must contain at least one bound.

    A threshold with neither lower nor upper cannot produce a decision.
    """

    # Build metrics.
    metrics = make_cell_metrics()

    # Build a threshold with no bounds.
    threshold = QCThreshold(
        axis="cell",
        metric="total_counts",
        rule_name="bad_rule",
        lower=None,
        upper=None,
        source="fixed",
    )

    # Initialize the decision table.
    decisions = initialize_decision_table(metrics.index, [threshold])

    # Confirm the threshold fails clearly.
    with pytest.raises(QCDecisionError, match="has no bounds"):
        apply_threshold_to_decision_table(
            metrics=metrics,
            decisions=decisions,
            threshold=threshold,
        )


def test_apply_threshold_to_decision_table_rejects_missing_metric_column() -> None:
    """
    Verify thresholds referencing missing metrics fail clearly.

    Missing metric columns should never silently pass.
    """

    # Build metrics.
    metrics = make_cell_metrics()

    # Build a threshold referencing a missing metric.
    threshold = QCThreshold(
        axis="cell",
        metric="missing_metric",
        rule_name="missing_metric_rule",
        lower=0.0,
        upper=None,
        source="fixed",
    )

    # Initialize the decision table.
    decisions = initialize_decision_table(metrics.index, [threshold])

    # Confirm missing metric columns fail clearly.
    with pytest.raises(QCDecisionError, match="missing required column"):
        apply_threshold_to_decision_table(
            metrics=metrics,
            decisions=decisions,
            threshold=threshold,
        )


def test_apply_threshold_to_decision_table_rejects_missing_groupby_column() -> None:
    """
    Verify group-specific thresholds require groupby columns.

    Group-specific decision logic cannot select rows if the groupby column is
    absent.
    """

    # Build metrics without the sample_id column.
    metrics = make_cell_metrics().drop(columns=["sample_id"])

    # Build a group-specific threshold.
    threshold = QCThreshold(
        axis="cell",
        metric="log1p_total_counts",
        rule_name="mad_log1p_total_counts",
        lower=1.5,
        upper=5.0,
        source="mad",
        groupby_columns=("sample_id",),
        group_values=("sample_1",),
    )

    # Initialize the decision table.
    decisions = initialize_decision_table(metrics.index, [threshold])

    # Confirm missing groupby columns fail clearly.
    with pytest.raises(QCDecisionError, match="missing required column"):
        apply_threshold_to_decision_table(
            metrics=metrics,
            decisions=decisions,
            threshold=threshold,
        )


def test_apply_thresholds_to_metric_table_finalizes_cell_decisions() -> None:
    """
    Verify applying multiple thresholds returns finalized cell decisions.

    The finalized decision table should include keep, fail_any_qc, failed_rules,
    and one boolean column per threshold rule.
    """

    # Build metrics and thresholds.
    metrics = make_cell_metrics()
    thresholds = make_cell_thresholds()

    # Apply thresholds.
    decisions, warnings = apply_thresholds_to_metric_table(
        metrics=metrics,
        thresholds=thresholds,
        axis="cell",
    )

    # Confirm summary columns are first.
    assert list(decisions.columns[:3]) == ["keep", "fail_any_qc", "failed_rules"]

    # Confirm every cell failed at least one rule with this threshold set.
    assert decisions["keep"].tolist() == [False, True, False, False]
    assert decisions["fail_any_qc"].tolist() == [True, False, True, True]

    # Confirm failed rule strings.
    assert decisions.loc["cell_1", "failed_rules"] == (
        "fixed_min_genes_per_cell;mad_log1p_total_counts"
    )
    assert decisions.loc["cell_2", "failed_rules"] == ""
    assert decisions.loc["cell_3", "failed_rules"] == "fixed_max_mito_percent"
    assert decisions.loc["cell_4", "failed_rules"] == (
        "fixed_max_mito_percent;mad_log1p_total_counts"
    )

    # Confirm no warnings were emitted.
    assert warnings == []


def test_apply_thresholds_to_metric_table_rejects_wrong_axis_threshold() -> None:
    """
    Verify thresholds cannot be applied to the wrong axis table.

    Cell thresholds should not be applied to gene metrics, and vice versa.
    """

    # Build gene metrics and a cell threshold.
    metrics = make_gene_metrics()
    threshold = make_cell_thresholds()[0]

    # Confirm wrong-axis thresholds fail clearly.
    with pytest.raises(QCDecisionError, match="Cannot apply cell threshold"):
        apply_thresholds_to_metric_table(
            metrics=metrics,
            thresholds=[threshold],
            axis="gene",
        )


def test_finalize_decision_table_handles_no_rules() -> None:
    """
    Verify finalization handles threshold-free decision tables.

    When no thresholds exist, all rows should be kept.
    """

    # Build an empty decision table with three rows.
    decisions = pd.DataFrame(index=["row_1", "row_2", "row_3"])

    # Finalize the decision table.
    finalized = finalize_decision_table(decisions)

    # Confirm all rows are kept.
    assert finalized["keep"].tolist() == [True, True, True]

    # Confirm no rows failed.
    assert finalized["fail_any_qc"].tolist() == [False, False, False]

    # Confirm failed rule strings are empty.
    assert finalized["failed_rules"].tolist() == ["", "", ""]


def test_build_failed_rule_strings_returns_semicolon_joined_rule_names() -> None:
    """
    Verify failed-rule string construction.

    Decision tables should expose a compact human-readable failed_rules column.
    """

    # Build a decision table with rule columns.
    decisions = pd.DataFrame(
        {
            "rule_a": [True, False, True],
            "rule_b": [False, False, True],
        },
        index=["row_1", "row_2", "row_3"],
    )

    # Build failed-rule strings.
    failed_rules = build_failed_rule_strings(decisions, ["rule_a", "rule_b"])

    # Confirm semicolon-joined failed rules.
    assert failed_rules.tolist() == ["rule_a", "", "rule_a;rule_b"]


def test_build_decision_summary_counts_cells_genes_and_rules() -> None:
    """
    Verify decision summaries count kept/failed rows and rule failures.

    The summary will later be written into QC artifacts and provenance.
    """

    # Build finalized cell decisions.
    cell_decisions = finalize_decision_table(
        pd.DataFrame(
            {
                "rule_a": [True, False, True],
                "rule_b": [False, False, True],
            },
            index=["cell_1", "cell_2", "cell_3"],
        )
    )

    # Build finalized gene decisions.
    gene_decisions = finalize_decision_table(
        pd.DataFrame(
            {
                "gene_rule": [True, False],
            },
            index=["gene_1", "gene_2"],
        )
    )

    # Build the summary.
    summary = build_decision_summary(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
    )

    # Confirm summary fields.
    assert summary == {
        "n_cells": 3,
        "n_cells_kept": 1,
        "n_cells_failed": 2,
        "n_genes": 2,
        "n_genes_kept": 1,
        "n_genes_failed": 1,
        "cell_failures_by_rule": {
            "rule_a": 2,
            "rule_b": 1,
        },
        "gene_failures_by_rule": {
            "gene_rule": 1,
        },
    }


def test_count_failures_by_rule_excludes_summary_columns() -> None:
    """
    Verify failure counting ignores keep/fail summary columns.

    Only per-rule boolean columns should be counted.
    """

    # Build a finalized decision table.
    decisions = finalize_decision_table(
        pd.DataFrame(
            {
                "rule_a": [True, False],
                "rule_b": [True, True],
            }
        )
    )

    # Count failures by rule.
    counts = count_failures_by_rule(decisions)

    # Confirm only rule columns were counted.
    assert counts == {
        "rule_a": 1,
        "rule_b": 2,
    }


def test_build_qc_decisions_applies_cell_and_gene_thresholds() -> None:
    """
    Verify full decision construction applies cell and gene thresholds.

    This is the primary integration test for decisions.py.
    """

    # Build cell and gene metrics.
    cell_metrics = make_cell_metrics()
    gene_metrics = make_gene_metrics()

    # Build combined thresholds.
    thresholds = QCThresholdResult(thresholds=[*make_cell_thresholds(), *make_gene_thresholds()])

    # Build QC decisions.
    result = build_qc_decisions(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        thresholds=thresholds,
        config=QCConfig(mode="flag_no_drop"),
    )

    # Confirm a structured result was returned.
    assert isinstance(result, QCDecisionResult)

    # Confirm cell decisions.
    assert result.cell_decisions["keep"].tolist() == [False, True, False, False]

    # Confirm gene decisions.
    assert result.gene_decisions["keep"].tolist() == [False, True, True]

    # Confirm summary fields.
    assert result.summary["n_cells"] == 4
    assert result.summary["n_cells_kept"] == 1
    assert result.summary["n_cells_failed"] == 3
    assert result.summary["n_genes"] == 3
    assert result.summary["n_genes_kept"] == 2
    assert result.summary["n_genes_failed"] == 1

    # Confirm warnings are preserved as a list.
    assert result.warnings == []


def test_qc_decision_result_summary_dict_includes_warnings() -> None:
    """
    Verify decision result summary serialization includes warnings.

    This keeps artifact and provenance writing simple.
    """

    # Build a decision result.
    result = QCDecisionResult(
        cell_decisions=pd.DataFrame(),
        gene_decisions=pd.DataFrame(),
        summary={"n_cells": 0},
        warnings=["example warning"],
    )

    # Convert the summary to a dictionary.
    payload = result.to_summary_dict()

    # Confirm summary fields were preserved.
    assert payload["n_cells"] == 0

    # Confirm warnings were included.
    assert payload["warnings"] == ["example warning"]

    # Mutate the returned warning list.
    payload["warnings"].append("mutated")

    # Confirm the original warnings were not mutated.
    assert result.warnings == ["example warning"]


def test_build_qc_decisions_rejects_invalid_config_type() -> None:
    """
    Verify full decision construction rejects invalid config objects.

    This prevents accidental dictionary use from reaching config helper methods.
    """

    # Confirm invalid config input fails clearly.
    with pytest.raises(QCDecisionError, match="QCConfig object"):
        build_qc_decisions(
            cell_metrics=make_cell_metrics(),
            gene_metrics=make_gene_metrics(),
            thresholds=[],
            config={"mode": "filter"},  # type: ignore[arg-type]
        )


def test_build_qc_decisions_rejects_invalid_metric_tables() -> None:
    """
    Verify full decision construction validates metric tables.

    Cell and gene metrics must be non-empty pandas DataFrames.
    """

    # Confirm invalid cell metrics fail clearly.
    with pytest.raises(QCDecisionError, match="cell_metrics must be a pandas DataFrame"):
        build_qc_decisions(
            cell_metrics={"not": "dataframe"},  # type: ignore[arg-type]
            gene_metrics=make_gene_metrics(),
            thresholds=[],
        )

    # Confirm invalid gene metrics fail clearly.
    with pytest.raises(QCDecisionError, match="gene_metrics must be a pandas DataFrame"):
        build_qc_decisions(
            cell_metrics=make_cell_metrics(),
            gene_metrics={"not": "dataframe"},  # type: ignore[arg-type]
            thresholds=[],
        )


def test_validate_non_empty_filtered_result_accepts_non_empty_results() -> None:
    """
    Verify non-empty filtered-result validation accepts retained cells and genes.

    This helper protects filter mode from silently producing empty datasets.
    """

    # Build cell decisions with one kept cell.
    cell_decisions = pd.DataFrame(
        {
            "keep": [True, False],
            "fail_any_qc": [False, True],
            "failed_rules": ["", "rule"],
        }
    )

    # Build gene decisions with one kept gene.
    gene_decisions = pd.DataFrame(
        {
            "keep": [True, False],
            "fail_any_qc": [False, True],
            "failed_rules": ["", "rule"],
        }
    )

    # Confirm validation does not raise.
    validate_non_empty_filtered_result(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
    )


def test_validate_non_empty_filtered_result_rejects_all_cells_removed() -> None:
    """
    Verify empty cell results fail in strict filter mode.

    Removing all cells is almost always a configuration or data problem.
    """

    # Build cell decisions with no kept cells.
    cell_decisions = pd.DataFrame(
        {
            "keep": [False, False],
            "fail_any_qc": [True, True],
            "failed_rules": ["rule", "rule"],
        }
    )

    # Build gene decisions with one kept gene.
    gene_decisions = pd.DataFrame(
        {
            "keep": [True],
            "fail_any_qc": [False],
            "failed_rules": [""],
        }
    )

    # Confirm all-cell removal fails clearly.
    with pytest.raises(QCDecisionError, match="remove all cells"):
        validate_non_empty_filtered_result(
            cell_decisions=cell_decisions,
            gene_decisions=gene_decisions,
        )


def test_validate_non_empty_filtered_result_rejects_all_genes_removed() -> None:
    """
    Verify empty gene results fail in strict filter mode.

    Removing all genes is almost always a configuration or data problem.
    """

    # Build cell decisions with one kept cell.
    cell_decisions = pd.DataFrame(
        {
            "keep": [True],
            "fail_any_qc": [False],
            "failed_rules": [""],
        }
    )

    # Build gene decisions with no kept genes.
    gene_decisions = pd.DataFrame(
        {
            "keep": [False, False],
            "fail_any_qc": [True, True],
            "failed_rules": ["rule", "rule"],
        }
    )

    # Confirm all-gene removal fails clearly.
    with pytest.raises(QCDecisionError, match="remove all genes"):
        validate_non_empty_filtered_result(
            cell_decisions=cell_decisions,
            gene_decisions=gene_decisions,
        )


def test_build_qc_decisions_filter_mode_rejects_empty_filtered_cells() -> None:
    """
    Verify full decision construction enforces non-empty cell results in filter mode.

    Report-only mode can audit an all-fail result, but filter mode should fail
    unless explicitly configured otherwise.
    """

    # Build cell metrics where every cell fails min genes.
    cell_metrics = make_cell_metrics()

    # Build gene metrics.
    gene_metrics = make_gene_metrics()

    # Build a strict cell threshold that fails every cell.
    thresholds = [
        QCThreshold(
            axis="cell",
            metric="n_genes_by_counts",
            rule_name="fixed_min_genes_per_cell",
            lower=1000.0,
            upper=None,
            source="fixed",
        )
    ]

    # Confirm filter mode rejects all-cell removal.
    with pytest.raises(QCDecisionError, match="remove all cells"):
        build_qc_decisions(
            cell_metrics=cell_metrics,
            gene_metrics=gene_metrics,
            thresholds=thresholds,
            config=QCConfig(mode="filter", fail_on_empty_result=True),
        )


def test_build_qc_decisions_allows_empty_filtered_result_when_configured() -> None:
    """
    Verify all-fail filtering can be allowed explicitly.

    Some diagnostic tests may intentionally allow empty filtered results.
    """

    # Build cell metrics where every cell fails min genes.
    cell_metrics = make_cell_metrics()

    # Build gene metrics.
    gene_metrics = make_gene_metrics()

    # Build a strict cell threshold that fails every cell.
    thresholds = [
        QCThreshold(
            axis="cell",
            metric="n_genes_by_counts",
            rule_name="fixed_min_genes_per_cell",
            lower=1000.0,
            upper=None,
            source="fixed",
        )
    ]

    # Build decisions with empty-result failure disabled.
    result = build_qc_decisions(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        thresholds=thresholds,
        config=QCConfig(mode="filter", fail_on_empty_result=False),
    )

    # Confirm all cells failed but no exception was raised.
    assert result.summary["n_cells_kept"] == 0
    assert result.summary["n_cells_failed"] == 4


def test_validate_decision_metric_table_accepts_non_empty_dataframe() -> None:
    """
    Verify decision metric table validation accepts non-empty DataFrames.

    Valid metric tables should pass silently.
    """

    # Confirm validation does not raise.
    validate_decision_metric_table(
        pd.DataFrame({"metric": [1.0]}),
        table_name="cell_metrics",
    )


def test_validate_decision_metric_table_rejects_non_dataframe() -> None:
    """
    Verify decision metric table validation rejects non-DataFrame inputs.

    Decision construction should fail before trying to access DataFrame methods.
    """

    # Confirm non-DataFrame inputs fail clearly.
    with pytest.raises(QCDecisionError, match="must be a pandas DataFrame"):
        validate_decision_metric_table(
            {"metric": [1.0]},  # type: ignore[arg-type]
            table_name="cell_metrics",
        )


def test_validate_decision_metric_table_rejects_empty_dataframe() -> None:
    """
    Verify decision metric table validation rejects empty DataFrames.

    Decision construction needs actual metric rows and columns.
    """

    # Confirm empty DataFrames fail clearly.
    with pytest.raises(QCDecisionError, match="at least one row and one column"):
        validate_decision_metric_table(pd.DataFrame(), table_name="cell_metrics")


def test_require_decision_metric_columns_accepts_existing_columns() -> None:
    """
    Verify decision metric-column validation accepts existing columns.

    Threshold application should proceed when all required metric columns exist.
    """

    # Build a metric table.
    table = pd.DataFrame({"a": [1.0], "b": [2.0]})

    # Confirm existing columns pass.
    require_decision_metric_columns(table, ["a", "b"], table_name="cell_metrics")


def test_require_decision_metric_columns_rejects_single_string_argument() -> None:
    """
    Verify decision metric-column validation rejects single strings.

    Strings are iterable, so callers must provide an explicit sequence of column
    names.
    """

    # Build a metric table.
    table = pd.DataFrame({"a": [1.0]})

    # Confirm single-string columns fail clearly.
    with pytest.raises(QCDecisionError, match="not a string"):
        require_decision_metric_columns(
            table,
            "a",  # type: ignore[arg-type]
            table_name="cell_metrics",
        )


def test_require_decision_metric_columns_rejects_missing_columns() -> None:
    """
    Verify decision metric-column validation rejects missing columns.

    Missing columns should fail before threshold application.
    """

    # Build a metric table.
    table = pd.DataFrame({"a": [1.0]})

    # Confirm missing columns fail clearly.
    with pytest.raises(QCDecisionError, match="missing required column"):
        require_decision_metric_columns(table, ["a", "b"], table_name="cell_metrics")


def _keep_frame(keep: list[bool], index: list[str]) -> pd.DataFrame:
    """
    Build a minimal finalized decision table for QC report-table tests.

    Only the ``keep`` column is required by the report-table builder, so the
    fixture keeps just that column plus its complement to mirror the real
    finalized decision table shape.
    """

    # Return a minimal decision table indexed by cell.
    return pd.DataFrame(
        {"keep": keep, "fail_any_qc": [not value for value in keep]},
        index=index,
    )


def test_build_qc_report_table_grouped_counts_and_total() -> None:
    """
    Verify the QC report table reports per-group before/removed/%/after + TOTAL.

    Group ``A`` loses one of two cells; group ``B`` keeps all three. The TOTAL
    row must equal the column-wise sum of the group rows, and the group rows
    must be deterministically ordered with TOTAL last.
    """

    # Build a five-cell decision table: A has 1 of 2 removed, B keeps all 3.
    decisions = _keep_frame(
        keep=[True, False, True, True, True],
        index=["c1", "c2", "c3", "c4", "c5"],
    )
    groups = pd.Series(
        ["A", "A", "B", "B", "B"],
        index=["c1", "c2", "c3", "c4", "c5"],
    )

    # Build the report table.
    report = build_qc_report_table(decisions, groups=groups)

    # Confirm rows are ordered A, B, TOTAL.
    assert list(report["cell_type"]) == ["A", "B", "TOTAL"]

    # Confirm per-group counts.
    row_a = report.loc[report["cell_type"] == "A"].iloc[0]
    assert int(row_a["cells_before_qc"]) == 2
    assert int(row_a["cells_removed"]) == 1
    assert row_a["pct_removed"] == pytest.approx(50.0)
    assert int(row_a["cells_after_qc"]) == 1

    row_b = report.loc[report["cell_type"] == "B"].iloc[0]
    assert int(row_b["cells_before_qc"]) == 3
    assert int(row_b["cells_removed"]) == 0
    assert row_b["pct_removed"] == pytest.approx(0.0)
    assert int(row_b["cells_after_qc"]) == 3

    # Confirm the TOTAL row equals the group-wise sums.
    total = report.loc[report["cell_type"] == "TOTAL"].iloc[0]
    assert int(total["cells_before_qc"]) == 5
    assert int(total["cells_removed"]) == 1
    assert total["pct_removed"] == pytest.approx(20.0)
    assert int(total["cells_after_qc"]) == 4


def test_build_qc_report_table_without_groups_is_single_total() -> None:
    """
    Verify the report table collapses to a single TOTAL row without groups.

    When no per-cell grouping is available (e.g. QC runs before annotation),
    the report should still summarize the whole cohort in one TOTAL row.
    """

    # Build a decision table with two of four cells removed.
    decisions = _keep_frame(
        keep=[True, False, True, False],
        index=["c1", "c2", "c3", "c4"],
    )

    # Build the report table with no grouping.
    report = build_qc_report_table(decisions, groups=None)

    # Confirm a single TOTAL row with correct counts.
    assert list(report["cell_type"]) == ["TOTAL"]
    total = report.iloc[0]
    assert int(total["cells_before_qc"]) == 4
    assert int(total["cells_removed"]) == 2
    assert total["pct_removed"] == pytest.approx(50.0)
    assert int(total["cells_after_qc"]) == 2


def test_build_qc_report_table_missing_group_labels_go_to_unassigned() -> None:
    """
    Verify cells with a missing group label are bucketed, preserving the sum.

    A NaN group label must not silently drop a cell from the accounting: the
    group rows must still sum to the TOTAL row.
    """

    # Build a decision table where one cell has no group label.
    decisions = _keep_frame(
        keep=[True, False, True],
        index=["c1", "c2", "c3"],
    )
    groups = pd.Series(["A", "A", np.nan], index=["c1", "c2", "c3"])

    # Build the report table.
    report = build_qc_report_table(decisions, groups=groups)

    # Confirm the NaN-labeled cell is bucketed under 'unassigned'.
    assert "unassigned" in set(report["cell_type"])

    # Confirm the group rows still sum to the TOTAL row.
    total = report.loc[report["cell_type"] == "TOTAL"].iloc[0]
    non_total = report.loc[report["cell_type"] != "TOTAL"]
    assert int(non_total["cells_before_qc"].sum()) == int(total["cells_before_qc"])
    assert int(non_total["cells_removed"].sum()) == int(total["cells_removed"])


def test_build_qc_report_table_requires_keep_column() -> None:
    """
    Verify the report-table builder rejects a decision table without ``keep``.

    The ``keep`` column is the sole required input; its absence is a caller
    error, not something to guess around.
    """

    # Build a decision table missing the keep column.
    decisions = pd.DataFrame({"fail_any_qc": [True, False]}, index=["c1", "c2"])

    # Confirm the missing keep column fails clearly.
    with pytest.raises(QCDecisionError, match="keep"):
        build_qc_report_table(decisions)
