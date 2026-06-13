"""QC decision-table construction utilities for CellQuorum."""

from __future__ import annotations

# Import Sequence for accepting threshold record collections.
from collections.abc import Sequence

# Import dataclass helpers for structured QC decision results.
from dataclasses import dataclass, field

# Import NumPy for finite-value checks.
import numpy as np

# Import pandas for decision-table construction.
import pandas as pd

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import QC configuration.
from cellquorum.qc.config import QCConfig

# Import threshold records and threshold result containers.
from cellquorum.qc.thresholds import QCThreshold, QCThresholdResult, ThresholdAxis


class QCDecisionError(CellQuorumDataError):
    """
    Report QC decision construction failures.

    QC decisions are the explicit application of threshold records to metric
    tables. Errors here should fail clearly because this layer determines which
    cells and genes would be kept or filtered.
    """


@dataclass(frozen=True)
class QCDecisionResult:
    """
    Store cell and gene QC decision tables.

    Args:
        cell_decisions: Cell-level keep/fail decision table.
        gene_decisions: Gene-level keep/fail decision table.
        summary: JSON-friendly QC decision summary.
        warnings: Non-fatal decision construction warnings.
    """

    # Store cell-level QC decisions.
    cell_decisions: pd.DataFrame

    # Store gene-level QC decisions.
    gene_decisions: pd.DataFrame

    # Store JSON-friendly decision summary values.
    summary: dict[str, object]

    # Store non-fatal warnings.
    warnings: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, object]:
        """
        Return a JSON-friendly summary dictionary.

        Returns:
            Dictionary containing summary values and warnings.
        """

        # Copy the summary so callers cannot mutate the stored object.
        payload = dict(self.summary)

        # Add warning messages.
        payload["warnings"] = list(self.warnings)

        # Return the summary payload.
        return payload


def build_qc_decisions(
    *,
    cell_metrics: pd.DataFrame,
    gene_metrics: pd.DataFrame,
    thresholds: QCThresholdResult | Sequence[QCThreshold],
    config: QCConfig | None = None,
) -> QCDecisionResult:
    """
    Apply QC thresholds to cell and gene metric tables.

    This function does not drop rows. It returns explicit decision tables with
    one boolean column per threshold rule, plus `fail_any_qc`, `failed_rules`,
    and `keep`. Filtering is performed later by the QC stage using these tables.

    Args:
        cell_metrics: Cell-level QC metric table.
        gene_metrics: Gene-level QC metric table.
        thresholds: Threshold records or a QCThresholdResult.
        config: Optional QC configuration. Defaults to QCConfig().

    Returns:
        QCDecisionResult containing cell decisions, gene decisions, summary, and
        warnings.

    Raises:
        QCDecisionError: If metrics, thresholds, or configured empty-result
            behavior are invalid.
    """

    # Resolve the QC configuration.
    qc_config = QCConfig() if config is None else config

    # Validate the QC configuration type.
    if not isinstance(qc_config, QCConfig):
        raise QCDecisionError(
            "build_qc_decisions expected config to be a QCConfig object. "
            f"Received: {type(qc_config).__name__}."
        )

    # Validate cell metrics.
    validate_decision_metric_table(cell_metrics, table_name="cell_metrics")

    # Validate gene metrics.
    validate_decision_metric_table(gene_metrics, table_name="gene_metrics")

    # Normalize threshold records.
    threshold_records = normalize_threshold_records(thresholds)

    # Split thresholds by axis.
    cell_thresholds = [threshold for threshold in threshold_records if threshold.axis == "cell"]
    gene_thresholds = [threshold for threshold in threshold_records if threshold.axis == "gene"]

    # Apply cell-level thresholds.
    cell_decisions, cell_warnings = apply_thresholds_to_metric_table(
        metrics=cell_metrics,
        thresholds=cell_thresholds,
        axis="cell",
    )

    # Apply gene-level thresholds.
    gene_decisions, gene_warnings = apply_thresholds_to_metric_table(
        metrics=gene_metrics,
        thresholds=gene_thresholds,
        axis="gene",
    )

    # Combine warnings.
    warnings = [*cell_warnings, *gene_warnings]

    # Build decision summary.
    summary = build_decision_summary(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
    )

    # Validate non-empty filtered results when filtering is requested.
    if qc_config.should_filter() and qc_config.fail_on_empty_result:
        validate_non_empty_filtered_result(
            cell_decisions=cell_decisions,
            gene_decisions=gene_decisions,
        )

    # Return the structured decision result.
    return QCDecisionResult(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
        summary=summary,
        warnings=warnings,
    )


def normalize_threshold_records(
    thresholds: QCThresholdResult | Sequence[QCThreshold],
) -> list[QCThreshold]:
    """
    Normalize threshold inputs into a list of QCThreshold records.

    Args:
        thresholds: QCThresholdResult or sequence of QCThreshold records.

    Returns:
        List of threshold records.

    Raises:
        QCDecisionError: If threshold input is invalid.
    """

    # Extract threshold records from a QCThresholdResult.
    if isinstance(thresholds, QCThresholdResult):
        return list(thresholds.thresholds)

    # Reject strings because they are sequences but invalid here.
    if isinstance(thresholds, str):
        raise QCDecisionError("thresholds must be QCThreshold records, not a string.")

    # Validate sequence-like threshold input.
    if not isinstance(thresholds, Sequence):
        raise QCDecisionError(
            "thresholds must be a QCThresholdResult or a sequence of QCThreshold records. "
            f"Received: {type(thresholds).__name__}."
        )

    # Convert threshold records to a list.
    threshold_list = list(thresholds)

    # Validate each threshold record.
    for threshold in threshold_list:
        # Reject non-threshold entries.
        if not isinstance(threshold, QCThreshold):
            raise QCDecisionError(
                "thresholds must contain only QCThreshold records. "
                f"Received: {type(threshold).__name__}."
            )

    # Return normalized threshold records.
    return threshold_list


def apply_thresholds_to_metric_table(
    *,
    metrics: pd.DataFrame,
    thresholds: Sequence[QCThreshold],
    axis: ThresholdAxis,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply threshold records to one metric table.

    Args:
        metrics: Metric table indexed by cells or genes.
        thresholds: Threshold records for the given axis.
        axis: Axis label for the metric table.

    Returns:
        Tuple containing the finalized decision table and warnings.

    Raises:
        QCDecisionError: If a threshold cannot be applied.
    """

    # Validate metric table input.
    validate_decision_metric_table(metrics, table_name=f"{axis}_metrics")

    # Validate threshold axes.
    for threshold in thresholds:
        # Reject thresholds for the wrong axis.
        if threshold.axis != axis:
            raise QCDecisionError(
                f"Cannot apply {threshold.axis} threshold '{threshold.rule_name}' "
                f"to {axis} metric table."
            )

    # Build an empty decision table with one column per unique rule.
    decisions = initialize_decision_table(metrics.index, thresholds)

    # Initialize warning messages.
    warnings: list[str] = []

    # Apply each threshold record.
    for threshold in thresholds:
        # Apply one threshold record and collect warnings.
        threshold_warnings = apply_threshold_to_decision_table(
            metrics=metrics,
            decisions=decisions,
            threshold=threshold,
        )

        # Store threshold warnings.
        warnings.extend(threshold_warnings)

    # Finalize keep/fail columns and failed rule summaries.
    finalized_decisions = finalize_decision_table(decisions)

    # Return the finalized decision table and warnings.
    return finalized_decisions, warnings


def initialize_decision_table(
    index: pd.Index,
    thresholds: Sequence[QCThreshold],
) -> pd.DataFrame:
    """
    Initialize a decision table with one boolean column per threshold rule.

    Args:
        index: Cell or gene index for the decision table.
        thresholds: Threshold records that will be applied.

    Returns:
        Decision table initialized with all-False rule columns.
    """

    # Initialize the decision table.
    decisions = pd.DataFrame(index=index)

    # Add one boolean failure column per unique rule name.
    for rule_name in unique_rule_names(thresholds):
        # Initialize all rows as not failing the rule.
        decisions[rule_name] = False

    # Return the initialized decision table.
    return decisions


def apply_threshold_to_decision_table(
    *,
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    threshold: QCThreshold,
) -> list[str]:
    """
    Apply one threshold to a decision table.

    Args:
        metrics: Metric table.
        decisions: Mutable decision table containing threshold rule columns.
        threshold: Threshold record to apply.

    Returns:
        Non-fatal warnings emitted while applying the threshold.

    Raises:
        QCDecisionError: If the threshold references missing columns or has no bounds.
    """

    # Validate that the threshold has at least one bound.
    if threshold.lower is None and threshold.upper is None:
        raise QCDecisionError(
            f"Threshold '{threshold.rule_name}' for metric '{threshold.metric}' has no bounds."
        )

    # Validate that the metric column exists.
    require_decision_metric_columns(
        metrics,
        [threshold.metric],
        table_name=f"{threshold.axis}_metrics",
    )

    # Validate that groupby columns exist for group-specific thresholds.
    if threshold.is_group_specific():
        require_decision_metric_columns(
            metrics,
            list(threshold.groupby_columns),
            table_name=f"{threshold.axis}_metrics",
        )

    # Validate that the decision table has this rule column.
    if threshold.rule_name not in decisions.columns:
        decisions[threshold.rule_name] = False

    # Build the row-selection mask for this threshold.
    selected_rows = build_threshold_row_selector(metrics, threshold)

    # Evaluate failures for selected rows.
    failures, warnings = evaluate_threshold_failures(
        metrics=metrics,
        threshold=threshold,
        selected_rows=selected_rows,
    )

    # OR failures into the rule column to support grouped thresholds sharing a rule name.
    decisions[threshold.rule_name] = decisions[threshold.rule_name] | failures

    # Return warnings.
    return warnings


def build_threshold_row_selector(
    metrics: pd.DataFrame,
    threshold: QCThreshold,
) -> pd.Series:
    """
    Build the row selector for a threshold.

    Global thresholds apply to every row. Group-specific thresholds apply only to
    rows matching the threshold's groupby column values.

    Args:
        metrics: Metric table.
        threshold: Threshold record.

    Returns:
        Boolean Series indexed like metrics.
    """

    # Start with all rows selected.
    selected_rows = pd.Series(True, index=metrics.index, dtype=bool)

    # Return all rows for global thresholds.
    if not threshold.is_group_specific():
        return selected_rows

    # Iterate over groupby columns and group values together.
    for column, value in zip(
        threshold.groupby_columns,
        threshold.group_values,
        strict=False,
    ):
        # Select missing values when the stored group value is the missing sentinel.
        if value == "<NA>":
            selected_rows = selected_rows & metrics[column].isna()

        # Otherwise select exact stringified group matches.
        else:
            selected_rows = selected_rows & (metrics[column].astype(str) == value)

    # Return the group-specific row selector.
    return selected_rows


def evaluate_threshold_failures(
    *,
    metrics: pd.DataFrame,
    threshold: QCThreshold,
    selected_rows: pd.Series,
) -> tuple[pd.Series, list[str]]:
    """
    Evaluate threshold failures for selected rows.

    Args:
        metrics: Metric table.
        threshold: Threshold record.
        selected_rows: Boolean row selector.

    Returns:
        Tuple containing failure mask and warning messages.
    """

    # Initialize all rows as not failing.
    failures = pd.Series(False, index=metrics.index, dtype=bool)

    # Initialize warnings.
    warnings: list[str] = []

    # Return early when a group-specific threshold matches no rows.
    if not bool(selected_rows.any()):
        warnings.append(
            f"Threshold '{threshold.rule_name}' for metric '{threshold.metric}' "
            "matched zero rows."
        )
        return failures, warnings

    # Convert selected metric values to numeric.
    numeric_values = pd.to_numeric(metrics.loc[selected_rows, threshold.metric], errors="coerce")

    # Identify finite values.
    finite_mask = np.isfinite(numeric_values.to_numpy(dtype=float))

    # Identify selected rows with non-finite metric values.
    non_finite_index = numeric_values.index[~finite_mask]

    # Treat non-finite selected values as failures.
    if len(non_finite_index) > 0:
        failures.loc[non_finite_index] = True
        warnings.append(
            f"Threshold '{threshold.rule_name}' for metric '{threshold.metric}' "
            f"encountered {len(non_finite_index)} non-finite value(s), marked as failures."
        )

    # Keep only finite selected values for numeric comparisons.
    finite_values = numeric_values.loc[finite_mask]

    # Apply lower-bound failures when configured.
    if threshold.lower is not None:
        failures.loc[finite_values.index] = failures.loc[finite_values.index] | (
            finite_values < threshold.lower
        )

    # Apply upper-bound failures when configured.
    if threshold.upper is not None:
        failures.loc[finite_values.index] = failures.loc[finite_values.index] | (
            finite_values > threshold.upper
        )

    # Return failures and warnings.
    return failures, warnings


def finalize_decision_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    Add keep/fail summary columns to a decision table.

    Args:
        decisions: Decision table containing one boolean column per QC rule.

    Returns:
        Finalized decision table with keep, fail_any_qc, and failed_rules columns.
    """

    # Copy the decision table to avoid mutating caller-owned data.
    finalized = decisions.copy()

    # Capture rule columns before adding summary columns.
    rule_columns = list(finalized.columns)

    # Convert all rule columns to boolean dtype.
    for column in rule_columns:
        # Ensure rule columns are boolean.
        finalized[column] = finalized[column].astype(bool)

    # Calculate whether each row failed any QC rule.
    if rule_columns:
        # Calculate row-wise failure status.
        fail_any_qc = finalized[rule_columns].any(axis=1)

    # Handle threshold-free decision tables.
    else:
        # Initialize every row as passing when no rules exist.
        fail_any_qc = pd.Series(False, index=finalized.index, dtype=bool)

    # Build failed-rule strings.
    failed_rules = build_failed_rule_strings(finalized, rule_columns)

    # Insert summary columns at the front.
    finalized.insert(0, "failed_rules", failed_rules)
    finalized.insert(0, "fail_any_qc", fail_any_qc)
    finalized.insert(0, "keep", ~fail_any_qc)

    # Return the finalized decision table.
    return finalized


def build_failed_rule_strings(
    decisions: pd.DataFrame,
    rule_columns: Sequence[str],
) -> pd.Series:
    """
    Build semicolon-separated failed-rule strings.

    Args:
        decisions: Decision table containing rule columns.
        rule_columns: Rule columns to inspect.

    Returns:
        Series containing semicolon-separated failed rule names.
    """

    # Return empty strings when there are no rule columns.
    if not rule_columns:
        return pd.Series("", index=decisions.index, dtype=str)

    # Build failed-rule strings row by row.
    return decisions[list(rule_columns)].apply(
        lambda row: ";".join([column for column in rule_columns if bool(row[column])]),
        axis=1,
    )


def build_decision_summary(
    *,
    cell_decisions: pd.DataFrame,
    gene_decisions: pd.DataFrame,
) -> dict[str, object]:
    """
    Build a JSON-friendly summary of QC decisions.

    Args:
        cell_decisions: Cell-level decision table.
        gene_decisions: Gene-level decision table.

    Returns:
        Decision summary dictionary.
    """

    # Build the summary dictionary.
    return {
        "n_cells": int(cell_decisions.shape[0]),
        "n_cells_kept": int(cell_decisions["keep"].sum()),
        "n_cells_failed": int(cell_decisions["fail_any_qc"].sum()),
        "n_genes": int(gene_decisions.shape[0]),
        "n_genes_kept": int(gene_decisions["keep"].sum()),
        "n_genes_failed": int(gene_decisions["fail_any_qc"].sum()),
        "cell_failures_by_rule": count_failures_by_rule(cell_decisions),
        "gene_failures_by_rule": count_failures_by_rule(gene_decisions),
    }


def count_failures_by_rule(decisions: pd.DataFrame) -> dict[str, int]:
    """
    Count failures for each QC rule column.

    Args:
        decisions: Finalized decision table.

    Returns:
        Mapping from rule name to failure count.
    """

    # Define non-rule summary columns.
    summary_columns = {"keep", "fail_any_qc", "failed_rules"}

    # Identify rule columns.
    rule_columns = [column for column in decisions.columns if column not in summary_columns]

    # Return failure counts for each rule.
    return {column: int(decisions[column].sum()) for column in rule_columns}


def validate_non_empty_filtered_result(
    *,
    cell_decisions: pd.DataFrame,
    gene_decisions: pd.DataFrame,
) -> None:
    """
    Validate that filtering would retain at least one cell and one gene.

    Args:
        cell_decisions: Cell-level decision table.
        gene_decisions: Gene-level decision table.

    Raises:
        QCDecisionError: If all cells or all genes would be filtered.
    """

    # Raise if no cells would be retained.
    if int(cell_decisions["keep"].sum()) == 0:
        raise QCDecisionError(
            "QC filtering would remove all cells. Adjust thresholds or set "
            "fail_on_empty_result=false to allow this explicitly."
        )

    # Raise if no genes would be retained.
    if int(gene_decisions["keep"].sum()) == 0:
        raise QCDecisionError(
            "QC filtering would remove all genes. Adjust thresholds or set "
            "fail_on_empty_result=false to allow this explicitly."
        )


def validate_decision_metric_table(table: pd.DataFrame, *, table_name: str) -> None:
    """
    Validate a metric table before applying QC decisions.

    Args:
        table: Candidate metric table.
        table_name: Human-readable table label.

    Raises:
        QCDecisionError: If the metric table is invalid.
    """

    # Validate the metric table type.
    if not isinstance(table, pd.DataFrame):
        raise QCDecisionError(
            f"{table_name} must be a pandas DataFrame. " f"Received: {type(table).__name__}."
        )

    # Reject empty metric tables.
    if table.empty:
        raise QCDecisionError(f"{table_name} must contain at least one row and one column.")


def require_decision_metric_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """
    Require metric columns before applying thresholds.

    Args:
        table: Metric table.
        columns: Required column names.
        table_name: Human-readable table label.

    Raises:
        QCDecisionError: If columns are invalid or missing.
    """

    # Reject a single string because strings are iterable.
    if isinstance(columns, str):
        raise QCDecisionError("columns must be a sequence of strings, not a string.")

    # Identify missing columns.
    missing_columns = [column for column in columns if column not in table.columns]

    # Raise a clear error when columns are missing.
    if missing_columns:
        raise QCDecisionError(
            f"{table_name} is missing required column(s): " f"{', '.join(missing_columns)}."
        )


def unique_rule_names(thresholds: Sequence[QCThreshold]) -> list[str]:
    """
    Return threshold rule names in first-seen order.

    Args:
        thresholds: Threshold records.

    Returns:
        Ordered unique rule names.
    """

    # Initialize seen rule names.
    seen: set[str] = set()

    # Initialize ordered rule names.
    rule_names: list[str] = []

    # Iterate over thresholds in input order.
    for threshold in thresholds:
        # Skip already-seen rule names.
        if threshold.rule_name in seen:
            continue

        # Mark the rule as seen.
        seen.add(threshold.rule_name)

        # Store the rule name.
        rule_names.append(threshold.rule_name)

    # Return ordered unique rule names.
    return rule_names


__all__ = [
    "QCDecisionError",
    "QCDecisionResult",
    "apply_threshold_to_decision_table",
    "apply_thresholds_to_metric_table",
    "build_decision_summary",
    "build_failed_rule_strings",
    "build_qc_decisions",
    "build_threshold_row_selector",
    "count_failures_by_rule",
    "evaluate_threshold_failures",
    "finalize_decision_table",
    "initialize_decision_table",
    "normalize_threshold_records",
    "require_decision_metric_columns",
    "unique_rule_names",
    "validate_decision_metric_table",
    "validate_non_empty_filtered_result",
]
