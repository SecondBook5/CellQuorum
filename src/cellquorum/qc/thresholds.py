"""QC threshold construction utilities for CellQuorum."""

from __future__ import annotations

# Import Sequence for validating metric and groupby inputs.
from collections.abc import Sequence

# Import dataclass helpers for structured threshold records.
from dataclasses import dataclass, field

# Import Literal for constrained threshold labels.
from typing import Literal

# Import NumPy for robust numeric threshold calculation.
import numpy as np

# Import pandas for metric table and threshold table handling.
import pandas as pd

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import QC configuration models.
from cellquorum.qc.config import QCConfig, QCMadThresholdConfig

# Define supported threshold axes.
type ThresholdAxis = Literal["cell", "gene"]

# Define supported threshold sources.
type ThresholdSource = Literal["fixed", "mad", "mad_mito"]


class QCThresholdError(CellQuorumDataError):
    """
    Report QC threshold construction failures.

    Thresholds are the bridge between metric calculation and filtering decisions.
    Errors here should be explicit because malformed thresholds would make cell
    and gene removal decisions impossible to audit.
    """


@dataclass(frozen=True)
class QCThreshold:
    """
    Store one QC threshold rule.

    A threshold rule describes how one metric should be filtered. It can apply to
    cells or genes, can have a lower bound, an upper bound, or both, and can be
    global or group-specific. Group-specific thresholds are used for per-sample,
    per-batch, or per-donor MAD filtering.

    Args:
        axis: Whether the threshold applies to cells or genes.
        metric: Metric column to threshold.
        rule_name: Stable rule name used in decision tables.
        lower: Optional inclusive lower bound.
        upper: Optional inclusive upper bound.
        source: Threshold source, such as fixed, mad, or mad_mito.
        groupby_columns: Metadata columns used for group-wise thresholds.
        group_values: Group values corresponding to groupby_columns.
        n_observations: Number of observations used to estimate the threshold.
    """

    # Store whether this threshold applies to cells or genes.
    axis: ThresholdAxis

    # Store the metric column to threshold.
    metric: str

    # Store a stable rule name for decision-table columns.
    rule_name: str

    # Store the optional inclusive lower bound.
    lower: float | None

    # Store the optional inclusive upper bound.
    upper: float | None

    # Store the threshold source.
    source: ThresholdSource

    # Store metadata columns used for group-wise thresholding.
    groupby_columns: tuple[str, ...] = field(default_factory=tuple)

    # Store metadata values used for group-wise thresholding.
    group_values: tuple[str, ...] = field(default_factory=tuple)

    # Store the number of observations used to estimate the threshold.
    n_observations: int | None = None

    def is_group_specific(self) -> bool:
        """
        Return whether this threshold is group-specific.

        Returns:
            True when groupby columns and values are present, otherwise False.
        """

        # Return whether group columns and values are present.
        return bool(self.groupby_columns and self.group_values)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the threshold record into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the threshold.
        """

        # Return a JSON-friendly threshold payload.
        return {
            "axis": self.axis,
            "metric": self.metric,
            "rule_name": self.rule_name,
            "lower": self.lower,
            "upper": self.upper,
            "source": self.source,
            "groupby_columns": list(self.groupby_columns),
            "group_values": list(self.group_values),
            "n_observations": self.n_observations,
        }


@dataclass(frozen=True)
class QCThresholdResult:
    """
    Store all QC thresholds and construction warnings.

    Args:
        thresholds: Threshold records.
        warnings: Non-fatal threshold construction warnings.
    """

    # Store threshold records.
    thresholds: list[QCThreshold]

    # Store non-fatal warnings.
    warnings: list[str] = field(default_factory=list)

    def cell_thresholds(self) -> list[QCThreshold]:
        """
        Return thresholds that apply to cells.

        Returns:
            Cell-level threshold records.
        """

        # Return thresholds whose axis is cell.
        return [threshold for threshold in self.thresholds if threshold.axis == "cell"]

    def gene_thresholds(self) -> list[QCThreshold]:
        """
        Return thresholds that apply to genes.

        Returns:
            Gene-level threshold records.
        """

        # Return thresholds whose axis is gene.
        return [threshold for threshold in self.thresholds if threshold.axis == "gene"]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert threshold records into a DataFrame.

        Returns:
            DataFrame containing one row per threshold record.
        """

        # Convert each threshold to a dictionary.
        rows = [threshold.to_dict() for threshold in self.thresholds]

        # Return an empty but schema-aware DataFrame when no thresholds exist.
        if not rows:
            return pd.DataFrame(
                columns=[
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
            )

        # Return the threshold table.
        return pd.DataFrame(rows)

    def to_summary_dict(self) -> dict[str, object]:
        """
        Convert threshold result metadata into a JSON-friendly summary.

        Returns:
            Summary dictionary.
        """

        # Return threshold summary metadata.
        return {
            "n_thresholds": len(self.thresholds),
            "n_cell_thresholds": len(self.cell_thresholds()),
            "n_gene_thresholds": len(self.gene_thresholds()),
            "warnings": list(self.warnings),
        }


def build_qc_thresholds(
    *,
    cell_metrics: pd.DataFrame,
    gene_metrics: pd.DataFrame,
    config: QCConfig | None = None,
) -> QCThresholdResult:
    """
    Build all configured QC thresholds.

    This function combines fixed threshold rules from QCConfig with adaptive MAD
    threshold rules estimated from cell-level metric tables. The returned
    threshold records are not applied here; they are consumed by the next QC
    decision layer.

    Args:
        cell_metrics: Cell-level QC metric table.
        gene_metrics: Gene-level QC metric table.
        config: Optional QC configuration. Defaults to QCConfig().

    Returns:
        QCThresholdResult containing threshold records and warnings.

    Raises:
        QCThresholdError: If metric tables or configuration are invalid.
    """

    # Resolve the QC configuration.
    qc_config = QCConfig() if config is None else config

    # Validate the QC configuration type.
    if not isinstance(qc_config, QCConfig):
        raise QCThresholdError(
            "build_qc_thresholds expected config to be a QCConfig object. "
            f"Received: {type(qc_config).__name__}."
        )

    # Validate the cell metric table.
    validate_metric_table(cell_metrics, table_name="cell_metrics")

    # Validate the gene metric table.
    validate_metric_table(gene_metrics, table_name="gene_metrics")

    # Initialize threshold records.
    thresholds: list[QCThreshold] = []

    # Initialize construction warnings.
    warnings: list[str] = []

    # Add fixed thresholds when requested.
    if qc_config.threshold_strategy in {"fixed", "fixed_and_mad"}:
        # Extend with fixed cell thresholds.
        thresholds.extend(build_fixed_cell_thresholds(qc_config))

        # Extend with fixed gene thresholds.
        thresholds.extend(build_fixed_gene_thresholds(qc_config))

    # Add MAD thresholds when requested and enabled.
    if qc_config.threshold_strategy in {"mad", "fixed_and_mad"} and qc_config.mad.enabled:
        # Build adaptive cell-level MAD thresholds.
        mad_result = build_mad_cell_thresholds(cell_metrics, qc_config.mad)

        # Extend threshold records with MAD thresholds.
        thresholds.extend(mad_result.thresholds)

        # Extend warnings with MAD construction warnings.
        warnings.extend(mad_result.warnings)

    # Return the combined threshold result.
    return QCThresholdResult(thresholds=thresholds, warnings=warnings)


def build_fixed_cell_thresholds(config: QCConfig) -> list[QCThreshold]:
    """
    Build fixed cell-level threshold records from QCConfig.

    Args:
        config: QC configuration.

    Returns:
        Fixed cell-level threshold records.
    """

    # Initialize threshold records.
    thresholds: list[QCThreshold] = []

    # Add minimum detected genes threshold when configured.
    if config.basic.min_genes_per_cell is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="n_genes_by_counts",
                rule_name="fixed_min_genes_per_cell",
                lower=float(config.basic.min_genes_per_cell),
                upper=None,
                source="fixed",
            )
        )

    # Add maximum detected genes threshold when configured.
    if config.basic.max_genes_per_cell is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="n_genes_by_counts",
                rule_name="fixed_max_genes_per_cell",
                lower=None,
                upper=float(config.basic.max_genes_per_cell),
                source="fixed",
            )
        )

    # Add minimum total counts threshold when configured.
    if config.basic.min_counts_per_cell is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="total_counts",
                rule_name="fixed_min_counts_per_cell",
                lower=float(config.basic.min_counts_per_cell),
                upper=None,
                source="fixed",
            )
        )

    # Add maximum total counts threshold when configured.
    if config.basic.max_counts_per_cell is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="total_counts",
                rule_name="fixed_max_counts_per_cell",
                lower=None,
                upper=float(config.basic.max_counts_per_cell),
                source="fixed",
            )
        )

    # Add maximum mitochondrial percentage threshold when configured.
    if config.basic.max_mito_percent is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="pct_counts_mito",
                rule_name="fixed_max_mito_percent",
                lower=None,
                upper=float(config.basic.max_mito_percent),
                source="fixed",
            )
        )

    # Add maximum ribosomal percentage threshold when configured.
    if config.basic.max_ribo_percent is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="pct_counts_ribo",
                rule_name="fixed_max_ribo_percent",
                lower=None,
                upper=float(config.basic.max_ribo_percent),
                source="fixed",
            )
        )

    # Add maximum hemoglobin percentage threshold when configured.
    if config.basic.max_hemoglobin_percent is not None:
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric="pct_counts_hemoglobin",
                rule_name="fixed_max_hemoglobin_percent",
                lower=None,
                upper=float(config.basic.max_hemoglobin_percent),
                source="fixed",
            )
        )

    # Return fixed cell-level thresholds.
    return thresholds


def build_fixed_gene_thresholds(config: QCConfig) -> list[QCThreshold]:
    """
    Build fixed gene-level threshold records from QCConfig.

    Args:
        config: QC configuration.

    Returns:
        Fixed gene-level threshold records.
    """

    # Initialize threshold records.
    thresholds: list[QCThreshold] = []

    # Add minimum cells per gene threshold when configured.
    if config.basic.min_cells_per_gene is not None:
        thresholds.append(
            QCThreshold(
                axis="gene",
                metric="n_cells_by_counts",
                rule_name="fixed_min_cells_per_gene",
                lower=float(config.basic.min_cells_per_gene),
                upper=None,
                source="fixed",
            )
        )

    # Return fixed gene-level thresholds.
    return thresholds


def build_mad_cell_thresholds(
    cell_metrics: pd.DataFrame,
    config: QCMadThresholdConfig,
) -> QCThresholdResult:
    """
    Build adaptive MAD cell-level threshold records.

    Args:
        cell_metrics: Cell-level QC metric table.
        config: MAD threshold configuration.

    Returns:
        QCThresholdResult containing MAD threshold records and warnings.

    Raises:
        QCThresholdError: If required metric or groupby columns are absent.
    """

    # Validate the cell metric table.
    validate_metric_table(cell_metrics, table_name="cell_metrics")

    # Validate all requested MAD metric columns.
    require_metric_columns(
        cell_metrics,
        [*config.metrics, config.mito_metric],
        table_name="cell_metrics",
    )

    # Validate groupby columns when requested.
    if config.groupby:
        require_metric_columns(cell_metrics, config.groupby, table_name="cell_metrics")

    # Initialize threshold records.
    thresholds: list[QCThreshold] = []

    # Initialize warnings.
    warnings: list[str] = []

    # Build general MAD thresholds.
    for metric in config.metrics:
        # Build thresholds for this general MAD metric.
        metric_result = build_mad_thresholds_for_metric(
            cell_metrics,
            metric=metric,
            n_mads=config.n_mads,
            source="mad",
            rule_name=f"mad_{metric}",
            groupby=config.groupby,
            skip_zero_mad=config.skip_zero_mad,
        )

        # Extend threshold records.
        thresholds.extend(metric_result.thresholds)

        # Extend warnings.
        warnings.extend(metric_result.warnings)

    # Build mitochondrial-specific MAD thresholds.
    mito_result = build_mad_thresholds_for_metric(
        cell_metrics,
        metric=config.mito_metric,
        n_mads=config.mito_n_mads,
        source="mad_mito",
        rule_name=f"mad_mito_{config.mito_metric}",
        groupby=config.groupby,
        skip_zero_mad=config.skip_zero_mad,
    )

    # Extend threshold records with mitochondrial thresholds.
    thresholds.extend(mito_result.thresholds)

    # Extend warnings with mitochondrial threshold warnings.
    warnings.extend(mito_result.warnings)

    # Return MAD threshold result.
    return QCThresholdResult(thresholds=thresholds, warnings=warnings)


def build_mad_thresholds_for_metric(
    cell_metrics: pd.DataFrame,
    *,
    metric: str,
    n_mads: float,
    source: ThresholdSource,
    rule_name: str,
    groupby: Sequence[str],
    skip_zero_mad: bool,
) -> QCThresholdResult:
    """
    Build MAD threshold records for one cell-level metric.

    Args:
        cell_metrics: Cell-level QC metric table.
        metric: Metric column to threshold.
        n_mads: Number of MADs used to define bounds.
        source: Threshold source label.
        rule_name: Stable rule name.
        groupby: Optional groupby columns.
        skip_zero_mad: Whether to skip zero-MAD thresholds.

    Returns:
        QCThresholdResult for one metric.
    """

    # Return a global threshold when no groupby columns are requested.
    if not groupby:
        return build_global_mad_threshold_for_metric(
            cell_metrics,
            metric=metric,
            n_mads=n_mads,
            source=source,
            rule_name=rule_name,
            skip_zero_mad=skip_zero_mad,
        )

    # Return group-specific thresholds when groupby columns are requested.
    return build_grouped_mad_thresholds_for_metric(
        cell_metrics,
        metric=metric,
        n_mads=n_mads,
        source=source,
        rule_name=rule_name,
        groupby=groupby,
        skip_zero_mad=skip_zero_mad,
    )


def build_global_mad_threshold_for_metric(
    cell_metrics: pd.DataFrame,
    *,
    metric: str,
    n_mads: float,
    source: ThresholdSource,
    rule_name: str,
    skip_zero_mad: bool,
) -> QCThresholdResult:
    """
    Build one global MAD threshold for one metric.

    Args:
        cell_metrics: Cell-level QC metric table.
        metric: Metric column to threshold.
        n_mads: Number of MADs used to define bounds.
        source: Threshold source label.
        rule_name: Stable rule name.
        skip_zero_mad: Whether to skip zero-MAD thresholds.

    Returns:
        QCThresholdResult containing zero or one threshold.
    """

    # Calculate MAD bounds for the metric.
    bounds = calculate_mad_bounds(
        cell_metrics[metric],
        n_mads=n_mads,
        metric=metric,
        skip_zero_mad=skip_zero_mad,
    )

    # Return only the warning when the threshold was skipped.
    if bounds.lower is None and bounds.upper is None:
        return QCThresholdResult(thresholds=[], warnings=bounds.warnings)

    # Build the global threshold record.
    threshold = QCThreshold(
        axis="cell",
        metric=metric,
        rule_name=rule_name,
        lower=bounds.lower,
        upper=bounds.upper,
        source=source,
        n_observations=bounds.n_observations,
    )

    # Return the threshold result.
    return QCThresholdResult(thresholds=[threshold], warnings=bounds.warnings)


def build_grouped_mad_thresholds_for_metric(
    cell_metrics: pd.DataFrame,
    *,
    metric: str,
    n_mads: float,
    source: ThresholdSource,
    rule_name: str,
    groupby: Sequence[str],
    skip_zero_mad: bool,
) -> QCThresholdResult:
    """
    Build group-specific MAD thresholds for one metric.

    Args:
        cell_metrics: Cell-level QC metric table.
        metric: Metric column to threshold.
        n_mads: Number of MADs used to define bounds.
        source: Threshold source label.
        rule_name: Stable rule name.
        groupby: Groupby columns used for per-group thresholds.
        skip_zero_mad: Whether to skip zero-MAD thresholds.

    Returns:
        QCThresholdResult containing group-specific thresholds and warnings.
    """

    # Initialize threshold records.
    thresholds: list[QCThreshold] = []

    # Initialize warnings.
    warnings: list[str] = []

    # Convert groupby columns to a tuple for stable records.
    groupby_columns = tuple(str(column) for column in groupby)

    # Iterate over groups without dropping missing values.
    for group_values_raw, group_frame in cell_metrics.groupby(list(groupby_columns), dropna=False):
        # Normalize group values into a tuple.
        group_values = normalize_group_values(group_values_raw)

        # Calculate MAD bounds for this group.
        bounds = calculate_mad_bounds(
            group_frame[metric],
            n_mads=n_mads,
            metric=metric,
            skip_zero_mad=skip_zero_mad,
            groupby_columns=groupby_columns,
            group_values=group_values,
        )

        # Extend warnings from this group.
        warnings.extend(bounds.warnings)

        # Skip threshold creation when bounds were skipped.
        if bounds.lower is None and bounds.upper is None:
            continue

        # Build the group-specific threshold record.
        thresholds.append(
            QCThreshold(
                axis="cell",
                metric=metric,
                rule_name=rule_name,
                lower=bounds.lower,
                upper=bounds.upper,
                source=source,
                groupby_columns=groupby_columns,
                group_values=group_values,
                n_observations=bounds.n_observations,
            )
        )

    # Return grouped threshold result.
    return QCThresholdResult(thresholds=thresholds, warnings=warnings)


@dataclass(frozen=True)
class MadBounds:
    """
    Store calculated MAD bounds and related warnings.

    Args:
        lower: Optional lower threshold bound.
        upper: Optional upper threshold bound.
        n_observations: Number of finite observations used.
        warnings: Non-fatal warnings emitted during calculation.
    """

    # Store the optional lower threshold.
    lower: float | None

    # Store the optional upper threshold.
    upper: float | None

    # Store the number of finite observations used.
    n_observations: int

    # Store non-fatal calculation warnings.
    warnings: list[str] = field(default_factory=list)


def calculate_mad_bounds(
    values: pd.Series,
    *,
    n_mads: float,
    metric: str,
    skip_zero_mad: bool,
    groupby_columns: tuple[str, ...] = (),
    group_values: tuple[str, ...] = (),
) -> MadBounds:
    """
    Calculate lower and upper MAD threshold bounds.

    Args:
        values: Metric values.
        n_mads: Number of MADs used to define bounds.
        metric: Metric name used in warnings.
        skip_zero_mad: Whether zero-MAD thresholds should be skipped.
        groupby_columns: Optional groupby columns for warning context.
        group_values: Optional group values for warning context.

    Returns:
        MadBounds containing bounds, observation count, and warnings.
    """

    # Convert values to numeric, coercing invalid entries to NaN.
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    # Keep only finite values.
    finite_values = numeric_values[np.isfinite(numeric_values)]

    # Count finite observations.
    n_observations = int(finite_values.size)

    # Build warning context.
    context = format_threshold_context(metric, groupby_columns, group_values)

    # Return a skipped threshold when there are no finite values.
    if n_observations == 0:
        return MadBounds(
            lower=None,
            upper=None,
            n_observations=0,
            warnings=[f"Skipped MAD threshold for {context}: no finite values."],
        )

    # Calculate the median.
    median = float(np.median(finite_values))

    # Calculate the raw median absolute deviation.
    mad = float(np.median(np.abs(finite_values - median)))

    # Handle zero-MAD metrics explicitly.
    if mad == 0.0:
        # Skip zero-MAD thresholds when configured.
        if skip_zero_mad:
            return MadBounds(
                lower=None,
                upper=None,
                n_observations=n_observations,
                warnings=[f"Skipped MAD threshold for {context}: MAD is zero."],
            )

        # Return exact-median bounds when zero-MAD thresholds are not skipped.
        return MadBounds(
            lower=median,
            upper=median,
            n_observations=n_observations,
            warnings=[f"MAD threshold for {context} has MAD zero; using median bounds."],
        )

    # Calculate the lower threshold bound.
    lower = median - (n_mads * mad)

    # Calculate the upper threshold bound.
    upper = median + (n_mads * mad)

    # Return calculated bounds.
    return MadBounds(
        lower=float(lower),
        upper=float(upper),
        n_observations=n_observations,
        warnings=[],
    )


def validate_metric_table(table: pd.DataFrame, *, table_name: str) -> None:
    """
    Validate a QC metric table.

    Args:
        table: Candidate metric table.
        table_name: Human-readable table label.

    Raises:
        QCThresholdError: If the table is invalid.
    """

    # Validate the metric table type.
    if not isinstance(table, pd.DataFrame):
        raise QCThresholdError(
            f"{table_name} must be a pandas DataFrame. " f"Received: {type(table).__name__}."
        )

    # Reject empty metric tables.
    if table.empty:
        raise QCThresholdError(f"{table_name} must contain at least one row and one column.")


def require_metric_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """
    Require metric table columns.

    Args:
        table: Metric table to inspect.
        columns: Required columns.
        table_name: Human-readable table label.

    Raises:
        QCThresholdError: If one or more columns are missing.
    """

    # Reject a single string because strings are iterable.
    if isinstance(columns, str):
        raise QCThresholdError("columns must be a sequence of strings, not a string.")

    # Identify missing columns.
    missing_columns = [column for column in columns if column not in table.columns]

    # Raise a clear error when columns are missing.
    if missing_columns:
        raise QCThresholdError(
            f"{table_name} is missing required column(s): " f"{', '.join(missing_columns)}."
        )


def normalize_group_values(group_values_raw: object) -> tuple[str, ...]:
    """
    Normalize pandas groupby keys into a tuple of strings.

    Args:
        group_values_raw: Raw groupby key returned by pandas.

    Returns:
        Tuple of group value strings.
    """

    # Preserve tuple group values.
    if isinstance(group_values_raw, tuple):
        return tuple(format_group_value(value) for value in group_values_raw)

    # Wrap scalar group values in a tuple.
    return (format_group_value(group_values_raw),)


def format_group_value(value: object) -> str:
    """
    Format one group value for threshold records.

    Args:
        value: Group value.

    Returns:
        String representation of the group value.
    """

    # Return a stable missing-value label for NaN group values.
    if pd.isna(value):
        return "<NA>"

    # Return the string representation.
    return str(value)


def format_threshold_context(
    metric: str,
    groupby_columns: tuple[str, ...],
    group_values: tuple[str, ...],
) -> str:
    """
    Format threshold calculation context for warnings.

    Args:
        metric: Metric name.
        groupby_columns: Optional groupby columns.
        group_values: Optional group values.

    Returns:
        Human-readable context string.
    """

    # Return metric-only context for global thresholds.
    if not groupby_columns:
        return metric

    # Build group context pairs.
    pairs = [
        f"{column}={value}" for column, value in zip(groupby_columns, group_values, strict=False)
    ]

    # Return metric plus group context.
    return f"{metric} ({', '.join(pairs)})"


__all__ = [
    "MadBounds",
    "QCThreshold",
    "QCThresholdError",
    "QCThresholdResult",
    "ThresholdAxis",
    "ThresholdSource",
    "build_fixed_cell_thresholds",
    "build_fixed_gene_thresholds",
    "build_global_mad_threshold_for_metric",
    "build_grouped_mad_thresholds_for_metric",
    "build_mad_cell_thresholds",
    "build_mad_thresholds_for_metric",
    "build_qc_thresholds",
    "calculate_mad_bounds",
    "format_group_value",
    "format_threshold_context",
    "normalize_group_values",
    "require_metric_columns",
    "validate_metric_table",
]
