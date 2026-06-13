"""QC metric calculation utilities for CellQuorum."""

from __future__ import annotations

# Import dataclass helpers for structured QC metric results.
from dataclasses import dataclass, field

# Import AnnData for runtime input validation.
import anndata as ad

# Import NumPy for dense matrix operations.
import numpy as np

# Import pandas for QC metric tables.
import pandas as pd

# Import sparse matrix utilities for large count matrices.
import scipy.sparse as sp

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import QC configuration models.
from cellquorum.qc.config import QCConfig, QCFeaturePatternConfig

# Import feature-mask utilities and column constants.
from cellquorum.qc.features import (
    CUSTOM_EXCLUDE_COLUMN,
    HEMOGLOBIN_COLUMN,
    MITO_COLUMN,
    RIBO_COLUMN,
    build_feature_masks,
    summarize_feature_masks,
)

# Import QC input validation and matrix resolution utilities.
from cellquorum.qc.validation import get_qc_matrix, validate_qc_input_adata


class QCMetricsError(CellQuorumDataError):
    """
    Report QC metric calculation failures.

    QC metrics are used by thresholding, filtering decisions, reports, and
    provenance. Metric errors should fail clearly because incorrect metric tables
    would produce misleading QC decisions downstream.
    """


@dataclass(frozen=True)
class QCMetricsResult:
    """
    Store calculated QC metrics and related metadata.

    This result object is the bridge between feature annotation and thresholding.
    It stores cell-level QC metrics, gene-level QC metrics, feature-family masks,
    a structured summary, and non-fatal warnings.

    Args:
        cell_metrics: Cell-level QC metrics indexed by observation name.
        gene_metrics: Gene-level QC metrics indexed by variable name.
        feature_masks: Boolean feature-family masks indexed by variable name.
        summary: JSON-friendly summary metrics.
        warnings: Non-fatal metric-calculation warnings.
    """

    # Store cell-level QC metrics.
    cell_metrics: pd.DataFrame

    # Store gene-level QC metrics.
    gene_metrics: pd.DataFrame

    # Store feature-family masks.
    feature_masks: pd.DataFrame

    # Store JSON-friendly summary values.
    summary: dict[str, object]

    # Store non-fatal warnings.
    warnings: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, object]:
        """
        Return a JSON-friendly summary dictionary.

        Returns:
            Dictionary containing summary values and warnings.
        """

        # Create a shallow copy so callers cannot mutate the stored summary.
        payload = dict(self.summary)

        # Add warnings as a JSON-friendly list.
        payload["warnings"] = list(self.warnings)

        # Return the summary payload.
        return payload


def calculate_qc_metrics(
    adata: ad.AnnData,
    config: QCConfig | None = None,
) -> QCMetricsResult:
    """
    Calculate CellQuorum QC metrics for an AnnData object.

    This function calculates Scanpy-style cell and gene QC metrics using the
    configured matrix source, percent-top settings, log1p behavior, and QC
    feature-family definitions. It supports dense and scipy sparse matrices.

    Args:
        adata: AnnData object containing count-like data.
        config: Optional QC configuration. Defaults to QCConfig().

    Returns:
        QCMetricsResult containing cell metrics, gene metrics, feature masks,
        summary values, and warnings.

    Raises:
        QCMetricsError: If metric calculation fails or matrix dimensions do not
            align with feature names.
    """

    # Resolve the QC configuration.
    qc_config = QCConfig() if config is None else config

    # Validate the AnnData input before metric calculation.
    validation_summary = validate_qc_input_adata(adata, qc_config)

    # Resolve the configured QC matrix.
    matrix, matrix_source = get_qc_matrix(adata, qc_config)

    # Resolve the feature names associated with the selected matrix.
    feature_names = resolve_qc_feature_names(adata, qc_config)

    # Validate that feature names align with the selected matrix.
    if len(feature_names) != validation_summary.matrix_n_vars:
        raise QCMetricsError(
            f"Selected QC matrix source '{matrix_source}' has "
            f"{validation_summary.matrix_n_vars} variables, but {len(feature_names)} "
            "feature names were resolved."
        )

    # Build feature-family masks for the selected feature names.
    feature_masks = build_feature_masks_for_names(
        feature_names,
        qc_config.features,
    )

    # Calculate cell-level QC metrics.
    cell_metrics = calculate_cell_qc_metrics(
        matrix,
        obs_names=adata.obs_names,
        feature_masks=feature_masks,
        percent_top=qc_config.metrics.percent_top,
        log1p=qc_config.metrics.log1p,
    )

    # Calculate gene-level QC metrics.
    gene_metrics = calculate_gene_qc_metrics(
        matrix,
        var_names=feature_names,
        log1p=qc_config.metrics.log1p,
    )

    # Build a structured metric summary.
    summary = build_qc_metric_summary(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        feature_masks=feature_masks,
        matrix_source=matrix_source,
    )

    # Return the structured QC metrics result.
    return QCMetricsResult(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        feature_masks=feature_masks,
        summary=summary,
        warnings=list(validation_summary.warnings),
    )


def resolve_qc_feature_names(adata: ad.AnnData, config: QCConfig) -> pd.Index:
    """
    Resolve feature names associated with the configured QC matrix source.

    Args:
        adata: AnnData object.
        config: QC configuration.

    Returns:
        Feature names aligned with the selected QC matrix.

    Raises:
        QCMetricsError: If raw feature names are requested but unavailable.
    """

    # Use AnnData.raw.var_names when raw.X is selected.
    if config.metrics.use_raw:
        # Validate that AnnData.raw exists.
        if adata.raw is None:
            raise QCMetricsError("Cannot resolve raw feature names because AnnData.raw is missing.")

        # Return raw variable names.
        return pd.Index(adata.raw.var_names.astype(str))

    # Use AnnData.var_names for AnnData.X and AnnData.layers.
    return pd.Index(adata.var_names.astype(str))


def build_feature_masks_for_names(
    feature_names: pd.Index,
    config: QCFeaturePatternConfig,
) -> pd.DataFrame:
    """
    Build QC feature masks for a standalone feature-name index.

    This helper keeps raw.X support possible because AnnData.raw may have feature
    names that differ from AnnData.var_names. It constructs a minimal sparse
    AnnData object only for feature-mask generation, without allocating a real
    dense expression matrix.

    Args:
        feature_names: Feature names aligned with the selected QC matrix.
        config: Feature-pattern configuration.

    Returns:
        Feature-mask DataFrame indexed by feature name.
    """

    # Validate that at least one feature name is available.
    if len(feature_names) == 0:
        raise QCMetricsError("Cannot build QC feature masks for zero feature names.")

    # Build a minimal zero sparse matrix with one observation and n variables.
    empty_matrix = sp.csr_matrix((1, len(feature_names)), dtype=float)

    # Build a minimal AnnData object carrying only feature names.
    feature_adata = ad.AnnData(
        X=empty_matrix,
        var=pd.DataFrame(index=feature_names),
    )

    # Build and return feature masks using the feature annotation layer.
    return build_feature_masks(feature_adata, config)


def calculate_cell_qc_metrics(
    matrix: object,
    *,
    obs_names: pd.Index,
    feature_masks: pd.DataFrame,
    percent_top: list[int],
    log1p: bool,
) -> pd.DataFrame:
    """
    Calculate cell-level QC metrics.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        obs_names: Observation names for the metric table index.
        feature_masks: Feature-family masks aligned to matrix columns.
        percent_top: Top-n gene ranks for cumulative count percentages.
        log1p: Whether to calculate log1p metrics.

    Returns:
        Cell-level QC metric table.

    Raises:
        QCMetricsError: If matrix dimensions and masks do not align.
    """

    # Validate matrix and feature-mask alignment.
    validate_matrix_mask_alignment(matrix, feature_masks)

    # Calculate total counts per observation.
    total_counts = sum_axis(matrix, axis=1)

    # Calculate number of detected genes per observation.
    n_genes_by_counts = count_positive_axis(matrix, axis=1)

    # Initialize the cell metrics table.
    metrics = pd.DataFrame(index=obs_names)

    # Store total counts.
    metrics["total_counts"] = total_counts

    # Store detected genes by counts.
    metrics["n_genes_by_counts"] = n_genes_by_counts

    # Add log1p-transformed base metrics when requested.
    if log1p:
        # Store log1p total counts.
        metrics["log1p_total_counts"] = np.log1p(total_counts)

        # Store log1p detected-gene counts.
        metrics["log1p_n_genes_by_counts"] = np.log1p(n_genes_by_counts)

    # Add percent-top metrics.
    for top_n in percent_top:
        # Build the metric column name.
        column_name = f"pct_counts_in_top_{top_n}_genes"

        # Calculate and store the percent-top metric.
        metrics[column_name] = calculate_percent_top(matrix, total_counts, top_n)

    # Add mitochondrial feature-family metrics.
    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[MITO_COLUMN].to_numpy(dtype=bool),
        family_name="mito",
        log1p=log1p,
    )

    # Add ribosomal feature-family metrics.
    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[RIBO_COLUMN].to_numpy(dtype=bool),
        family_name="ribo",
        log1p=log1p,
    )

    # Add hemoglobin feature-family metrics.
    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[HEMOGLOBIN_COLUMN].to_numpy(dtype=bool),
        family_name="hemoglobin",
        log1p=log1p,
    )

    # Add custom-exclude feature-family metrics.
    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[CUSTOM_EXCLUDE_COLUMN].to_numpy(dtype=bool),
        family_name="custom_exclude",
        log1p=log1p,
    )

    # Return the cell-level QC metrics.
    return metrics


def calculate_gene_qc_metrics(
    matrix: object,
    *,
    var_names: pd.Index,
    log1p: bool,
) -> pd.DataFrame:
    """
    Calculate gene-level QC metrics.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        var_names: Variable names for the metric table index.
        log1p: Whether to calculate log1p gene-level metrics.

    Returns:
        Gene-level QC metric table.

    Raises:
        QCMetricsError: If matrix dimensions and variable names do not align.
    """

    # Validate that the matrix exposes a shape.
    if not hasattr(matrix, "shape"):
        raise QCMetricsError("QC matrix must expose shape for gene-level metrics.")

    # Validate the variable-name count.
    if int(matrix.shape[1]) != len(var_names):
        raise QCMetricsError(
            f"QC matrix has {int(matrix.shape[1])} variables, but "
            f"{len(var_names)} variable names were provided."
        )

    # Calculate total counts per variable.
    total_counts = sum_axis(matrix, axis=0)

    # Calculate number of cells with positive counts per variable.
    n_cells_by_counts = count_positive_axis(matrix, axis=0)

    # Calculate mean counts per variable.
    mean_counts = total_counts / float(int(matrix.shape[0]))

    # Calculate percent dropout per variable.
    pct_dropout_by_counts = 100.0 * (1.0 - (n_cells_by_counts / float(int(matrix.shape[0]))))

    # Initialize the gene metrics table.
    metrics = pd.DataFrame(index=var_names)

    # Store cell-detection counts.
    metrics["n_cells_by_counts"] = n_cells_by_counts

    # Store mean counts.
    metrics["mean_counts"] = mean_counts

    # Store percent dropout.
    metrics["pct_dropout_by_counts"] = pct_dropout_by_counts

    # Store total counts.
    metrics["total_counts"] = total_counts

    # Add log1p gene-level metrics when requested.
    if log1p:
        # Store log1p mean counts.
        metrics["log1p_mean_counts"] = np.log1p(mean_counts)

        # Store log1p total counts.
        metrics["log1p_total_counts"] = np.log1p(total_counts)

    # Return the gene-level QC metrics.
    return metrics


def add_feature_family_cell_metrics(
    metrics: pd.DataFrame,
    matrix: object,
    *,
    total_counts: np.ndarray,
    mask: np.ndarray,
    family_name: str,
    log1p: bool,
) -> None:
    """
    Add count and percentage metrics for one feature family.

    Args:
        metrics: Cell-level metric table to update.
        matrix: Dense or sparse observation-by-variable matrix.
        total_counts: Total counts per observation.
        mask: Boolean variable mask for the feature family.
        family_name: Suffix used in metric column names.
        log1p: Whether to add log1p family-count metrics.
    """

    # Calculate total counts from this feature family per cell.
    family_counts = sum_columns_by_mask(matrix, mask)

    # Store family-specific total counts.
    metrics[f"total_counts_{family_name}"] = family_counts

    # Add log1p family counts when requested.
    if log1p:
        # Store log1p family-specific total counts.
        metrics[f"log1p_total_counts_{family_name}"] = np.log1p(family_counts)

    # Store family-specific percentage of total counts.
    metrics[f"pct_counts_{family_name}"] = safe_percent(family_counts, total_counts)


def validate_matrix_mask_alignment(matrix: object, feature_masks: pd.DataFrame) -> None:
    """
    Validate that feature masks align with matrix columns.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        feature_masks: Feature-mask DataFrame.

    Raises:
        QCMetricsError: If dimensions are incompatible.
    """

    # Validate that the matrix exposes shape.
    if not hasattr(matrix, "shape"):
        raise QCMetricsError("QC matrix must expose shape for feature-mask alignment.")

    # Validate that matrix variables match feature-mask rows.
    if int(matrix.shape[1]) != int(feature_masks.shape[0]):
        raise QCMetricsError(
            f"QC matrix has {int(matrix.shape[1])} variables, but feature masks contain "
            f"{int(feature_masks.shape[0])} rows."
        )


def sum_axis(matrix: object, *, axis: int) -> np.ndarray:
    """
    Sum a dense or sparse matrix over one axis.

    Args:
        matrix: Dense or sparse matrix.
        axis: Axis over which to sum.

    Returns:
        One-dimensional float array.
    """

    # Calculate sums using sparse-aware matrix methods.
    summed = matrix.sum(axis=axis)

    # Convert the result into a one-dimensional float array.
    return np.asarray(summed, dtype=float).ravel()


def count_positive_axis(matrix: object, *, axis: int) -> np.ndarray:
    """
    Count positive values along one matrix axis.

    Args:
        matrix: Dense or sparse matrix.
        axis: Axis over which to count positive values.

    Returns:
        One-dimensional integer array.
    """

    # Build a positive-value mask for sparse matrices.
    if sp.issparse(matrix):
        # Count positive sparse entries along the requested axis.
        counted = (matrix > 0).sum(axis=axis)

        # Return the result as a one-dimensional integer array.
        return np.asarray(counted, dtype=int).ravel()

    # Convert dense matrix-like input to an array.
    dense_matrix = np.asarray(matrix)

    # Count positive dense entries along the requested axis.
    return np.asarray((dense_matrix > 0).sum(axis=axis), dtype=int).ravel()


def sum_columns_by_mask(matrix: object, mask: np.ndarray) -> np.ndarray:
    """
    Sum selected matrix columns for each observation.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        mask: Boolean variable mask.

    Returns:
        One-dimensional float array containing selected-column sums.
    """

    # Validate that the mask length matches matrix variables.
    if int(matrix.shape[1]) != int(mask.shape[0]):
        raise QCMetricsError(
            f"Feature mask has length {int(mask.shape[0])}, but matrix has "
            f"{int(matrix.shape[1])} variables."
        )

    # Return zeros when no variables are selected.
    if not bool(mask.any()):
        return np.zeros(int(matrix.shape[0]), dtype=float)

    # Subset selected columns and sum across variables.
    return sum_axis(matrix[:, mask], axis=1)


def safe_percent(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Calculate percentages while avoiding divide-by-zero errors.

    Args:
        numerator: Numerator values.
        denominator: Denominator values.

    Returns:
        Percentage values, with zero returned where denominator is zero.
    """

    # Convert numerator to float array.
    numerator_float = np.asarray(numerator, dtype=float)

    # Convert denominator to float array.
    denominator_float = np.asarray(denominator, dtype=float)

    # Initialize the output array with zeros.
    output = np.zeros_like(numerator_float, dtype=float)

    # Identify non-zero denominator positions.
    nonzero_mask = denominator_float != 0.0

    # Calculate percentages only where denominators are non-zero.
    output[nonzero_mask] = (numerator_float[nonzero_mask] / denominator_float[nonzero_mask]) * 100.0

    # Return the percentage array.
    return output


def calculate_percent_top(
    matrix: object,
    total_counts: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """
    Calculate percent of counts contained in the top n genes per cell.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        total_counts: Total counts per observation.
        top_n: Number of top genes to include.

    Returns:
        One-dimensional percentage array.

    Raises:
        QCMetricsError: If top_n is not positive.
    """

    # Reject invalid top_n values defensively.
    if top_n <= 0:
        raise QCMetricsError(f"top_n must be > 0. Received: {top_n}.")

    # Initialize top-count sums.
    top_sums = np.zeros(int(matrix.shape[0]), dtype=float)

    # Use sparse row iteration for sparse matrices.
    if sp.issparse(matrix):
        # Convert the sparse matrix to CSR for efficient row access.
        csr_matrix = matrix.tocsr()

        # Iterate over rows.
        for row_index in range(csr_matrix.shape[0]):
            # Extract non-zero stored values for this row.
            row_values = csr_matrix.getrow(row_index).data

            # Store the top-n sum for this sparse row.
            top_sums[row_index] = sum_top_n_values(row_values, top_n)

        # Return safe percentages.
        return safe_percent(top_sums, total_counts)

    # Convert dense matrix-like input to a NumPy array.
    dense_matrix = np.asarray(matrix, dtype=float)

    # Iterate over dense rows.
    for row_index in range(dense_matrix.shape[0]):
        # Store the top-n sum for this dense row.
        top_sums[row_index] = sum_top_n_values(dense_matrix[row_index], top_n)

    # Return safe percentages.
    return safe_percent(top_sums, total_counts)


def sum_top_n_values(values: np.ndarray, top_n: int) -> float:
    """
    Sum the largest n values in a one-dimensional array.

    Args:
        values: One-dimensional numeric array.
        top_n: Number of largest values to sum.

    Returns:
        Sum of the largest n values.
    """

    # Return zero for empty rows.
    if values.size == 0:
        return 0.0

    # Convert values to a float array.
    float_values = np.asarray(values, dtype=float)

    # Use all values when top_n is greater than or equal to row length.
    if top_n >= float_values.size:
        return float(np.sum(float_values))

    # Partition values so the largest top_n values are in the last positions.
    partitioned = np.partition(float_values, -top_n)

    # Sum and return the largest top_n values.
    return float(np.sum(partitioned[-top_n:]))


def build_qc_metric_summary(
    *,
    cell_metrics: pd.DataFrame,
    gene_metrics: pd.DataFrame,
    feature_masks: pd.DataFrame,
    matrix_source: str,
) -> dict[str, object]:
    """
    Build a JSON-friendly summary of calculated QC metrics.

    Args:
        cell_metrics: Cell-level QC metric table.
        gene_metrics: Gene-level QC metric table.
        feature_masks: Feature-family mask table.
        matrix_source: Matrix source label used for QC.

    Returns:
        Dictionary containing summary statistics.
    """

    # Summarize feature masks.
    feature_summary = summarize_feature_masks(feature_masks).to_dict()

    # Build the summary dictionary.
    summary: dict[str, object] = {
        "matrix_source": matrix_source,
        "n_cells": int(cell_metrics.shape[0]),
        "n_genes": int(gene_metrics.shape[0]),
        "total_counts_sum": float(cell_metrics["total_counts"].sum()),
        "median_total_counts": float(cell_metrics["total_counts"].median()),
        "median_n_genes_by_counts": float(cell_metrics["n_genes_by_counts"].median()),
        "feature_masks": feature_summary,
    }

    # Add mitochondrial percentage summary when available.
    if "pct_counts_mito" in cell_metrics.columns:
        # Store mean mitochondrial percentage.
        summary["mean_pct_counts_mito"] = float(cell_metrics["pct_counts_mito"].mean())

    # Add ribosomal percentage summary when available.
    if "pct_counts_ribo" in cell_metrics.columns:
        # Store mean ribosomal percentage.
        summary["mean_pct_counts_ribo"] = float(cell_metrics["pct_counts_ribo"].mean())

    # Add hemoglobin percentage summary when available.
    if "pct_counts_hemoglobin" in cell_metrics.columns:
        # Store mean hemoglobin percentage.
        summary["mean_pct_counts_hemoglobin"] = float(cell_metrics["pct_counts_hemoglobin"].mean())

    # Return the summary dictionary.
    return summary


__all__ = [
    "QCMetricsError",
    "QCMetricsResult",
    "add_feature_family_cell_metrics",
    "build_feature_masks_for_names",
    "build_qc_metric_summary",
    "calculate_cell_qc_metrics",
    "calculate_gene_qc_metrics",
    "calculate_percent_top",
    "calculate_qc_metrics",
    "count_positive_axis",
    "resolve_qc_feature_names",
    "safe_percent",
    "sum_axis",
    "sum_columns_by_mask",
    "sum_top_n_values",
    "validate_matrix_mask_alignment",
]
