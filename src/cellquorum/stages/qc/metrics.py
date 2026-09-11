"""QC metric calculation utilities for CellQuorum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc._types import ExpressionMatrix
from cellquorum.stages.qc.config import QCConfig, QCFeaturePatternConfig
from cellquorum.stages.qc.features import (
    CUSTOM_EXCLUDE_COLUMN,
    HEMOGLOBIN_COLUMN,
    MITO_COLUMN,
    RIBO_COLUMN,
    build_feature_masks,
    summarize_feature_masks,
)
from cellquorum.stages.qc.validation import get_qc_matrix, validate_qc_input_adata


class QCMetricsError(CellQuorumDataError):
    """Report QC metric calculation failures."""


@dataclass(frozen=True)
class QCMetricsResult:
    """Store calculated QC metrics and related metadata.

    Args:
        cell_metrics: Cell-level QC metrics indexed by observation name.
        gene_metrics: Gene-level QC metrics indexed by variable name.
        feature_masks: Boolean feature-family masks indexed by variable name.
        summary: JSON-friendly summary metrics.
        warnings: Non-fatal metric-calculation warnings.
    """

    cell_metrics: pd.DataFrame
    gene_metrics: pd.DataFrame
    feature_masks: pd.DataFrame
    summary: dict[str, object]
    warnings: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-friendly summary dictionary.

        Returns:
            Dictionary containing summary values and warnings.
        """

        payload = dict(self.summary)

        payload["warnings"] = list(self.warnings)

        return payload


def calculate_qc_metrics(
    adata: ad.AnnData,
    config: QCConfig | None = None,
) -> QCMetricsResult:
    """Calculate CellQuorum QC metrics for an AnnData object.

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

    qc_config = QCConfig() if config is None else config

    validation_summary = validate_qc_input_adata(adata, qc_config)

    matrix, matrix_source = get_qc_matrix(adata, qc_config)

    feature_names = resolve_qc_feature_names(adata, qc_config)

    if len(feature_names) != validation_summary.matrix_n_vars:
        raise QCMetricsError(
            f"Selected QC matrix source '{matrix_source}' has "
            f"{validation_summary.matrix_n_vars} variables, but {len(feature_names)} "
            "feature names were resolved."
        )

    feature_masks = build_feature_masks_for_names(
        feature_names,
        qc_config.features,
    )

    cell_metrics = calculate_cell_qc_metrics(
        matrix,
        obs_names=adata.obs_names,
        feature_masks=feature_masks,
        percent_top=qc_config.metrics.percent_top,
        log1p=qc_config.metrics.log1p,
    )

    attach_groupby_columns_from_obs(
        cell_metrics=cell_metrics,
        adata=adata,
        groupby_columns=collect_groupby_columns(qc_config),
    )

    gene_metrics = calculate_gene_qc_metrics(
        matrix,
        var_names=feature_names,
        log1p=qc_config.metrics.log1p,
    )

    summary = build_qc_metric_summary(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        feature_masks=feature_masks,
        matrix_source=matrix_source,
    )

    return QCMetricsResult(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        feature_masks=feature_masks,
        summary=summary,
        warnings=list(validation_summary.warnings),
    )


def collect_groupby_columns(config: QCConfig) -> list[str]:
    """Collect every obs column any group-wise QC rule needs, in a stable order.

    Args:
        config: QC configuration.

    Returns:
        Deduplicated grouping column names, ordered by first appearance.
    """

    requested: list[str] = []
    if config.sampleqc.enabled:
        requested.append(config.sampleqc.sample_key)
    if config.mito_mixture.enabled:
        requested.extend(config.mito_mixture.groupby)
        for grouping in config.mito_mixture.fallback_groupby:
            requested.extend(grouping)

    return list(dict.fromkeys(requested))


def attach_groupby_columns_from_obs(
    *,
    cell_metrics: pd.DataFrame,
    adata: ad.AnnData,
    groupby_columns: list[str],
) -> None:
    """Copy group-wise QC grouping columns from ``adata.obs`` onto the metric table.

    Args:
        cell_metrics: Cell-level QC metric table, indexed by observation name.
        adata: Source AnnData whose ``obs`` holds the grouping columns.
        groupby_columns: obs column names any group-wise rule groups by.

    Raises:
        QCMetricsError: If a requested groupby column is absent from ``obs``.
    """

    if not groupby_columns:
        return

    missing = [column for column in groupby_columns if column not in adata.obs.columns]
    if missing:
        raise QCMetricsError(
            "QC groupby column(s) not found in AnnData.obs: "
            f"{', '.join(missing)}. Available obs columns include: "
            f"{', '.join(map(str, adata.obs.columns[:20]))}."
        )

    for column in groupby_columns:
        if column in cell_metrics.columns:
            continue
        cell_metrics[column] = adata.obs[column].reindex(cell_metrics.index).to_numpy()


def resolve_qc_feature_names(adata: ad.AnnData, config: QCConfig) -> pd.Index:
    """Resolve feature names associated with the configured QC matrix source.

    Args:
        adata: AnnData object.
        config: QC configuration.

    Returns:
        Feature names aligned with the selected QC matrix.

    Raises:
        QCMetricsError: If raw feature names are requested but unavailable.
    """

    if config.metrics.use_raw:
        if adata.raw is None:
            raise QCMetricsError("Cannot resolve raw feature names because AnnData.raw is missing.")

        return pd.Index(adata.raw.var_names.astype(str))

    return pd.Index(adata.var_names.astype(str))


def build_feature_masks_for_names(
    feature_names: pd.Index,
    config: QCFeaturePatternConfig,
) -> pd.DataFrame:
    """Build QC feature masks for a standalone feature-name index.

    Args:
        feature_names: Feature names aligned with the selected QC matrix.
        config: Feature-pattern configuration.

    Returns:
        Feature-mask DataFrame indexed by feature name.
    """

    if len(feature_names) == 0:
        raise QCMetricsError("Cannot build QC feature masks for zero feature names.")

    empty_matrix = sp.csr_matrix((1, len(feature_names)), dtype=float)

    feature_adata = ad.AnnData(
        X=empty_matrix,
        var=pd.DataFrame(index=feature_names),
    )

    return build_feature_masks(feature_adata, config)


def calculate_cell_qc_metrics(
    matrix: ExpressionMatrix,
    *,
    obs_names: pd.Index,
    feature_masks: pd.DataFrame,
    percent_top: list[int],
    log1p: bool,
) -> pd.DataFrame:
    """Calculate cell-level QC metrics.

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

    validate_matrix_mask_alignment(matrix, feature_masks)

    total_counts = sum_axis(matrix, axis=1)

    n_genes_by_counts = count_positive_axis(matrix, axis=1)

    metrics = pd.DataFrame(index=obs_names)

    metrics["total_counts"] = total_counts

    metrics["n_genes_by_counts"] = n_genes_by_counts

    if log1p:
        metrics["log1p_total_counts"] = np.log1p(total_counts)

        metrics["log1p_n_genes_by_counts"] = np.log1p(n_genes_by_counts)

    for top_n in percent_top:
        column_name = f"pct_counts_in_top_{top_n}_genes"

        metrics[column_name] = calculate_percent_top(matrix, total_counts, top_n)

    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[MITO_COLUMN].to_numpy(dtype=bool),
        family_name="mito",
        log1p=log1p,
    )

    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[RIBO_COLUMN].to_numpy(dtype=bool),
        family_name="ribo",
        log1p=log1p,
    )

    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[HEMOGLOBIN_COLUMN].to_numpy(dtype=bool),
        family_name="hemoglobin",
        log1p=log1p,
    )

    add_feature_family_cell_metrics(
        metrics,
        matrix,
        total_counts=total_counts,
        mask=feature_masks[CUSTOM_EXCLUDE_COLUMN].to_numpy(dtype=bool),
        family_name="custom_exclude",
        log1p=log1p,
    )

    return metrics


def calculate_gene_qc_metrics(
    matrix: ExpressionMatrix,
    *,
    var_names: pd.Index,
    log1p: bool,
) -> pd.DataFrame:
    """Calculate gene-level QC metrics.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        var_names: Variable names for the metric table index.
        log1p: Whether to calculate log1p gene-level metrics.

    Returns:
        Gene-level QC metric table.

    Raises:
        QCMetricsError: If matrix dimensions and variable names do not align.
    """

    if not hasattr(matrix, "shape"):
        raise QCMetricsError("QC matrix must expose shape for gene-level metrics.")

    if int(matrix.shape[1]) != len(var_names):
        raise QCMetricsError(
            f"QC matrix has {int(matrix.shape[1])} variables, but "
            f"{len(var_names)} variable names were provided."
        )

    total_counts = sum_axis(matrix, axis=0)

    n_cells_by_counts = count_positive_axis(matrix, axis=0)

    mean_counts = total_counts / float(int(matrix.shape[0]))

    pct_dropout_by_counts = 100.0 * (1.0 - (n_cells_by_counts / float(int(matrix.shape[0]))))

    metrics = pd.DataFrame(index=var_names)

    metrics["n_cells_by_counts"] = n_cells_by_counts

    metrics["mean_counts"] = mean_counts

    metrics["pct_dropout_by_counts"] = pct_dropout_by_counts

    metrics["total_counts"] = total_counts

    if log1p:
        metrics["log1p_mean_counts"] = np.log1p(mean_counts)

        metrics["log1p_total_counts"] = np.log1p(total_counts)

    return metrics


def add_feature_family_cell_metrics(
    metrics: pd.DataFrame,
    matrix: ExpressionMatrix,
    *,
    total_counts: np.ndarray,
    mask: np.ndarray,
    family_name: str,
    log1p: bool,
) -> None:
    """Add count and percentage metrics for one feature family.

    Args:
        metrics: Cell-level metric table to update.
        matrix: Dense or sparse observation-by-variable matrix.
        total_counts: Total counts per observation.
        mask: Boolean variable mask for the feature family.
        family_name: Suffix used in metric column names.
        log1p: Whether to add log1p family-count metrics.
    """

    family_counts = sum_columns_by_mask(matrix, mask)

    metrics[f"total_counts_{family_name}"] = family_counts

    if log1p:
        metrics[f"log1p_total_counts_{family_name}"] = np.log1p(family_counts)

    metrics[f"pct_counts_{family_name}"] = safe_percent(family_counts, total_counts)


def validate_matrix_mask_alignment(matrix: ExpressionMatrix, feature_masks: pd.DataFrame) -> None:
    """Validate that feature masks align with matrix columns.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        feature_masks: Feature-mask DataFrame.

    Raises:
        QCMetricsError: If dimensions are incompatible.
    """

    if not hasattr(matrix, "shape"):
        raise QCMetricsError("QC matrix must expose shape for feature-mask alignment.")

    if int(matrix.shape[1]) != int(feature_masks.shape[0]):
        raise QCMetricsError(
            f"QC matrix has {int(matrix.shape[1])} variables, but feature masks contain "
            f"{int(feature_masks.shape[0])} rows."
        )


def sum_axis(matrix: ExpressionMatrix, *, axis: int) -> np.ndarray:
    """Sum a dense or sparse matrix over one axis.

    Args:
        matrix: Dense or sparse matrix.
        axis: Axis over which to sum.

    Returns:
        One-dimensional float array.
    """

    summed = matrix.sum(axis=axis)

    return np.asarray(summed, dtype=float).ravel()


def count_positive_axis(matrix: ExpressionMatrix, *, axis: int) -> np.ndarray:
    """Count positive values along one matrix axis.

    Args:
        matrix: Dense or sparse matrix.
        axis: Axis over which to count positive values.

    Returns:
        One-dimensional integer array.
    """

    if sp.issparse(matrix):
        counted = (matrix > 0).sum(axis=axis)

        return np.asarray(counted, dtype=int).ravel()

    dense_matrix = np.asarray(matrix)

    return np.asarray((dense_matrix > 0).sum(axis=axis), dtype=int).ravel()


def sum_columns_by_mask(matrix: ExpressionMatrix, mask: np.ndarray) -> np.ndarray:
    """Sum selected matrix columns for each observation.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        mask: Boolean variable mask.

    Returns:
        One-dimensional float array containing selected-column sums.
    """

    if int(matrix.shape[1]) != int(mask.shape[0]):
        raise QCMetricsError(
            f"Feature mask has length {int(mask.shape[0])}, but matrix has "
            f"{int(matrix.shape[1])} variables."
        )

    if not bool(mask.any()):
        return np.zeros(int(matrix.shape[0]), dtype=float)

    return sum_axis(matrix[:, mask], axis=1)


def safe_percent(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Calculate percentages while avoiding divide-by-zero errors.

    Args:
        numerator: Numerator values.
        denominator: Denominator values.

    Returns:
        Percentage values, with zero returned where denominator is zero.
    """

    numerator_float = np.asarray(numerator, dtype=float)

    denominator_float = np.asarray(denominator, dtype=float)

    output = np.zeros_like(numerator_float, dtype=float)

    nonzero_mask = denominator_float != 0.0

    output[nonzero_mask] = (numerator_float[nonzero_mask] / denominator_float[nonzero_mask]) * 100.0

    return output


def calculate_percent_top(
    matrix: ExpressionMatrix,
    total_counts: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """Calculate percent of counts contained in the top n genes per cell.

    Args:
        matrix: Dense or sparse observation-by-variable matrix.
        total_counts: Total counts per observation.
        top_n: Number of top genes to include.

    Returns:
        One-dimensional percentage array.

    Raises:
        QCMetricsError: If top_n is not positive.
    """

    if top_n <= 0:
        raise QCMetricsError(f"top_n must be > 0. Received: {top_n}.")
    top_sums = np.zeros(int(matrix.shape[0]), dtype=float)

    if sp.issparse(matrix):
        csr_matrix = cast("sp.csr_matrix", matrix).tocsr()

        for row_index in range(csr_matrix.shape[0]):
            row_values = csr_matrix.getrow(row_index).data

            top_sums[row_index] = sum_top_n_values(row_values, top_n)

        return safe_percent(top_sums, total_counts)

    dense_matrix = np.asarray(matrix, dtype=float)

    for row_index in range(dense_matrix.shape[0]):
        top_sums[row_index] = sum_top_n_values(dense_matrix[row_index], top_n)

    return safe_percent(top_sums, total_counts)


def sum_top_n_values(values: np.ndarray, top_n: int) -> float:
    """Sum the largest n values in a one-dimensional array.

    Args:
        values: One-dimensional numeric array.
        top_n: Number of largest values to sum.

    Returns:
        Sum of the largest n values.
    """

    if values.size == 0:
        return 0.0

    float_values = np.asarray(values, dtype=float)

    if top_n >= float_values.size:
        return float(np.sum(float_values))

    partitioned = np.partition(float_values, -top_n)

    return float(np.sum(partitioned[-top_n:]))


def build_qc_metric_summary(
    *,
    cell_metrics: pd.DataFrame,
    gene_metrics: pd.DataFrame,
    feature_masks: pd.DataFrame,
    matrix_source: str,
) -> dict[str, object]:
    """Build a JSON-friendly summary of calculated QC metrics.

    Args:
        cell_metrics: Cell-level QC metric table.
        gene_metrics: Gene-level QC metric table.
        feature_masks: Feature-family mask table.
        matrix_source: Matrix source label used for QC.

    Returns:
        Dictionary containing summary statistics.
    """

    feature_summary = summarize_feature_masks(feature_masks).to_dict()

    summary: dict[str, object] = {
        "matrix_source": matrix_source,
        "n_cells": int(cell_metrics.shape[0]),
        "n_genes": int(gene_metrics.shape[0]),
        "total_counts_sum": float(cell_metrics["total_counts"].sum()),
        "median_total_counts": float(cell_metrics["total_counts"].median()),
        "median_n_genes_by_counts": float(cell_metrics["n_genes_by_counts"].median()),
        "feature_masks": feature_summary,
    }

    if "pct_counts_mito" in cell_metrics.columns:
        summary["mean_pct_counts_mito"] = float(cell_metrics["pct_counts_mito"].mean())

    if "pct_counts_ribo" in cell_metrics.columns:
        summary["mean_pct_counts_ribo"] = float(cell_metrics["pct_counts_ribo"].mean())

    if "pct_counts_hemoglobin" in cell_metrics.columns:
        summary["mean_pct_counts_hemoglobin"] = float(cell_metrics["pct_counts_hemoglobin"].mean())

    return summary


__all__ = [
    "QCMetricsError",
    "QCMetricsResult",
    "add_feature_family_cell_metrics",
    "attach_groupby_columns_from_obs",
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
