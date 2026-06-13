"""Tests for CellQuorum QC metric calculation utilities."""

from __future__ import annotations

# Import AnnData for constructing small metric test objects.
import anndata as ad

# Import NumPy for deterministic numeric assertions.
import numpy as np

# Import pandas for AnnData metadata and DataFrame assertions.
import pandas as pd

# Import pandas testing helpers for dense/sparse equality checks.
import pandas.testing as pdt

# Import pytest for exception assertions.
import pytest

# Import scipy sparse matrices for sparse QC metric tests.
import scipy.sparse as sp

# Import QC configuration models used by metric calculation.
from cellquorum.qc.config import QCConfig

# Import feature-mask column constants used in low-level metric tests.
from cellquorum.qc.features import (
    CUSTOM_EXCLUDE_COLUMN,
    HEMOGLOBIN_COLUMN,
    MITO_COLUMN,
    RIBO_COLUMN,
    build_feature_masks,
)

# Import QC metric utilities under test.
from cellquorum.qc.metrics import (
    QCMetricsError,
    QCMetricsResult,
    build_feature_masks_for_names,
    calculate_cell_qc_metrics,
    calculate_gene_qc_metrics,
    calculate_percent_top,
    calculate_qc_metrics,
    count_positive_axis,
    resolve_qc_feature_names,
    safe_percent,
    sum_axis,
    sum_columns_by_mask,
    sum_top_n_values,
    validate_matrix_mask_alignment,
)


def make_metrics_test_adata(*, sparse: bool = False) -> ad.AnnData:
    """
    Build a small AnnData object with hand-checkable QC metrics.

    The feature names intentionally include mitochondrial, ribosomal, hemoglobin,
    housekeeping, and custom-exclude genes. The matrix includes a zero-count cell
    so percentage calculations must handle zero denominators safely.

    Args:
        sparse: Whether to return the expression matrix as sparse CSR.

    Returns:
        Small AnnData object for QC metric tests.
    """

    # Build a deterministic cell-by-gene count matrix.
    matrix = np.array(
        [
            [5, 1, 1, 0, 0, 3, 0],
            [0, 0, 0, 2, 1, 2, 0],
            [1, 0, 0, 0, 0, 0, 9],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    # Convert the matrix to sparse CSR when requested.
    expression = sp.csr_matrix(matrix) if sparse else matrix

    # Define observation names.
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3", "cell_4"])

    # Define feature names with QC-relevant biology.
    var = pd.DataFrame(
        index=[
            "MT-ND1",
            "RPS3",
            "RPLP0",
            "HBA1",
            "HBB",
            "ACTB",
            "MALAT1",
        ]
    )

    # Return the AnnData object.
    return ad.AnnData(X=expression, obs=obs, var=var)


def make_metrics_config() -> QCConfig:
    """
    Build the QC config used for hand-checkable metric tests.

    The config uses percent_top=[2] so the top-gene percentage calculation is
    meaningful on the tiny seven-gene test matrix. It also flags MALAT1 through
    a custom-exclude prefix.

    Returns:
        QCConfig configured for deterministic QC metric tests.
    """

    # Return the deterministic QC metric test configuration.
    return QCConfig(
        metrics={
            "percent_top": [2],
            "log1p": True,
        },
        features={
            "custom_exclude_prefixes": ["MALAT"],
        },
    )


def test_calculate_qc_metrics_returns_expected_cell_metrics_dense() -> None:
    """
    Verify dense AnnData cell-level QC metrics are numerically correct.

    This test checks total counts, detected genes, percent-top counts, log1p
    metrics, mitochondrial percentage, ribosomal percentage, hemoglobin
    percentage, and custom-exclude percentage.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Build the deterministic QC configuration.
    config = make_metrics_config()

    # Calculate QC metrics.
    result = calculate_qc_metrics(adata, config)

    # Extract the cell-level metric table.
    cell_metrics = result.cell_metrics

    # Confirm the cell metric table index follows observation names.
    assert list(cell_metrics.index) == ["cell_1", "cell_2", "cell_3", "cell_4"]

    # Confirm total counts per cell.
    np.testing.assert_allclose(cell_metrics["total_counts"], [10.0, 5.0, 10.0, 0.0])

    # Confirm detected genes per cell.
    np.testing.assert_array_equal(cell_metrics["n_genes_by_counts"], [4, 3, 2, 0])

    # Confirm log1p total counts.
    np.testing.assert_allclose(
        cell_metrics["log1p_total_counts"],
        np.log1p([10.0, 5.0, 10.0, 0.0]),
    )

    # Confirm log1p detected genes.
    np.testing.assert_allclose(
        cell_metrics["log1p_n_genes_by_counts"],
        np.log1p([4, 3, 2, 0]),
    )

    # Confirm percent of counts in the top two genes.
    np.testing.assert_allclose(
        cell_metrics["pct_counts_in_top_2_genes"],
        [80.0, 80.0, 100.0, 0.0],
    )

    # Confirm mitochondrial total counts per cell.
    np.testing.assert_allclose(cell_metrics["total_counts_mito"], [5.0, 0.0, 1.0, 0.0])

    # Confirm mitochondrial percentage per cell.
    np.testing.assert_allclose(cell_metrics["pct_counts_mito"], [50.0, 0.0, 10.0, 0.0])

    # Confirm ribosomal total counts per cell.
    np.testing.assert_allclose(cell_metrics["total_counts_ribo"], [2.0, 0.0, 0.0, 0.0])

    # Confirm ribosomal percentage per cell.
    np.testing.assert_allclose(cell_metrics["pct_counts_ribo"], [20.0, 0.0, 0.0, 0.0])

    # Confirm hemoglobin total counts per cell.
    np.testing.assert_allclose(
        cell_metrics["total_counts_hemoglobin"],
        [0.0, 3.0, 0.0, 0.0],
    )

    # Confirm hemoglobin percentage per cell.
    np.testing.assert_allclose(
        cell_metrics["pct_counts_hemoglobin"],
        [0.0, 60.0, 0.0, 0.0],
    )

    # Confirm custom-exclude total counts per cell.
    np.testing.assert_allclose(
        cell_metrics["total_counts_custom_exclude"],
        [0.0, 0.0, 9.0, 0.0],
    )

    # Confirm custom-exclude percentage per cell.
    np.testing.assert_allclose(
        cell_metrics["pct_counts_custom_exclude"],
        [0.0, 0.0, 90.0, 0.0],
    )


def test_calculate_qc_metrics_returns_expected_gene_metrics_dense() -> None:
    """
    Verify dense AnnData gene-level QC metrics are numerically correct.

    Gene-level metrics should count cells with positive expression, total counts,
    mean counts, dropout percentage, and optional log1p-transformed values.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Build the deterministic QC configuration.
    config = make_metrics_config()

    # Calculate QC metrics.
    result = calculate_qc_metrics(adata, config)

    # Extract the gene-level metric table.
    gene_metrics = result.gene_metrics

    # Confirm the gene metric table index follows variable names.
    assert list(gene_metrics.index) == [
        "MT-ND1",
        "RPS3",
        "RPLP0",
        "HBA1",
        "HBB",
        "ACTB",
        "MALAT1",
    ]

    # Confirm number of cells with positive counts per gene.
    np.testing.assert_array_equal(
        gene_metrics["n_cells_by_counts"],
        [2, 1, 1, 1, 1, 2, 1],
    )

    # Confirm total counts per gene.
    np.testing.assert_allclose(
        gene_metrics["total_counts"],
        [6.0, 1.0, 1.0, 2.0, 1.0, 5.0, 9.0],
    )

    # Confirm mean counts per gene.
    np.testing.assert_allclose(
        gene_metrics["mean_counts"],
        [1.5, 0.25, 0.25, 0.5, 0.25, 1.25, 2.25],
    )

    # Confirm dropout percentage per gene.
    np.testing.assert_allclose(
        gene_metrics["pct_dropout_by_counts"],
        [50.0, 75.0, 75.0, 75.0, 75.0, 50.0, 75.0],
    )

    # Confirm log1p mean counts.
    np.testing.assert_allclose(
        gene_metrics["log1p_mean_counts"],
        np.log1p([1.5, 0.25, 0.25, 0.5, 0.25, 1.25, 2.25]),
    )

    # Confirm log1p total counts.
    np.testing.assert_allclose(
        gene_metrics["log1p_total_counts"],
        np.log1p([6.0, 1.0, 1.0, 2.0, 1.0, 5.0, 9.0]),
    )


def test_calculate_qc_metrics_returns_expected_feature_masks_and_summary() -> None:
    """
    Verify calculated QC metrics include feature masks and summary payloads.

    The summary should be JSON-friendly and contain matrix source, dimensions,
    count summaries, and feature-mask counts.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Build the deterministic QC configuration.
    config = make_metrics_config()

    # Calculate QC metrics.
    result = calculate_qc_metrics(adata, config)

    # Confirm a structured result object was returned.
    assert isinstance(result, QCMetricsResult)

    # Confirm feature-mask dimensions.
    assert result.feature_masks.shape == (7, 4)

    # Confirm feature-mask counts.
    assert result.summary["feature_masks"] == {
        "n_vars": 7,
        "n_mito": 1,
        "n_ribo": 2,
        "n_hemoglobin": 2,
        "n_custom_exclude": 1,
    }

    # Confirm the matrix source is AnnData.X.
    assert result.summary["matrix_source"] == "X"

    # Confirm the cell count summary.
    assert result.summary["n_cells"] == 4

    # Confirm the gene count summary.
    assert result.summary["n_genes"] == 7

    # Confirm the total count sum.
    assert result.summary["total_counts_sum"] == 25.0

    # Confirm median total counts.
    assert result.summary["median_total_counts"] == 7.5

    # Confirm median detected genes.
    assert result.summary["median_n_genes_by_counts"] == 2.5

    # Confirm mean mitochondrial percentage.
    assert result.summary["mean_pct_counts_mito"] == 15.0

    # Confirm warnings are available as a list.
    assert result.warnings == []


def test_qc_metrics_result_summary_dict_includes_warnings() -> None:
    """
    Verify QCMetricsResult summary serialization includes warnings.

    The summary dictionary will later be written into stage metrics and
    provenance.
    """

    # Build a minimal QC metrics result.
    result = QCMetricsResult(
        cell_metrics=pd.DataFrame(),
        gene_metrics=pd.DataFrame(),
        feature_masks=pd.DataFrame(),
        summary={"n_cells": 0},
        warnings=["example warning"],
    )

    # Convert the summary to a dictionary.
    payload = result.to_summary_dict()

    # Confirm summary fields are preserved.
    assert payload["n_cells"] == 0

    # Confirm warnings are included.
    assert payload["warnings"] == ["example warning"]

    # Mutate the returned payload.
    payload["warnings"].append("mutated")

    # Confirm the original warning list is not mutated.
    assert result.warnings == ["example warning"]


def test_calculate_qc_metrics_sparse_matches_dense() -> None:
    """
    Verify sparse and dense matrices produce identical QC metrics.

    Real scRNA-seq matrices are usually sparse, so sparse behavior must match the
    hand-checked dense implementation.
    """

    # Build the dense test AnnData object.
    dense_adata = make_metrics_test_adata(sparse=False)

    # Build the sparse test AnnData object.
    sparse_adata = make_metrics_test_adata(sparse=True)

    # Build the deterministic QC configuration.
    config = make_metrics_config()

    # Calculate dense QC metrics.
    dense_result = calculate_qc_metrics(dense_adata, config)

    # Calculate sparse QC metrics.
    sparse_result = calculate_qc_metrics(sparse_adata, config)

    # Confirm cell-level metrics match exactly.
    pdt.assert_frame_equal(dense_result.cell_metrics, sparse_result.cell_metrics)

    # Confirm gene-level metrics match exactly.
    pdt.assert_frame_equal(dense_result.gene_metrics, sparse_result.gene_metrics)

    # Confirm feature masks match exactly.
    pdt.assert_frame_equal(dense_result.feature_masks, sparse_result.feature_masks)


def test_calculate_qc_metrics_uses_configured_layer() -> None:
    """
    Verify QC metrics can be calculated from a configured AnnData layer.

    This supports workflows where raw counts are stored in a layer such as
    `counts` while AnnData.X contains another representation.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Store doubled counts in a named layer.
    adata.layers["counts"] = adata.X.copy() * 2.0

    # Build a QC configuration that uses the counts layer.
    config = QCConfig(
        metrics={
            "layer": "counts",
            "percent_top": [2],
        },
        features={
            "custom_exclude_prefixes": ["MALAT"],
        },
    )

    # Calculate QC metrics.
    result = calculate_qc_metrics(adata, config)

    # Confirm the matrix source points to the configured layer.
    assert result.summary["matrix_source"] == "layers[counts]"

    # Confirm total counts were calculated from the layer.
    np.testing.assert_allclose(result.cell_metrics["total_counts"], [20.0, 10.0, 20.0, 0.0])


def test_calculate_qc_metrics_uses_raw_matrix_when_requested() -> None:
    """
    Verify QC metrics can be calculated from AnnData.raw.X.

    This supports workflows where raw counts are stored in AnnData.raw and X
    contains transformed values.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Store the original counts in AnnData.raw.
    adata.raw = adata.copy()

    # Replace AnnData.X with zeros to prove raw.X is used.
    adata.X = np.zeros_like(adata.X)

    # Build a QC configuration that uses raw.X.
    config = QCConfig(
        metrics={
            "use_raw": True,
            "percent_top": [2],
        },
        features={
            "custom_exclude_prefixes": ["MALAT"],
        },
    )

    # Calculate QC metrics.
    result = calculate_qc_metrics(adata, config)

    # Confirm the matrix source points to raw.X.
    assert result.summary["matrix_source"] == "raw.X"

    # Confirm total counts came from raw.X rather than the zeroed AnnData.X.
    np.testing.assert_allclose(result.cell_metrics["total_counts"], [10.0, 5.0, 10.0, 0.0])


def test_calculate_qc_metrics_omits_log1p_columns_when_disabled() -> None:
    """
    Verify log1p QC metrics can be disabled.

    Some downstream workflows may want only raw count-derived QC metrics.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Build a QC configuration with log1p metrics disabled.
    config = QCConfig(
        metrics={
            "percent_top": [2],
            "log1p": False,
        },
        features={
            "custom_exclude_prefixes": ["MALAT"],
        },
    )

    # Calculate QC metrics.
    result = calculate_qc_metrics(adata, config)

    # Confirm cell-level log1p columns are absent.
    assert "log1p_total_counts" not in result.cell_metrics.columns
    assert "log1p_n_genes_by_counts" not in result.cell_metrics.columns

    # Confirm feature-family log1p columns are absent.
    assert "log1p_total_counts_mito" not in result.cell_metrics.columns

    # Confirm gene-level log1p columns are absent.
    assert "log1p_mean_counts" not in result.gene_metrics.columns
    assert "log1p_total_counts" not in result.gene_metrics.columns


def test_resolve_qc_feature_names_returns_var_names_by_default() -> None:
    """
    Verify feature names resolve from AnnData.var_names by default.

    AnnData.X and AnnData.layers should both use AnnData.var_names.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Resolve feature names with the default config.
    feature_names = resolve_qc_feature_names(adata, QCConfig())

    # Confirm feature names match AnnData.var_names.
    assert list(feature_names) == list(adata.var_names)


def test_resolve_qc_feature_names_returns_raw_var_names_when_requested() -> None:
    """
    Verify feature names resolve from AnnData.raw.var_names when raw.X is used.

    Raw feature names may differ from AnnData.var_names after subsetting.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Store the original object in AnnData.raw.
    adata.raw = adata.copy()

    # Resolve raw feature names.
    feature_names = resolve_qc_feature_names(adata, QCConfig(metrics={"use_raw": True}))

    # Confirm raw feature names were returned.
    assert list(feature_names) == list(adata.raw.var_names)


def test_resolve_qc_feature_names_rejects_missing_raw() -> None:
    """
    Verify raw feature-name resolution fails when AnnData.raw is missing.

    This keeps feature-name resolution consistent with matrix-source validation.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Confirm missing raw feature names fail clearly.
    with pytest.raises(QCMetricsError, match="AnnData.raw is missing"):
        resolve_qc_feature_names(adata, QCConfig(metrics={"use_raw": True}))


def test_build_feature_masks_for_names_matches_expected_families() -> None:
    """
    Verify feature masks can be built from standalone feature names.

    This helper is needed for raw.X support, where raw feature names may differ
    from AnnData.var_names.
    """

    # Build standalone feature names.
    feature_names = pd.Index(["MT-CO1", "RPS3", "HBA1", "ACTB", "MALAT1"])

    # Build feature masks.
    masks = build_feature_masks_for_names(
        feature_names,
        QCConfig(features={"custom_exclude_prefixes": ["MALAT"]}).features,
    )

    # Confirm mitochondrial matching.
    assert bool(masks.loc["MT-CO1", MITO_COLUMN]) is True

    # Confirm ribosomal matching.
    assert bool(masks.loc["RPS3", RIBO_COLUMN]) is True

    # Confirm hemoglobin matching.
    assert bool(masks.loc["HBA1", HEMOGLOBIN_COLUMN]) is True

    # Confirm custom-exclude matching.
    assert bool(masks.loc["MALAT1", CUSTOM_EXCLUDE_COLUMN]) is True


def test_build_feature_masks_for_names_rejects_empty_index() -> None:
    """
    Verify standalone feature-mask construction rejects zero feature names.

    Metric calculation cannot proceed without feature names.
    """

    # Confirm empty feature-name indices fail clearly.
    with pytest.raises(QCMetricsError, match="zero feature names"):
        build_feature_masks_for_names(pd.Index([]), QCConfig().features)


def test_calculate_cell_qc_metrics_accepts_direct_inputs() -> None:
    """
    Verify direct cell-level metric calculation works with explicit masks.

    This tests the lower-level metric function without going through the full
    AnnData-level entry point.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Build the deterministic QC configuration.
    config = make_metrics_config()

    # Build feature masks.
    feature_masks = build_feature_masks(adata, config.features)

    # Calculate cell-level metrics directly.
    metrics = calculate_cell_qc_metrics(
        adata.X,
        obs_names=adata.obs_names,
        feature_masks=feature_masks,
        percent_top=[2],
        log1p=True,
    )

    # Confirm direct cell metrics match expected total counts.
    np.testing.assert_allclose(metrics["total_counts"], [10.0, 5.0, 10.0, 0.0])

    # Confirm direct cell metrics include mitochondrial percentage.
    np.testing.assert_allclose(metrics["pct_counts_mito"], [50.0, 0.0, 10.0, 0.0])


def test_calculate_gene_qc_metrics_accepts_direct_inputs() -> None:
    """
    Verify direct gene-level metric calculation works with explicit names.

    This tests the lower-level gene metric function independently from AnnData
    validation and feature-mask logic.
    """

    # Build the dense test AnnData object.
    adata = make_metrics_test_adata()

    # Calculate gene-level metrics directly.
    metrics = calculate_gene_qc_metrics(
        adata.X,
        var_names=adata.var_names,
        log1p=True,
    )

    # Confirm direct gene metrics match expected total counts.
    np.testing.assert_allclose(
        metrics["total_counts"],
        [6.0, 1.0, 1.0, 2.0, 1.0, 5.0, 9.0],
    )


def test_validate_matrix_mask_alignment_rejects_mismatched_masks() -> None:
    """
    Verify matrix and feature-mask dimension mismatches fail clearly.

    Feature masks must have one row per matrix variable.
    """

    # Build a two-variable matrix.
    matrix = np.ones((2, 2), dtype=float)

    # Build a one-row feature-mask table.
    masks = pd.DataFrame({MITO_COLUMN: [True]})

    # Confirm mismatched masks fail validation.
    with pytest.raises(QCMetricsError, match="feature masks contain"):
        validate_matrix_mask_alignment(matrix, masks)


def test_calculate_gene_qc_metrics_rejects_var_name_mismatch() -> None:
    """
    Verify gene metric calculation rejects mismatched variable names.

    Gene-level metrics must align exactly to one feature name per matrix column.
    """

    # Build a two-variable matrix.
    matrix = np.ones((2, 2), dtype=float)

    # Build a one-name variable index.
    var_names = pd.Index(["gene_1"])

    # Confirm mismatched variable names fail clearly.
    with pytest.raises(QCMetricsError, match="variable names were provided"):
        calculate_gene_qc_metrics(matrix, var_names=var_names, log1p=True)


def test_sum_axis_matches_expected_values_dense_and_sparse() -> None:
    """
    Verify sparse-aware axis summation works for dense and sparse matrices.

    This helper underlies total count calculations.
    """

    # Build a small dense matrix.
    dense_matrix = np.array([[1, 0, 3], [0, 2, 1]], dtype=float)

    # Build the corresponding sparse matrix.
    sparse_matrix = sp.csr_matrix(dense_matrix)

    # Confirm dense row sums.
    np.testing.assert_allclose(sum_axis(dense_matrix, axis=1), [4.0, 3.0])

    # Confirm sparse row sums.
    np.testing.assert_allclose(sum_axis(sparse_matrix, axis=1), [4.0, 3.0])

    # Confirm dense column sums.
    np.testing.assert_allclose(sum_axis(dense_matrix, axis=0), [1.0, 2.0, 4.0])

    # Confirm sparse column sums.
    np.testing.assert_allclose(sum_axis(sparse_matrix, axis=0), [1.0, 2.0, 4.0])


def test_count_positive_axis_matches_expected_values_dense_and_sparse() -> None:
    """
    Verify positive-value counting works for dense and sparse matrices.

    This helper underlies detected-gene and detected-cell metrics.
    """

    # Build a small dense matrix.
    dense_matrix = np.array([[1, 0, 3], [0, 2, 1]], dtype=float)

    # Build the corresponding sparse matrix.
    sparse_matrix = sp.csr_matrix(dense_matrix)

    # Confirm dense row positive counts.
    np.testing.assert_array_equal(count_positive_axis(dense_matrix, axis=1), [2, 2])

    # Confirm sparse row positive counts.
    np.testing.assert_array_equal(count_positive_axis(sparse_matrix, axis=1), [2, 2])

    # Confirm dense column positive counts.
    np.testing.assert_array_equal(count_positive_axis(dense_matrix, axis=0), [1, 1, 2])

    # Confirm sparse column positive counts.
    np.testing.assert_array_equal(count_positive_axis(sparse_matrix, axis=0), [1, 1, 2])


def test_sum_columns_by_mask_returns_selected_column_sums() -> None:
    """
    Verify selected-column summation works.

    Feature-family count totals rely on summing masked columns per cell.
    """

    # Build a small dense matrix.
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    # Build a mask selecting columns zero and two.
    mask = np.array([True, False, True])

    # Confirm selected-column sums.
    np.testing.assert_allclose(sum_columns_by_mask(matrix, mask), [4.0, 10.0])


def test_sum_columns_by_mask_returns_zero_when_no_columns_selected() -> None:
    """
    Verify selected-column summation handles empty masks.

    If a feature family is absent, its per-cell count totals should be zero.
    """

    # Build a small dense matrix.
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    # Build an all-False mask.
    mask = np.array([False, False, False])

    # Confirm selected-column sums are zero.
    np.testing.assert_allclose(sum_columns_by_mask(matrix, mask), [0.0, 0.0])


def test_sum_columns_by_mask_rejects_mask_length_mismatch() -> None:
    """
    Verify selected-column summation rejects mask length mismatches.

    A mask length mismatch would assign feature families to the wrong columns.
    """

    # Build a three-variable matrix.
    matrix = np.array([[1, 2, 3]], dtype=float)

    # Build a two-entry mask.
    mask = np.array([True, False])

    # Confirm mismatched masks fail clearly.
    with pytest.raises(QCMetricsError, match="Feature mask has length"):
        sum_columns_by_mask(matrix, mask)


def test_safe_percent_handles_zero_denominators() -> None:
    """
    Verify percentage calculation safely handles zero denominators.

    Zero-count cells should receive zero percentage rather than NaN or infinity.
    """

    # Build numerator values.
    numerator = np.array([5.0, 0.0, 1.0])

    # Build denominator values with a zero denominator.
    denominator = np.array([10.0, 0.0, 4.0])

    # Confirm safe percentages.
    np.testing.assert_allclose(safe_percent(numerator, denominator), [50.0, 0.0, 25.0])


def test_sum_top_n_values_handles_empty_and_large_top_n() -> None:
    """
    Verify top-n summation handles empty and oversized requests.

    Percent-top metrics should work for sparse zero rows and top_n values larger
    than the number of genes.
    """

    # Confirm empty arrays sum to zero.
    assert sum_top_n_values(np.array([], dtype=float), 2) == 0.0

    # Confirm top_n larger than row length sums all values.
    assert sum_top_n_values(np.array([1.0, 3.0, 2.0]), 10) == 6.0

    # Confirm normal top-n summation.
    assert sum_top_n_values(np.array([1.0, 3.0, 2.0]), 2) == 5.0


def test_calculate_percent_top_matches_expected_dense_and_sparse() -> None:
    """
    Verify percent-top metrics are correct for dense and sparse matrices.

    This checks the metric used by the best-practices MAD filtering defaults.
    """

    # Build a small dense matrix.
    dense_matrix = np.array([[5, 1, 3], [0, 0, 0], [2, 2, 1]], dtype=float)

    # Build the corresponding sparse matrix.
    sparse_matrix = sp.csr_matrix(dense_matrix)

    # Build total counts.
    total_counts = np.array([9.0, 0.0, 5.0])

    # Define expected top-two percentages.
    expected = [88.8888888889, 0.0, 80.0]

    # Confirm dense percent-top values.
    np.testing.assert_allclose(
        calculate_percent_top(dense_matrix, total_counts, 2),
        expected,
    )

    # Confirm sparse percent-top values.
    np.testing.assert_allclose(
        calculate_percent_top(sparse_matrix, total_counts, 2),
        expected,
    )


def test_calculate_percent_top_rejects_non_positive_top_n() -> None:
    """
    Verify percent-top calculation rejects non-positive top_n values.

    This is a defensive check because QCConfig should normally prevent invalid
    percent_top settings.
    """

    # Build a small matrix.
    matrix = np.array([[1, 2]], dtype=float)

    # Build total counts.
    total_counts = np.array([3.0])

    # Confirm non-positive top_n fails clearly.
    with pytest.raises(QCMetricsError, match="top_n must be > 0"):
        calculate_percent_top(matrix, total_counts, 0)
