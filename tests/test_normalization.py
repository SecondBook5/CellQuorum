"""Tests for preprocessing normalization implementation."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellquorum.preprocessing.config import NormalizationConfig
from cellquorum.preprocessing.normalization import (
    NormalizationResult,
    PreprocessingNormalizationError,
    normalize_adata,
)


def make_test_adata() -> ad.AnnData:
    """
    Build a deterministic tiny AnnData for testing.

    Matrix:
      [[5, 5, 0, 0],   # cell 0: depth 10
       [9, 0, 0, 0],   # cell 1: depth 9
       [0, 1, 1, 1]]   # cell 2: depth 3

    Genes: MT-ND1, ACTB, RPS3, MALAT1
    """
    matrix = np.array(
        [
            [5, 5, 0, 0],
            [9, 0, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    genes = ["MT-ND1", "ACTB", "RPS3", "MALAT1"]
    cells = ["cell_0", "cell_1", "cell_2"]

    return ad.AnnData(X=matrix, obs={"cell_id": cells}, var={"gene_name": genes})


def test_normalize_adata_returns_result():
    """Test that normalize_adata returns NormalizationResult."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="none")

    result = normalize_adata(adata, config)

    assert isinstance(result, NormalizationResult)
    assert isinstance(result.adata, ad.AnnData)
    assert result.recipe == "none"


def test_normalize_adata_copy_preserves_original():
    """Test that copy=True preserves the original AnnData."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="none")

    original_x = adata.X.copy()

    result = normalize_adata(adata, config, copy=True)

    # Original is unchanged.
    assert np.array_equal(adata.X, original_x)

    # Result is different object.
    assert result.adata is not adata


def test_normalize_adata_preserves_counts_layer():
    """Test that raw counts are preserved in a separate layer."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="cellquorum_pf_v1", preserve_counts_layer="counts")

    result = normalize_adata(adata, config)

    # Counts layer should exist.
    assert "counts" in result.adata.layers

    # Counts layer should match original X.
    assert np.array_equal(result.adata.layers["counts"], adata.X)


def test_normalize_adata_writes_output_layer():
    """Test that normalized values are written to output layer."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="cellquorum_pf_v1", output_layer="normalized")

    result = normalize_adata(adata, config)

    # Output layer should exist.
    assert "normalized" in result.adata.layers

    # Output layer should be numeric.
    assert np.issubdtype(result.adata.layers["normalized"].dtype, np.number)


def test_normalize_adata_overwrite_protection():
    """Test that overwrite=False protects existing layers."""
    adata = make_test_adata()
    adata.layers["normalized"] = np.zeros_like(adata.X)

    config = NormalizationConfig(
        recipe="cellquorum_pf_v1",
        output_layer="normalized",
        overwrite=False,
    )

    with pytest.raises(PreprocessingNormalizationError, match="already exists"):
        normalize_adata(adata, config)


def test_normalize_adata_overwrite_allowed():
    """Test that overwrite=True allows replacing existing layers."""
    adata = make_test_adata()
    adata.layers["normalized"] = np.zeros_like(adata.X)

    config = NormalizationConfig(
        recipe="cellquorum_pf_v1",
        output_layer="normalized",
        overwrite=True,
    )

    result = normalize_adata(adata, config)

    # Layer should be overwritten.
    assert not np.array_equal(result.adata.layers["normalized"], np.zeros_like(adata.X))


def test_normalize_adata_writes_provenance():
    """Test that normalization provenance is written to uns."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="cellquorum_pf_v1")

    result = normalize_adata(adata, config)

    # Provenance should be written.
    assert "cellquorum" in result.adata.uns
    assert "preprocessing" in result.adata.uns["cellquorum"]
    assert "normalization" in result.adata.uns["cellquorum"]["preprocessing"]

    provenance = result.adata.uns["cellquorum"]["preprocessing"]["normalization"]
    assert provenance["recipe"] == "cellquorum_pf_v1"
    assert provenance["output_layer"] == "cellquorum_normalized"


def test_recipe_none():
    """Test that recipe 'none' produces unchanged output."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="none", output_layer="normalized")

    result = normalize_adata(adata, config)

    # Output should match raw counts.
    assert np.array_equal(result.adata.layers["normalized"], adata.X)


def test_recipe_pf_v1():
    """Test recipe 'cellquorum_pf_v1' (proportional fractions)."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="cellquorum_pf_v1", output_layer="normalized")

    result = normalize_adata(adata, config)

    normalized = result.adata.layers["normalized"]

    # Cell 0: depth 10 -> [5/10, 5/10, 0/10, 0/10] = [0.5, 0.5, 0.0, 0.0]
    assert np.allclose(normalized[0, :], [0.5, 0.5, 0.0, 0.0])

    # Cell 1: depth 9 -> [9/9, 0/9, 0/9, 0/9] = [1.0, 0.0, 0.0, 0.0]
    assert np.allclose(normalized[1, :], [1.0, 0.0, 0.0, 0.0])

    # Cell 2: depth 3 -> [0/3, 1/3, 1/3, 1/3]
    assert np.allclose(normalized[2, :], [0.0, 1 / 3, 1 / 3, 1 / 3])


def test_recipe_log1p_cp10k_v1():
    """Test recipe 'cellquorum_log1p_cp10k_v1' (counts per 10k + log1p)."""
    adata = make_test_adata()
    config = NormalizationConfig(
        recipe="cellquorum_log1p_cp10k_v1",
        target_sum=10000.0,
        output_layer="normalized",
    )

    result = normalize_adata(adata, config)

    normalized = result.adata.layers["normalized"]

    # Cell 0: depth 10
    # Scaled: [5/10 * 10000, 5/10 * 10000, 0, 0] = [5000, 5000, 0, 0]
    # Log1p: [log(5001), log(5001), log(1), log(1)]
    expected_cell_0 = np.log1p([5000, 5000, 0, 0])
    assert np.allclose(normalized[0, :], expected_cell_0)


def test_recipe_log1p_pf_v1():
    """Test recipe 'cellquorum_log1p_pf_v1' (log1p proportional fractions)."""
    adata = make_test_adata()
    config = NormalizationConfig(
        recipe="cellquorum_log1p_pf_v1",
        output_layer="normalized",
    )

    result = normalize_adata(adata, config)

    normalized = result.adata.layers["normalized"]

    # Cell 0: depth 10 -> pf = [0.5, 0.5, 0.0, 0.0]
    # Log1p: [log(1.5), log(1.5), log(1.0), log(1.0)]
    expected_cell_0 = np.log1p([0.5, 0.5, 0.0, 0.0])
    assert np.allclose(normalized[0, :], expected_cell_0)


def test_recipe_pf_log1p_pf_v1():
    """Test recipe 'cellquorum_pf_log1p_pf_v1' (shifted CLR-like)."""
    adata = make_test_adata()
    config = NormalizationConfig(
        recipe="cellquorum_pf_log1p_pf_v1",
        pseudocount=1.0,
        output_layer="normalized",
    )

    result = normalize_adata(adata, config)

    normalized = result.adata.layers["normalized"]

    # Cell 0: depth 10 -> pf = [0.5, 0.5, 0.0, 0.0]
    # u_plus = log(pf + 1.0) = [log(1.5), log(1.5), log(1.0), log(1.0)]
    # mean = mean([log(1.5), log(1.5), log(1.0), log(1.0)])
    u_plus = np.log([0.5 + 1.0, 0.5 + 1.0, 0.0 + 1.0, 0.0 + 1.0])
    mean_val = u_plus.mean()
    expected_cell_0 = u_plus - mean_val

    assert np.allclose(normalized[0, :], expected_cell_0)


def test_zero_depth_cell_handling():
    """Test that zero-depth cells produce warnings."""
    # Create matrix with one zero-depth cell.
    matrix = np.array(
        [
            [5, 5, 0, 0],
            [0, 0, 0, 0],  # zero-depth cell
        ],
        dtype=np.float32,
    )

    adata = ad.AnnData(X=matrix)
    config = NormalizationConfig(recipe="cellquorum_pf_v1")

    result = normalize_adata(adata, config)

    # Should have warnings.
    assert len(result.warnings) > 0
    assert "zero-depth" in result.warnings[0].lower()


def test_zero_depth_cell_pf_log1p_pf_v1():
    """Test that zero-depth cells produce all-zero rows in CLR-like recipe."""
    # Create matrix with one zero-depth cell.
    matrix = np.array(
        [
            [5, 5, 0, 0],
            [0, 0, 0, 0],  # zero-depth cell
        ],
        dtype=np.float32,
    )

    adata = ad.AnnData(X=matrix)
    config = NormalizationConfig(
        recipe="cellquorum_pf_log1p_pf_v1",
        pseudocount=1.0,
        output_layer="normalized",
    )

    result = normalize_adata(adata, config)

    # Should have warning about zero-depth cells with updated text.
    assert len(result.warnings) > 0
    zero_depth_warning = [w for w in result.warnings if "zero-depth" in w.lower()]
    assert len(zero_depth_warning) > 0
    assert "all-zero centered rows" in zero_depth_warning[0]

    # Zero-depth cell should produce all-zero row.
    normalized = result.adata.layers["normalized"]
    assert np.allclose(normalized[1, :], 0.0)


def test_negative_matrix_rejected():
    """Test that matrices with negative values are rejected."""
    matrix = np.array(
        [
            [5, -1, 0, 0],
            [9, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    adata = ad.AnnData(X=matrix)
    config = NormalizationConfig(recipe="cellquorum_pf_v1")

    with pytest.raises(PreprocessingNormalizationError, match="negative values"):
        normalize_adata(adata, config)


def test_non_finite_matrix_rejected():
    """Test that matrices with NaN or Inf are rejected."""
    matrix = np.array(
        [
            [5, np.nan, 0, 0],
            [9, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    adata = ad.AnnData(X=matrix)
    config = NormalizationConfig(recipe="cellquorum_pf_v1")

    with pytest.raises(PreprocessingNormalizationError, match="non-finite values"):
        normalize_adata(adata, config)


def test_sparse_matrix_support():
    """Test that sparse matrices are supported."""
    adata = make_test_adata()
    adata.X = sp.csr_matrix(adata.X)

    config = NormalizationConfig(recipe="cellquorum_pf_v1", output_layer="normalized")

    result = normalize_adata(adata, config)

    # Should succeed.
    assert "normalized" in result.adata.layers


def test_input_layer_support():
    """Test that input_layer parameter works."""
    adata = make_test_adata()
    adata.layers["raw"] = adata.X.copy() * 2

    config = NormalizationConfig(
        recipe="cellquorum_pf_v1",
        input_layer="raw",
        output_layer="normalized",
        preserve_counts_layer="counts",
    )

    result = normalize_adata(adata, config)

    # Counts layer should preserve the raw layer.
    assert np.array_equal(result.adata.layers["counts"], adata.layers["raw"])

    # Normalized layer should be based on raw layer.
    assert not np.array_equal(result.adata.layers["normalized"], adata.X)


def test_input_layer_missing():
    """Test that missing input_layer is rejected."""
    adata = make_test_adata()

    config = NormalizationConfig(
        recipe="cellquorum_pf_v1",
        input_layer="missing_layer",
    )

    with pytest.raises(PreprocessingNormalizationError, match="not found"):
        normalize_adata(adata, config)


def test_diagnostics_populated():
    """Test that diagnostics contain expected fields."""
    adata = make_test_adata()
    config = NormalizationConfig(recipe="cellquorum_pf_v1")

    result = normalize_adata(adata, config)

    # Diagnostics should contain key fields.
    assert "n_cells" in result.diagnostics
    assert "n_genes" in result.diagnostics
    assert "recipe" in result.diagnostics
    assert "input_total_counts_min" in result.diagnostics
    assert "input_total_counts_max" in result.diagnostics

    # Values should match test data.
    assert result.diagnostics["n_cells"] == 3
    assert result.diagnostics["n_genes"] == 4
