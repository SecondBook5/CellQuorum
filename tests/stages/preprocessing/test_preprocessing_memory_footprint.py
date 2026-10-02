"""Tests for the memory invariants preprocessing has to hold at atlas scale.

These are regression tests for two defects that made the pipeline unable to
normalize a large object at all, and that neither of the existing normalization
test modules could catch because every fixture in them is small enough that a
dense copy is free.

The invariants, stated as invariants rather than as byte counts, because the size
that breaks is a property of the caller's data and not of this repository:

* Nothing in the normalization path may densify a sparse count matrix. A sparse
  matrix is used precisely because its dense form does not fit, so a single
  ``.toarray()`` anywhere in the path is equivalent to not supporting sparse
  input. This is asserted with a tripwire matrix that raises when densified,
  which pins the behaviour exactly instead of inferring it from a memory
  measurement that depends on the machine.
* The normalized layer is stored in single precision. It is the largest array the
  pipeline carries and it is carried for the whole run, so a float64 layer
  doubles the footprint of every stage after preprocessing and of every
  checkpoint written along the way.
"""

from __future__ import annotations

# Import AnnData to exercise the layer-writing path on a real object.
import anndata as ad

# Import NumPy for the dense fixtures and dtype assertions.
import numpy as np

# Import pytest for the exception assertions.
import pytest

# Import scipy.sparse for the sparse fixtures.
import scipy.sparse as sp

# Import the functions under test.
from cellquorum.stages.preprocessing.normalization import (
    PreprocessingNormalizationError,
    downcast_to_float32,
    validate_count_matrix,
    write_normalized_layer,
)


class NoDensifyCSR(sp.csr_matrix):
    """
    A CSR matrix that fails loudly instead of producing a dense copy.

    Used to assert the "never densify" invariant directly. A test that instead
    built a matrix too large to densify would depend on the host's memory and
    overcommit settings, and would fail by hanging rather than by reporting.
    """

    def toarray(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201, D102
        raise AssertionError("densified a sparse count matrix")

    def todense(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201, D102
        raise AssertionError("densified a sparse count matrix")


def make_sparse_counts(values: list[float]) -> NoDensifyCSR:
    """
    Build a one-row tripwire count matrix holding ``values`` as its nonzeros.

    Args:
        values: Stored values, in column order.

    Returns:
        A CSR matrix that raises if anything tries to densify it.
    """

    # Place every value in its own column so none of them is a structural zero.
    data = np.asarray(values, dtype=np.float64)
    indices = np.arange(len(values), dtype=np.int32)
    indptr = np.asarray([0, len(values)], dtype=np.int32)
    return NoDensifyCSR((data, indices, indptr), shape=(1, len(values)))


def test_validate_count_matrix_does_not_densify_sparse_input():
    """A valid sparse count matrix validates without ever being densified."""
    validate_count_matrix(make_sparse_counts([3.0, 0.0, 7.0]))


def test_validate_count_matrix_rejects_negative_sparse_value():
    """A negative stored value is still caught on the sparse path."""
    with pytest.raises(PreprocessingNormalizationError, match="negative"):
        validate_count_matrix(make_sparse_counts([3.0, -1.0]))


def test_validate_count_matrix_rejects_nonfinite_sparse_value():
    """A NaN stored value is still caught on the sparse path."""
    with pytest.raises(PreprocessingNormalizationError, match="non-finite"):
        validate_count_matrix(make_sparse_counts([3.0, np.nan]))


def test_validate_count_matrix_rejects_infinite_sparse_value():
    """An infinite stored value is still caught on the sparse path."""
    with pytest.raises(PreprocessingNormalizationError, match="non-finite"):
        validate_count_matrix(make_sparse_counts([3.0, np.inf]))


def test_validate_count_matrix_rejects_negative_dense_value():
    """The dense path is unchanged: a negative entry is still rejected."""
    with pytest.raises(PreprocessingNormalizationError, match="negative"):
        validate_count_matrix(np.array([[1.0, -2.0]], dtype=np.float32))


def test_validate_count_matrix_rejects_nonfinite_dense_value():
    """The dense path is unchanged: a NaN entry is still rejected."""
    with pytest.raises(PreprocessingNormalizationError, match="non-finite"):
        validate_count_matrix(np.array([[1.0, np.nan]], dtype=np.float32))


def test_validate_count_matrix_rejects_non_numeric_matrix():
    """A non-numeric matrix is rejected before either value check runs."""
    with pytest.raises(PreprocessingNormalizationError, match="numeric"):
        validate_count_matrix(np.array([["a", "b"]]))


def test_downcast_to_float32_returns_float32_input_unchanged():
    """An already-single-precision matrix is returned as-is, not copied."""
    matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    assert downcast_to_float32(matrix) is matrix


def test_downcast_to_float32_recasts_dense_float64():
    """A dense float64 matrix comes back as float32 with its values intact."""
    result = downcast_to_float32(np.array([[1.5, 2.25]], dtype=np.float64))
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[1.5, 2.25]])


def test_downcast_to_float32_keeps_sparse_sparse():
    """A sparse float64 matrix stays sparse, in the same format, as float32."""
    matrix = sp.csr_matrix(np.array([[1.5, 0.0, 2.25]], dtype=np.float64))
    result = downcast_to_float32(matrix)
    assert sp.issparse(result)
    assert result.format == "csr"
    assert result.dtype == np.float32
    np.testing.assert_allclose(result.data, [1.5, 2.25])


def test_downcast_to_float32_does_not_densify_sparse_input():
    """The downcast never reaches for a dense copy of a sparse matrix."""
    result = downcast_to_float32(make_sparse_counts([1.5, 2.25]))
    assert sp.issparse(result)
    assert result.dtype == np.float32


def test_write_normalized_layer_stores_sparse_layer_as_float32():
    """A sparse normalized matrix is stored single-precision and still sparse."""
    adata = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32))
    normalized = sp.csr_matrix(np.array([[1.5, 0.0, 2.5], [0.0, 3.5, 0.0]], dtype=np.float64))

    write_normalized_layer(adata, normalized, output_layer="lognorm", overwrite=False)

    stored = adata.layers["lognorm"]
    assert sp.issparse(stored)
    assert stored.dtype == np.float32


def test_write_normalized_layer_stores_dense_layer_as_float32():
    """A dense normalized matrix is stored single-precision too."""
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    normalized = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    write_normalized_layer(adata, normalized, output_layer="lognorm", overwrite=False)

    assert adata.layers["lognorm"].dtype == np.float32


def test_write_normalized_layer_still_refuses_to_overwrite():
    """The overwrite guard is unaffected by the downcast."""
    adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata.layers["lognorm"] = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(PreprocessingNormalizationError, match="already exists"):
        write_normalized_layer(
            adata, np.zeros((2, 2), dtype=np.float64), output_layer="lognorm", overwrite=False
        )
