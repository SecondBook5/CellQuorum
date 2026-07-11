"""Tests for preprocessing visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellquorum.preprocessing.visualization import write_normalization_figures


def make_test_adata_with_normalization() -> ad.AnnData:
    """Create test AnnData with counts and normalized layers."""
    counts = np.array(
        [
            [5, 5, 0, 0, 1, 2],
            [9, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 3],
            [2, 3, 1, 0, 5, 1],
        ],
        dtype=np.float32,
    )

    # Simulate normalization (log1p of proportional fractions)
    cell_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.log1p(counts / cell_totals)

    adata = ad.AnnData(X=counts)
    adata.layers["counts"] = counts.copy()
    adata.layers["cellquorum_normalized"] = normalized

    return adata


def test_write_normalization_figures_creates_expected_files() -> None:
    """Test that normalization figures are created successfully."""
    adata = make_test_adata_with_normalization()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = write_normalization_figures(adata, output_dir, dpi=100)

        # Should have created 4 figures
        assert len(result.figure_paths) == 4

        # Check that files exist
        for fig_path in result.figure_paths:
            assert fig_path.exists()
            assert fig_path.stat().st_size > 0

        # Should have no warnings
        assert len(result.warnings) == 0


def test_write_normalization_figures_with_sparse_matrices() -> None:
    """Test that normalization figures handle sparse matrices without densification."""
    adata = make_test_adata_with_normalization()

    # Convert to sparse
    adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])
    adata.layers["cellquorum_normalized"] = sp.csr_matrix(adata.layers["cellquorum_normalized"])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = write_normalization_figures(adata, output_dir, dpi=100)

        # Should succeed with sparse matrices
        assert len(result.figure_paths) == 4
        assert len(result.warnings) == 0

        # Verify matrices are still sparse
        assert sp.issparse(adata.layers["counts"])
        assert sp.issparse(adata.layers["cellquorum_normalized"])


def test_write_normalization_figures_missing_layers() -> None:
    """Test that missing layers are handled gracefully."""
    adata = ad.AnnData(X=np.random.rand(10, 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = write_normalization_figures(adata, output_dir, dpi=100)

        # Should return early with warnings
        assert len(result.figure_paths) == 0
        assert len(result.warnings) > 0
        assert any("counts" in w for w in result.warnings)


def test_write_normalization_figures_respects_overwrite() -> None:
    """Test that overwrite flag works correctly."""
    adata = make_test_adata_with_normalization()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First write
        result1 = write_normalization_figures(adata, output_dir, dpi=100, overwrite=False)
        assert len(result1.figure_paths) == 4

        # Second write without overwrite
        result2 = write_normalization_figures(adata, output_dir, dpi=100, overwrite=False)
        assert len(result2.figure_paths) == 4

        # With overwrite=True
        result3 = write_normalization_figures(adata, output_dir, dpi=100, overwrite=True)
        assert len(result3.figure_paths) == 4


def test_write_normalization_figures_different_formats() -> None:
    """Test that different figure formats work."""
    adata = make_test_adata_with_normalization()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Test PNG
        result_png = write_normalization_figures(adata, output_dir, figure_format="png", dpi=100)
        assert all(p.suffix == ".png" for p in result_png.figure_paths)

        # Test PDF
        result_pdf = write_normalization_figures(
            adata, output_dir, figure_format="pdf", dpi=100, overwrite=True
        )
        assert all(p.suffix == ".pdf" for p in result_pdf.figure_paths)


def test_write_normalization_figures_with_many_cells() -> None:
    """Test that figures are created efficiently with many cells."""
    # Create larger dataset
    np.random.seed(42)
    counts = np.random.poisson(5, size=(10000, 100)).astype(np.float32)
    cell_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.log1p(counts / cell_totals)

    adata = ad.AnnData(X=counts)
    adata.layers["counts"] = counts.copy()
    adata.layers["cellquorum_normalized"] = normalized

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = write_normalization_figures(adata, output_dir, dpi=100)

        # Should succeed and create all figures
        assert len(result.figure_paths) == 4
        assert len(result.warnings) == 0
