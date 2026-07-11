"""Tests for QC visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.qc.visualization import write_qc_figures


def make_test_adata_with_qc() -> ad.AnnData:
    """Create test AnnData with QC metrics."""
    matrix = np.array(
        [
            [5, 5, 0, 0, 1, 2],
            [9, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 3],
            [2, 3, 1, 0, 5, 1],
        ],
        dtype=np.float32,
    )

    obs = pd.DataFrame(
        {
            "total_counts": [13, 11, 6, 12],
            "n_genes_by_counts": [4, 3, 4, 5],
            "pct_counts_mito": [10.0, 15.0, 5.0, 12.0],
            "cellquorum_qc_keep": [True, True, False, True],
        },
        index=[f"cell_{i}" for i in range(4)],
    )

    var = pd.DataFrame(
        {"n_cells_by_counts": [3, 3, 2, 1, 3, 4]}, index=[f"gene_{i}" for i in range(6)]
    )

    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_write_qc_figures_creates_expected_files() -> None:
    """Test that QC figures are created successfully."""
    adata = make_test_adata_with_qc()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = write_qc_figures(adata, output_dir, dpi=100)

        # Should have created multiple figures
        assert len(result.figure_paths) > 0

        # Check that files exist
        for fig_path in result.figure_paths:
            assert fig_path.exists()
            assert fig_path.stat().st_size > 0


def test_write_qc_figures_without_optional_metrics() -> None:
    """Test that QC figures handle missing optional metrics gracefully."""
    adata = make_test_adata_with_qc()

    # Remove optional metrics
    adata.obs = adata.obs.drop(columns=["pct_counts_mito", "cellquorum_qc_keep"])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = write_qc_figures(adata, output_dir, dpi=100)

        # Should succeed with warnings
        assert len(result.warnings) > 0
        assert any("pct_counts_mito" in w for w in result.warnings)
        assert any("cellquorum_qc_keep" in w for w in result.warnings)

        # Should still have created some figures
        assert len(result.figure_paths) > 0


def test_write_qc_figures_respects_overwrite() -> None:
    """Test that overwrite flag works correctly."""
    adata = make_test_adata_with_qc()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First write
        result1 = write_qc_figures(adata, output_dir, dpi=100, overwrite=False)
        original_paths = result1.figure_paths.copy()

        # Get modification times
        original_mtimes = {p: p.stat().st_mtime for p in original_paths}

        # Second write without overwrite (should reuse existing)
        result2 = write_qc_figures(adata, output_dir, dpi=100, overwrite=False)

        # Files should still exist and NOT be recreated (mtimes unchanged).
        for p in result2.figure_paths:
            assert p.exists()
            if p in original_mtimes:
                assert p.stat().st_mtime == original_mtimes[p]

        # With overwrite=True, files should be recreated
        result3 = write_qc_figures(adata, output_dir, dpi=100, overwrite=True)
        assert len(result3.figure_paths) > 0


def test_write_qc_figures_different_formats() -> None:
    """Test that different figure formats work."""
    adata = make_test_adata_with_qc()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Test PNG (default)
        result_png = write_qc_figures(adata, output_dir, figure_format="png", dpi=100)
        assert all(p.suffix == ".png" for p in result_png.figure_paths)

        # Test PDF
        result_pdf = write_qc_figures(
            adata, output_dir, figure_format="pdf", dpi=100, overwrite=True
        )
        assert all(p.suffix == ".pdf" for p in result_pdf.figure_paths)
