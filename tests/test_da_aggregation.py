"""Tests for differential abundance aggregation helpers."""

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.comparative.differential_abundance.aggregation import aggregate_celltype_counts


def _adata():
    """Create a minimal test AnnData with 2 donors x 2 conditions x 3 cell type observations."""
    obs = pd.DataFrame(
        {
            "patient_id": (["d1"] * 6 + ["d2"] * 6),
            "condition": (["Normal"] * 3 + ["LE"] * 3) * 2,
            "cell_type": (["Tcell", "Fib", "Fib"] * 4),
        }
    )
    a = ad.AnnData(X=np.zeros((12, 4)), obs=obs)
    return a


def test_aggregate_celltype_counts_shape_and_totals():
    """Test basic aggregation shape, cell counts, and condition recording."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    # 2 donors x 2 conditions = up to 4 samples; 2 cell types.
    assert set(res.counts.columns) == {"Tcell", "Fib"}
    assert res.counts.values.sum() == 12
    # Each sample's condition is recorded.
    assert set(res.sample_meta["condition"]).issubset({"Normal", "LE"})


def test_aggregate_celltype_counts_row_sums():
    """Test that row sums match cell counts per sample."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    # Each sample's row sum should match the number of cells in that sample
    # d1, Normal: 3 cells; d1, LE: 3 cells; d2, Normal: 3 cells; d2, LE: 3 cells
    expected_totals = {
        "d1_Normal": 3,
        "d1_LE": 3,
        "d2_Normal": 3,
        "d2_LE": 3,
    }
    for sample_id, expected_count in expected_totals.items():
        if sample_id in res.counts.index:
            assert res.counts.loc[sample_id].sum() == expected_count


def test_aggregate_celltype_counts_integer_values():
    """Test that counts are integer-valued."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    assert res.counts.values.dtype == np.int64 or res.counts.values.dtype == int


def test_aggregate_celltype_counts_meta_alignment():
    """Test that sample_meta is properly aligned with counts index."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    # sample_meta should have the same index as counts
    assert (res.counts.index == res.sample_meta.index).all()
    # sample_meta should have donor_col and condition_col
    assert "patient_id" in res.sample_meta.columns
    assert "condition" in res.sample_meta.columns
