import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.differential_expression.pseudobulk import aggregate_pseudobulk


def _toy_adata():
    # 6 cells, 3 genes, 2 donors x 2 conditions (one group has 2 cells).
    counts = np.array(
        [
            [1, 0, 2],
            [3, 0, 0],  # d1 / Normal  (2 cells -> sums to [4,0,2])
            [0, 5, 1],  # d1 / LE
            [2, 2, 2],  # d2 / Normal
            [0, 1, 0],
            [1, 1, 1],
        ],  # d2 / LE      (2 cells -> sums to [1,2,1])
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "patient_id": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "condition": ["Normal", "Normal", "LE", "Normal", "LE", "LE"],
        }
    )
    a = ad.AnnData(X=sp.csr_matrix(counts), obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = ["GeneA", "GeneB", "GeneC"]
    return a


def test_aggregate_shapes_and_labels():
    res = aggregate_pseudobulk(
        _toy_adata(), layer="counts", donor_col="patient_id", condition_col="condition"
    )
    # 4 pseudo-samples (d1/Normal, d1/LE, d2/Normal, d2/LE), 3 genes.
    assert res.counts.shape == (4, 3)
    assert set(res.counts.index) == {"d1__Normal", "d1__LE", "d2__Normal", "d2__LE"}
    assert list(res.counts.columns) == ["GeneA", "GeneB", "GeneC"]


def test_aggregate_sums_within_group():
    res = aggregate_pseudobulk(
        _toy_adata(), layer="counts", donor_col="patient_id", condition_col="condition"
    )
    # d1/Normal is the sum of the first two cells: [1,0,2] + [3,0,0] = [4,0,2].
    np.testing.assert_array_equal(res.counts.loc["d1__Normal"].to_numpy(), [4, 0, 2])
    # d2/LE is the sum of the last two cells: [0,1,0] + [1,1,1] = [1,2,1].
    np.testing.assert_array_equal(res.counts.loc["d2__LE"].to_numpy(), [1, 2, 1])


def test_sample_meta_aligns_to_counts():
    res = aggregate_pseudobulk(
        _toy_adata(), layer="counts", donor_col="patient_id", condition_col="condition"
    )
    assert list(res.sample_meta.index) == list(res.counts.index)
    assert res.sample_meta.loc["d1__LE", "patient_id"] == "d1"
    assert res.sample_meta.loc["d1__LE", "condition"] == "LE"


def test_counts_are_integer_valued():
    res = aggregate_pseudobulk(
        _toy_adata(), layer="counts", donor_col="patient_id", condition_col="condition"
    )
    assert (res.counts.to_numpy() == res.counts.to_numpy().astype(int)).all()
