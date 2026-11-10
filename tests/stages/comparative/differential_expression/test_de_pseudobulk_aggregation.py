import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.stages.comparative.differential_expression.pseudobulk import aggregate_pseudobulk


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


def test_counts_and_design_match_the_four_donor_condition_pseudobulks():
    res = aggregate_pseudobulk(
        _toy_adata(), layer="counts", donor_col="patient_id", condition_col="condition"
    )
    expected = {
        "d1__Normal": [4, 0, 2],
        "d1__LE": [0, 5, 1],
        "d2__Normal": [2, 2, 2],
        "d2__LE": [1, 2, 1],
    }
    assert list(res.counts.columns) == ["GeneA", "GeneB", "GeneC"]
    assert res.sample_meta.index.equals(res.counts.index)
    assert res.counts.index.is_unique and set(res.counts.index) == set(expected)
    for sample, values in expected.items():
        np.testing.assert_array_equal(res.counts.loc[sample], values)
        row = res.sample_meta.loc[sample]
        assert (row["patient_id"], row["condition"]) == tuple(sample.split("__"))
