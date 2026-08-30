import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from cellquorum.stages.trajectory.cellrank_method import CellRankMethod

cr = pytest.importorskip("cellrank")


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp
            figures = tmp / "f"

        self.paths = P()


def _adata(n=200):
    X = np.random.RandomState(0).poisson(1.0, size=(n, 50)).astype("float32")
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(50)]
    a.obs["ct"] = np.random.RandomState(1).choice(list("ABC"), n)
    a.obs["ct"] = a.obs["ct"].astype("category")
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=15)
    a.obs["pt"] = np.linspace(0, 1, n)
    return a


def test_cellrank_writes_readable_estimator_pickle(tmp_path):
    ctx = _Ctx(tmp_path)
    cfg = {
        "cluster_key": "ct",
        "pseudotime_key": "pt",
        "n_components": 6,
        "n_states": 3,
        "seed": 0,
        "terminal_method": "top_n",
        "n_terminal_states": 2,
    }
    CellRankMethod()._run(_adata(), cfg, ctx)
    pkl = tmp_path / "trajectory" / "cellrank" / "gpcca_estimator.pickle"
    assert pkl.exists()
    g = cr.estimators.GPCCA.read(str(pkl))
    assert g.macrostates is not None
