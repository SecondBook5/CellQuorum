# tests/test_trajectory_viz_macrostate.py
import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory.viz.macrostate_viz import MacrostateVizMethod

cr = pytest.importorskip("cellrank")


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp
            figures = tmp / "f"

        self.paths = P()


def _fitted_estimator(tmp):
    n = 200
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
    sc.tl.umap(a)
    a.obs["pt"] = np.linspace(0, 1, n)
    pk = cr.kernels.PseudotimeKernel(a, time_key="pt").compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(a).compute_transition_matrix()
    comb = 0.8 * pk + 0.2 * ck
    comb.compute_transition_matrix()
    g = cr.estimators.GPCCA(comb)
    g.compute_schur(n_components=6, method="brandts")
    g.compute_macrostates(n_states=3, cluster_key="ct")
    d = tmp / "trajectory" / "cellrank"
    d.mkdir(parents=True)
    g.write(str(d / "gpcca_estimator.pickle"))
    a.write_h5ad(d / "fate_mapping.h5ad")


def test_renders_macrostate_figures(tmp_path):
    _fitted_estimator(tmp_path)
    res = MacrostateVizMethod()._run(
        ad.AnnData(np.zeros((2, 2))), {"figure_formats": ["png"], "dpi": 72}, _Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    figs = list((tmp_path / "f" / "trajectory").glob("*.png"))
    assert len(figs) >= 1


def test_skips_without_pickle(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    res = MacrostateVizMethod()._run(ad.AnnData(np.zeros((2, 2))), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
