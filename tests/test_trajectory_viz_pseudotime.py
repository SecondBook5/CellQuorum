import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory.viz._pseudotime_plots import PseudotimeVizMethod


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "res"
        self.figures = tmp / "fig"


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata(n=40, with_pt=True, with_basis=True):
    a = ad.AnnData(np.zeros((n, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    if with_basis:
        a.obsm["X_umap"] = np.random.RandomState(0).rand(n, 2)
    if with_pt:
        a.obs["dpt_pseudotime"] = np.linspace(0, 1, n)
    return a


def test_renders_pseudotime_scatter(tmp_path):
    ctx = _Ctx(tmp_path)
    res = PseudotimeVizMethod()._run(_adata(), {"figure_formats": ["png"], "dpi": 72}, ctx)
    assert not isinstance(res, MethodSkip)
    figs = list((tmp_path / "fig" / "trajectory").glob("pseudotime_dpt_pseudotime.png"))
    assert len(figs) == 1
    assert res.metrics["n_figures"] == 1


def test_skips_without_basis(tmp_path):
    res = PseudotimeVizMethod()._run(_adata(with_basis=False), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)


def test_skips_without_pseudotime(tmp_path):
    res = PseudotimeVizMethod()._run(_adata(with_pt=False), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
