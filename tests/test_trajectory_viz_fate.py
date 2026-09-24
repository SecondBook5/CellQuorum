import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory.viz._kernel_plots import FateVizMethod


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp / "r"
            figures = tmp / "f"

        self.paths = P()


def _adata(n=40, fate=True):
    a = ad.AnnData(np.zeros((n, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obsm["X_umap"] = np.random.RandomState(0).rand(n, 2)
    if fate:
        fp = np.random.RandomState(1).dirichlet(np.ones(2), size=n)
        a.obsm["cellrank_fate_probabilities"] = fp
        a.uns["trajectory"] = {"cellrank": {"fate_names": ["L0", "L1"]}}
    return a


def test_renders_fate_per_lineage(tmp_path):
    res = FateVizMethod()._run(_adata(), {"figure_formats": ["png"], "dpi": 72}, _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    figs = list((tmp_path / "f" / "trajectory").glob("fate_*.png"))
    assert len(figs) == 2  # one per lineage


def test_skips_without_fate(tmp_path):
    res = FateVizMethod()._run(_adata(fate=False), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)


def test_handles_fate_names_as_a_numpy_array_not_a_list(tmp_path):
    """Real crash on the fibroblast cohort: uns['trajectory']['cellrank']
    ['fate_names'] round-trips through h5ad as a numpy array of strings, not
    the plain Python list every existing test constructs in-memory.
    `if fate_names and len(fate_names) == n_lin:` raised "The truth value of
    an array with more than one element is ambiguous" the moment a real,
    multi-element array reached this check -- a crash that a list-only test
    fixture could never catch.
    """
    a = _adata()
    a.uns["trajectory"]["cellrank"]["fate_names"] = np.array(["L0", "L1"], dtype=object)
    res = FateVizMethod()._run(a, {"figure_formats": ["png"], "dpi": 72}, _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    figs = list((tmp_path / "f" / "trajectory").glob("fate_*.png"))
    assert len(figs) == 2


def test_fallback_lineage_names_when_uns_trajectory_not_dict(tmp_path):
    """Regression: uns['trajectory'] as non-dict must not raise AttributeError."""
    a = ad.AnnData(np.zeros((40, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(40)]
    a.obsm["X_umap"] = np.random.RandomState(0).rand(40, 2)
    fp = np.random.RandomState(1).dirichlet(np.ones(2), size=40)
    a.obsm["cellrank_fate_probabilities"] = fp
    a.uns["trajectory"] = "not_a_dict"  # Poison the uns chain
    res = FateVizMethod()._run(a, {"figure_formats": ["png"], "dpi": 72}, _Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)  # Should render with fallback lineage_N names
    figs = list((tmp_path / "f" / "trajectory").glob("fate_*.png"))
    assert len(figs) == 2  # Falls back to lineage_0, lineage_1
