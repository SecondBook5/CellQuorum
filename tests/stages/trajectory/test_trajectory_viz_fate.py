import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np
import pandas as pd

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


def test_fate_labels_reach_the_scatter_title_and_violin_ticks(tmp_path, monkeypatch):
    """Real ask: bare macrostate/subcluster ids ("0", "1", "4") on fate_*.png
    titles and fate_violin_*.png x-ticks are illegible without a separate
    lookup table -- the same fate_labels dict already used by
    driver_viz/velocity_viz/coarse_transition_heatmap must resolve them here
    too, in both places.
    """
    from cellquorum.stages.trajectory.viz import _kernel_plots

    a = _adata()
    a.obs["sub"] = pd.Categorical(["0"] * 20 + ["4"] * 20)
    real_scatter = _kernel_plots._helpers.embedding_scatter
    real_violin = _kernel_plots._helpers.grouped_violin
    scatter_titles = []
    violin_calls = []

    def _scatter_spy(coords, values, *, title, cbar_label):
        scatter_titles.append(title)
        return real_scatter(coords, values, title=title, cbar_label=cbar_label)

    def _violin_spy(groups, *, title, ylabel, labels=None):
        violin_calls.append((title, labels))
        return real_violin(groups, title=title, ylabel=ylabel, labels=labels)

    monkeypatch.setattr(_kernel_plots._helpers, "embedding_scatter", _scatter_spy)
    monkeypatch.setattr(_kernel_plots._helpers, "grouped_violin", _violin_spy)

    res = FateVizMethod()._run(
        a,
        {
            "figure_formats": ["png"],
            "dpi": 72,
            "cluster_key": "sub",
            "fate_labels": {"0": "Perivascular", "4": "Inflammatory", "L0": "Terminal state"},
        },
        _Ctx(tmp_path),
    )
    assert not isinstance(res, MethodSkip)
    assert "Terminal state" in "".join(scatter_titles)  # "L0 (Terminal state)"
    assert violin_calls
    for title, labels in violin_calls:
        assert "Terminal state" in title or "L1" in title
        assert labels == {"0": "Perivascular", "4": "Inflammatory", "L0": "Terminal state"}


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
