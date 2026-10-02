"""Tests for the activity-along-pseudotime cascade (trajectory viz)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory.viz import _helpers
from cellquorum.stages.trajectory.viz import activity_cascade as cascade
from cellquorum.stages.trajectory.viz._activity_cascade import ActivityCascadeVizMethod


def _monotone_activity(n: int = 300):
    """Activity frame: 'late' rises with pt, 'early' falls, plus the pt vector."""
    rng = np.random.default_rng(0)
    pt = np.linspace(0, 1, n)
    late = pt + rng.normal(0, 0.01, n)  # rises with pt   (rho ~ +1)
    early = (1 - pt) + rng.normal(0, 0.01, n)  # falls with pt   (rho ~ -1)
    return pd.DataFrame({"late": late, "early": early}), pt


# ── pure drawing library ─────────────────────────────────────────────────────


def test_rank_by_pseudotime_orders_by_abs_rho():
    df, pt = _monotone_activity()
    ranked = cascade.rank_by_pseudotime(df, pt)
    assert list(ranked["name"]) == ["late", "early"] or list(ranked["name"]) == ["early", "late"]
    rho = dict(zip(ranked["name"], ranked["rho"], strict=True))
    assert rho["late"] > 0.9
    assert rho["early"] < -0.9


def test_rank_by_pseudotime_flat_source_is_zero():
    n = 200
    pt = np.linspace(0, 1, n)
    df = pd.DataFrame({"flat": np.full(n, 3.0), "rising": pt})
    ranked = cascade.rank_by_pseudotime(df, pt)
    absr = dict(zip(ranked["name"], ranked["abs"], strict=True))
    assert absr["flat"] == 0.0
    assert absr["rising"] > 0.9


def test_center_of_mass_orders_early_to_late():
    mat = pd.DataFrame(
        [
            [3.0, 1.0, 0.0, 0.0],  # peaks early
            [0.0, 0.0, 1.0, 3.0],  # peaks late
            [0.0, 3.0, 1.0, 0.0],  # peaks mid-early
        ],
        index=["A_early", "C_late", "B_mid"],
        columns=[0, 1, 2, 3],
    )
    assert cascade.center_of_mass_order(mat) == ["A_early", "B_mid", "C_late"]


def test_binned_matrix_shape_and_direction():
    df, pt = _monotone_activity(n=120)
    mat = cascade.binned_matrix(df, pt, ["late", "early"], n_bins=10)
    assert mat.shape == (2, 10)
    assert mat.loc["late"].iloc[-1] > mat.loc["late"].iloc[0]
    assert mat.loc["early"].iloc[-1] < mat.loc["early"].iloc[0]


def test_binned_matrix_uses_rank_not_raw_value_so_skew_does_not_empty_most_bins():
    """Real bug found on the fibroblast cohort's activity_cascade figures
    (same root cause as the pseudotime heatmap's binning bug): dpt_pseudotime
    had 99.84% of cells packed into the first 5% of its raw value range, with
    a handful of straggler cells stretching the max to 1.0. Equal-WIDTH
    binning (np.linspace(pt.min(), pt.max(), n_bins+1)) left most bins
    completely empty (NaN) -- exactly the near-blank-heatmap-with-a-few-
    populated-strips a user actually saw. Binning by RANK instead must keep
    every bin populated regardless of skew.
    """
    n_dense, n_stray = 995, 5
    pt = np.concatenate([np.linspace(0.0, 0.03, n_dense), np.linspace(0.1, 1.0, n_stray)])
    df = pd.DataFrame({"src": np.random.RandomState(0).rand(len(pt))})
    mat = cascade.binned_matrix(df, pt, ["src"], n_bins=100)
    empty = mat.loc["src"].isna().sum()
    assert empty == 0, f"{empty} of 100 bins are empty (NaN)"


def test_clean_label_strips_hallmark_prefix():
    assert cascade.clean_label("HALLMARK_TNFA_SIGNALING_VIA_NFKB") == "Tnfa Signaling Via Nfkb"
    assert cascade.clean_label("EGFR") == "EGFR"


def test_cascade_heatmap_returns_figure_and_saves(tmp_path):
    df, pt = _monotone_activity()
    fig = cascade.cascade_heatmap(df, pt, top=None, n_bins=12, title="T")
    assert fig is not None
    paths = _helpers.save_figure(fig, tmp_path, "activity_cascade_test")
    assert {p.suffix for p in paths} == {".pdf", ".png"}
    for p in paths:
        assert Path(p).exists()


def test_cascade_heatmap_diagonal_row_order():
    df, pt = _monotone_activity()
    fig = cascade.cascade_heatmap(df, pt, top=None, n_bins=12)
    # labels use order[::-1], so the top row is the latest-peaking source.
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels[0] == "late"
    assert labels[-1] == "early"


def test_cascade_heatmap_none_when_no_association():
    n = 200
    pt = np.linspace(0, 1, n)
    df = pd.DataFrame({"a": np.full(n, 1.0), "b": np.full(n, 2.0)})
    assert cascade.cascade_heatmap(df, pt) is None


# ── method: registration + skip path + offline glue ──────────────────────────


def test_activity_cascade_registered():
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("trajectory_viz", "activity_cascade")


class _Ctx:
    def __init__(self, tmp_path: Path):
        class _Paths:
            figures = tmp_path / "figures"
            results = tmp_path / "results"

        self.paths = _Paths()


def test_activity_cascade_skips_without_pseudotime(tmp_path):
    adata = ad.AnnData(
        X=np.zeros((20, 5)),
        obs=pd.DataFrame({"cell_type": ["LEC"] * 20}),
    )
    result = ActivityCascadeVizMethod()._run(adata, {}, _Ctx(tmp_path))
    assert isinstance(result, MethodSkip)
    assert "pseudotime" in result.reason.lower()


def test_activity_cascade_offline_glue(tmp_path, monkeypatch):
    """End-to-end through decoupler with a hand-built net (no network)."""
    pytest.importorskip("decoupler")

    n, n_genes = 200, 20
    rng = np.random.default_rng(1)
    pt = np.linspace(0, 1, n)
    genes = [f"G{i}" for i in range(n_genes)]
    X = rng.normal(0.0, 0.05, (n, n_genes))
    # First 5 genes rise with pt (an "up" program), next 5 fall (a "down" program).
    for j in range(5):
        X[:, j] += pt
    for j in range(5, 10):
        X[:, j] += 1 - pt

    adata = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"palantir_pseudotime": pt}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["cellquorum_normalized"] = X.copy()

    net = pd.DataFrame(
        {
            "source": ["S_up"] * 5 + ["S_down"] * 5,
            "target": genes[:5] + genes[5:10],
            "weight": [1.0] * 10,
        }
    )
    monkeypatch.setattr(
        "cellquorum.stages.trajectory.viz._activity_cascade.get_net",
        lambda *a, **k: net,
    )

    config = {"activity_resources": ["myset"], "min_size": 5, "cascade_n_bins": 8}
    result = ActivityCascadeVizMethod()._run(adata, config, _Ctx(tmp_path))

    assert not isinstance(result, MethodSkip)
    assert result.metrics["resources"] == ["myset"]
    figs = [a for a in result.artifacts if a.kind == "figure"]
    assert figs, "expected cascade figure artifacts"
    suffixes = {Path(a.path).suffix for a in figs}
    assert suffixes == {".pdf", ".png"}
    for a in figs:
        assert Path(a.path).exists()


def test_activity_cascade_works_from_a_sparse_layer_without_manual_densification(
    tmp_path, monkeypatch
):
    """Per-cell activity scoring must pass the sparse layer straight to decoupler
    rather than manually building a dense pandas DataFrame from the full
    expression matrix first -- the same "12 GB in a minute, VM died" failure
    mode already fixed twice elsewhere (state_scoring/aucell_method.py,
    comparative/enrichment/activity_method.py). At real cohort scale
    (201,871 x 33,417) a manual dense conversion is ~54 GB; decoupler's own
    AnnData-input path handles a sparse layer natively via internal batching.

    This fixture is far too small to observe that memory difference directly
    (decoupler densifies small inputs internally regardless of caller intent --
    confirmed empirically, and orthogonal to what this test can check), so this
    checks the property that IS testable at this scale: the method never builds
    its own dense copy of the full gene space before calling decoupler, and
    still produces correct results from a sparse layer.
    """
    pytest.importorskip("decoupler")
    import scipy.sparse as sp

    n, n_genes = 200, 20
    rng = np.random.default_rng(1)
    pt = np.linspace(0, 1, n)
    genes = [f"G{i}" for i in range(n_genes)]
    X = (rng.random((n, n_genes)) + 0.1).astype(np.float32)
    for j in range(5):
        X[:, j] += pt
    for j in range(5, 10):
        X[:, j] += 1 - pt
    sparse_X = sp.csr_matrix(X)

    adata = ad.AnnData(
        X=sparse_X.copy(),
        obs=pd.DataFrame({"palantir_pseudotime": pt}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["cellquorum_normalized"] = sparse_X.copy()

    net = pd.DataFrame(
        {
            "source": ["S_up"] * 5 + ["S_down"] * 5,
            "target": genes[:5] + genes[5:10],
            "weight": [1.0] * 10,
        }
    )
    monkeypatch.setattr(
        "cellquorum.stages.trajectory.viz._activity_cascade.get_net",
        lambda *a, **k: net,
    )

    # The method itself must never call pd.DataFrame on a densified copy of the
    # full gene space -- only decoupler's own internals may (and, at this
    # fixture's scale, do).
    import pandas as real_pd

    original_dataframe_init = real_pd.DataFrame.__init__

    def _guarded_init(self, data=None, *args, **kwargs):
        if isinstance(data, np.ndarray) and data.shape == (n, n_genes):
            raise AssertionError(
                "activity_cascade built its own dense DataFrame from the FULL "
                "expression matrix instead of passing the sparse layer straight "
                "to decoupler."
            )
        return original_dataframe_init(self, data, *args, **kwargs)

    monkeypatch.setattr(real_pd.DataFrame, "__init__", _guarded_init)

    config = {"activity_resources": ["myset"], "min_size": 5, "cascade_n_bins": 8}
    result = ActivityCascadeVizMethod()._run(adata, config, _Ctx(tmp_path))

    assert not isinstance(result, MethodSkip)
    assert result.metrics["resources"] == ["myset"]
