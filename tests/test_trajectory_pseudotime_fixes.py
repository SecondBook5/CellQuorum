"""Regression tests for final-review findings (skip-not-crash, config isolation)."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cellquorum.core.stage import StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory import _pseudotime
from cellquorum.trajectory.config import CellRankConfig, DptConfig, TrajectoryConfig
from cellquorum.trajectory.dpt_method import DptMethod
from cellquorum.trajectory.stage import TrajectoryStage


def _adata(n=200):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, n)
    base = np.outer(t, rng.normal(size=20)) + rng.normal(scale=0.1, size=(n, 20))
    a = ad.AnnData(base.astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(20)]
    sc.pp.pca(a, n_comps=10)
    a.obs["stem_score"] = 1.0 - t
    return a


def _ctx(a, tmp_path):
    config = types.SimpleNamespace()
    paths = types.SimpleNamespace(results=str(tmp_path))
    return types.SimpleNamespace(
        require_adata=lambda: a, config=config, paths=paths, donor_col=None
    )


# ---- Finding #1: non-numeric obs column must not escape as a plain ValueError ---- #


def test_resolve_root_non_numeric_marker_is_recoverable():
    """A non-numeric marker-score column raises a recoverable PseudotimeComputeError."""
    a = ad.AnnData(np.zeros((5, 3), dtype="float32"))
    a.obsm["X_pca"] = np.random.default_rng(0).normal(size=(5, 4))
    a.obs["celltype"] = pd.Categorical(["A", "B", "A", "C", "B"])
    try:
        _pseudotime.resolve_root(
            a, rep="X_pca", marker_score_key="celltype", root_key=None, root_group=None
        )
    except _pseudotime.PseudotimeComputeError:
        pass  # recoverable → will become a MethodSkip
    else:
        raise AssertionError("expected a PseudotimeComputeError, not success")


def test_dpt_skips_on_non_numeric_marker(tmp_path):
    """A non-numeric marker column degrades to MethodSkip, never a stage crash."""
    a = _adata()
    a.obs["bad_marker"] = pd.Categorical(["x", "y"] * (a.n_obs // 2))
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "root_marker_score_key": "bad_marker",
    }
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_dpt_non_numeric_orient_does_not_crash(tmp_path):
    """A non-numeric orient_by_score_key is ignored (no reorientation), not a crash."""
    a = _adata()
    a.obs["bad_orient"] = pd.Categorical(["x", "y"] * (a.n_obs // 2))
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "n_comps": 10,
        "root_marker_score_key": "stem_score",
        "orient_by_score_key": "bad_orient",
    }
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, StageResult)
    assert res.adata.uns["trajectory"]["dpt"]["oriented"] is False


# ---- Finding #2: multi-method chains must not share one flattened namespace ---- #


def test_multi_method_chain_flattens_per_entry():
    """In a [dpt, cellrank] chain each method reads ITS OWN block, no key collision."""
    traj = TrajectoryConfig(
        methods=[{"method": "dpt"}, {"method": "cellrank"}],
        dpt=DptConfig(n_comps=7, use_rep="X_pca", root_marker_score_key="stem_score"),
        cellrank=CellRankConfig(n_components=20, use_rep=None, pseudotime_key="dpt_pseudotime"),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)
    augmented = TrajectoryStage()._augment_config(
        context, {"methods": [{"method": "dpt"}, {"method": "cellrank"}]}
    )
    entries = {e["method"]: e for e in augmented["methods"]}
    # dpt keeps its own use_rep (previously blocked by cellrank's None at the shared level).
    assert entries["dpt"]["use_rep"] == "X_pca"
    assert entries["dpt"]["n_comps"] == 7
    # cellrank keeps its own Schur-vector count (a DIFFERENT meaning of n_components).
    assert entries["cellrank"]["n_components"] == 20
    # The colliding keys never live at the shared top level in a chain.
    assert "n_components" not in augmented
    assert "use_rep" not in augmented


# ---- Finding #3: outlier exclusion must rebuild the neighbor graph ---- #


def test_exclude_outliers_rebuilds_neighbor_graph(tmp_path, monkeypatch):
    """The outlier-free subset drops the parent's sliced graph so DPT rebuilds clean."""
    a = _adata()
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    assert "neighbors" in a.uns
    # Inject one blatant outlier so flag_outliers fires.
    a.obsm["X_pca"][0] = float(np.abs(a.obsm["X_pca"]).max()) * 1000.0

    captured: dict = {}

    def fake_compute_dpt(work, **kwargs):
        captured["has_neighbors"] = "neighbors" in work.uns
        captured["has_conn"] = "connectivities" in work.obsp
        return {
            "pseudotime": np.zeros(work.n_obs, dtype="float64"),
            "n_dcs": int(kwargs["n_dcs"]),
            "notes": [],
        }

    monkeypatch.setattr(_pseudotime, "compute_dpt", fake_compute_dpt)
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "root_marker_score_key": "stem_score",
        "exclude_outliers": True,
        "outlier_mad": 3.0,
    }
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, StageResult)
    assert captured["has_neighbors"] is False
    assert captured["has_conn"] is False
