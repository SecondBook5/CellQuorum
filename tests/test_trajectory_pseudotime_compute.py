"""Compute-helper tests for the pseudotime methods."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellquorum.trajectory import _pseudotime as pt


def _adata(n=120, d=10):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.normal(size=(n, 6)).astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obsm["X_pca"] = rng.normal(size=(n, d)).astype("float32")
    a.obs["stem_score"] = np.linspace(0.0, 1.0, n)
    a.obs["grp"] = pd.Categorical(["root"] * 20 + ["mid"] * 60 + ["tip"] * 40)
    return a


def test_resolve_rep_prefers_configured_then_fallback():
    a = _adata()
    assert pt.resolve_rep(a, "X_pca", ["X_scVI"]) == "X_pca"
    assert pt.resolve_rep(a, None, ["X_scVI", "X_pca"]) == "X_pca"
    assert pt.resolve_rep(a, None, ["X_scVI"]) is None


def test_resolve_root_by_marker_score_is_argmax():
    a = _adata()
    root = pt.resolve_root(
        a, rep="X_pca", marker_score_key="stem_score", root_key=None, root_group=None
    )
    assert root == int(np.argmax(a.obs["stem_score"].to_numpy()))


def test_resolve_root_by_group_returns_index_in_group():
    a = _adata()
    root = pt.resolve_root(a, rep="X_pca", marker_score_key=None, root_key="grp", root_group="root")
    assert a.obs["grp"].to_numpy()[root] == "root"


def test_resolve_root_unresolved_raises():
    a = _adata()
    with pytest.raises(pt.RootUnresolved):
        pt.resolve_root(a, rep="X_pca", marker_score_key=None, root_key=None, root_group=None)


def test_resolve_root_missing_group_raises():
    a = _adata()
    with pytest.raises(pt.RootUnresolved):
        pt.resolve_root(a, rep="X_pca", marker_score_key=None, root_key="grp", root_group="absent")


def test_flag_outliers_shape_and_dtype():
    a = _adata()
    mask = pt.flag_outliers(a, "X_pca", 5.0)
    assert mask.shape == (a.n_obs,)
    assert mask.dtype == bool


def test_error_hierarchy():
    for sub in (
        pt.PseudotimeUnavailable,
        pt.NoRepresentation,
        pt.RootUnresolved,
        pt.DiffmapFailed,
        pt.PseudotimeFailed,
    ):
        assert issubclass(sub, pt.PseudotimeComputeError)


def _linear_adata(n=200):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, n)
    # 20 genes whose expression ramps along t (a clean 1-D trajectory)
    base = np.outer(t, rng.normal(size=20)) + rng.normal(scale=0.1, size=(n, 20))
    a = ad.AnnData(base.astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(20)]
    sc.pp.pca(a, n_comps=10)
    a.obs["stem_score"] = 1.0 - t  # high at the t=0 end
    return a


def test_compute_dpt_writes_pseudotime():
    a = _linear_adata()
    iroot = int(np.argmax(a.obs["stem_score"].to_numpy()))
    res = pt.compute_dpt(
        a,
        use_rep="X_pca",
        use_rep_fallback=["X_pca"],
        n_neighbors=15,
        n_comps=10,
        n_dcs=10,
        n_branchings=0,
        iroot=iroot,
    )
    assert res["pseudotime"].shape[0] == a.n_obs
    assert "dpt_pseudotime" in a.obs
    finite = np.isfinite(res["pseudotime"])
    assert finite.sum() > 0


def test_compute_dpt_diffmap_failure_retyped(monkeypatch):
    a = _linear_adata()
    monkeypatch.setattr(
        sc.tl, "diffmap", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(pt.DiffmapFailed):
        pt.compute_dpt(
            a,
            use_rep="X_pca",
            use_rep_fallback=["X_pca"],
            n_neighbors=15,
            n_comps=10,
            n_dcs=10,
            n_branchings=0,
            iroot=0,
        )


def test_compute_palantir_runs_three_steps():
    pytest.importorskip("palantir")
    a = _linear_adata()
    root = a.obs_names[int(np.argmax(a.obs["stem_score"].to_numpy()))]
    res = pt.compute_palantir(
        a,
        use_rep="X_pca",
        use_rep_fallback=["X_pca"],
        n_components=10,
        knn=30,
        n_eigs=10,
        num_waypoints=50,
        early_cell=root,
        seed=0,
    )
    assert len(res["pseudotime"]) == a.n_obs
    assert len(res["entropy"]) == a.n_obs
    assert "DM_EigenVectors_multiscaled" in a.obsm
