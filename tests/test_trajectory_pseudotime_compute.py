"""Compute-helper tests for the pseudotime methods."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

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
