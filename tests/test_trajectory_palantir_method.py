"""PalantirMethod orchestration tests."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

pytest.importorskip("palantir")

from cellquorum.core.stage import StageResult  # noqa: E402
from cellquorum.methods.base import MethodSkip  # noqa: E402
from cellquorum.stages.trajectory import _pseudotime  # noqa: E402
from cellquorum.stages.trajectory.palantir_method import PalantirMethod  # noqa: E402


def _adata(n=200):
    rng = np.random.default_rng(2)
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


def test_palantir_writes_pseudotime(tmp_path):
    a = _adata()
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "n_components": 10,
        "knn": 30,
        "n_eigs": 10,
        "num_waypoints": 50,
        "root_marker_score_key": "stem_score",
        "seed": 0,
    }
    res = PalantirMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, StageResult)
    assert "palantir_pseudotime" in res.adata.obs
    assert "palantir_entropy" in res.adata.obs


def test_palantir_skips_without_root(tmp_path):
    a = _adata()
    cfg = {"use_rep": "X_pca", "use_rep_fallback": ["X_pca"], "num_waypoints": 50}
    res = PalantirMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_palantir_subsample_nan_outside(tmp_path):
    a = _adata(n=200)
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "n_components": 10,
        "knn": 30,
        "n_eigs": 10,
        "num_waypoints": 40,
        "max_cells": 120,
        "root_marker_score_key": "stem_score",
        "seed": 0,
    }
    res = PalantirMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, StageResult)
    pt_col = res.adata.obs["palantir_pseudotime"].to_numpy()
    assert np.isnan(pt_col).sum() > 0  # unsampled cells are NaN
    assert np.isfinite(pt_col).sum() <= 120


def test_palantir_skips_on_compute_failure(tmp_path, monkeypatch):
    a = _adata()
    monkeypatch.setattr(
        _pseudotime,
        "compute_palantir",
        lambda *x, **k: (_ for _ in ()).throw(_pseudotime.PseudotimeFailed("boom")),
    )
    cfg = {"use_rep": "X_pca", "use_rep_fallback": ["X_pca"], "root_marker_score_key": "stem_score"}
    res = PalantirMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)
