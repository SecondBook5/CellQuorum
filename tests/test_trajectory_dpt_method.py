"""DptMethod orchestration tests."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.core.stage import StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory import _pseudotime
from cellquorum.trajectory.dpt_method import DptMethod


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


def test_dpt_writes_pseudotime(tmp_path):
    a = _adata()
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "n_comps": 10,
        "root_marker_score_key": "stem_score",
        "seed": 0,
    }
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, StageResult)
    assert "dpt_pseudotime" in res.adata.obs
    assert res.adata.uns["trajectory"]["dpt"]["root_source"] == "marker_score"


def test_dpt_skips_without_root(tmp_path):
    a = _adata()
    cfg = {"use_rep": "X_pca", "use_rep_fallback": ["X_pca"]}
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_dpt_skips_when_no_rep(tmp_path):
    a = _adata()
    del a.obsm["X_pca"]
    cfg = {
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_scVI"],
        "root_marker_score_key": "stem_score",
    }
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_dpt_skips_on_compute_failure(tmp_path, monkeypatch):
    a = _adata()
    monkeypatch.setattr(
        _pseudotime,
        "compute_dpt",
        lambda *x, **k: (_ for _ in ()).throw(_pseudotime.PseudotimeFailed("boom")),
    )
    cfg = {"use_rep": "X_pca", "use_rep_fallback": ["X_pca"], "root_marker_score_key": "stem_score"}
    res = DptMethod()._run(a, cfg, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)
