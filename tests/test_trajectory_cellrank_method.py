"""CellRankMethod orchestration tests (writeback, subsample, skip-not-crash)."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory import _cellrank
from cellquorum.trajectory.cellrank_method import CellRankMethod

cr = pytest.importorskip("cellrank")


def _make_adata(n=300):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, 60)).astype("float32")
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(60)]
    a.obs["cell_type"] = pd.Categorical(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    a.obs["pseudotime"] = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=20)
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    return a


def _context(adata, tmp_path):
    paths = types.SimpleNamespace(results=str(tmp_path))
    return types.SimpleNamespace(require_adata=lambda: adata, paths=paths, config=None)


def _config(**over):
    base = {
        "cluster_key": "cell_type",
        "pseudotime_key": "pseudotime",
        "cytotrace_key": None,
        "use_rep": None,
        "use_rep_fallback": ["X_pca"],
        "n_neighbors": 15,
        "weight_connectivities": 0.2,
        "n_components": 10,
        "n_states": 3,
        "n_terminal_states": 2,
        "terminal_method": "stability",
        "predict_initial_states": False,
        "n_initial_states": 1,
        "max_cells": None,
        "seed": 0,
    }
    base.update(over)
    return base


def test_method_runs_and_writes_back(tmp_path):
    a = _make_adata()
    ctx = _context(a, tmp_path)
    result = CellRankMethod().run(a, _config(), ctx, donor_col=None)
    assert not isinstance(result, MethodSkip)
    assert "cellrank_macrostates" in result.adata.obs
    assert "cellrank_terminal_states" in result.adata.obs
    assert "cellrank_fate_probabilities" in result.adata.obsm
    uns = result.adata.uns["trajectory"]["cellrank"]
    assert uns["n_macrostates_actual"] == len(uns["macrostate_names"])
    assert uns["fate_names"]  # non-empty lineage order
    # obsm array columns match recorded lineage order.
    assert result.adata.obsm["cellrank_fate_probabilities"].shape[1] == len(uns["fate_names"])
    # An h5ad artifact was produced.
    assert any(art.kind == "h5ad" for art in result.artifacts)


def test_method_skips_when_cluster_key_absent(tmp_path):
    a = _make_adata()
    ctx = _context(a, tmp_path)
    result = CellRankMethod().run(a, _config(cluster_key="not_a_col"), ctx, donor_col=None)
    assert isinstance(result, MethodSkip)


def test_method_skips_on_import_unavailable(tmp_path, monkeypatch):
    a = _make_adata()
    ctx = _context(a, tmp_path)

    def _boom(*args, **kwargs):
        raise _cellrank.CellRankUnavailable("no cellrank")

    monkeypatch.setattr(_cellrank, "build_kernel", _boom)
    result = CellRankMethod().run(a, _config(), ctx, donor_col=None)
    assert isinstance(result, MethodSkip)


def test_method_skips_on_no_kernel_input(tmp_path, monkeypatch):
    a = _make_adata()
    ctx = _context(a, tmp_path)

    def _boom(*args, **kwargs):
        raise _cellrank.NoKernelInput("nothing")

    monkeypatch.setattr(_cellrank, "build_kernel", _boom)
    result = CellRankMethod().run(a, _config(), ctx, donor_col=None)
    assert isinstance(result, MethodSkip)


def test_method_skips_on_schur_failure(tmp_path, monkeypatch):
    a = _make_adata()
    ctx = _context(a, tmp_path)

    def _boom(*args, **kwargs):
        raise _cellrank.SchurFailed("schur")

    monkeypatch.setattr(_cellrank, "run_gpcca", _boom)
    result = CellRankMethod().run(a, _config(), ctx, donor_col=None)
    assert isinstance(result, MethodSkip)


def test_cellrank_passes_velocity_adata(tmp_path, monkeypatch):
    """With use_velocity + a present whole_object.h5ad, the loaded velocity
    object reaches build_kernel (and the new params are forwarded)."""
    a = _make_adata()
    ctx = _context(a, tmp_path)

    # Write a whole-object velocity h5ad where CellRankMethod looks for it.
    velo_dir = tmp_path / "trajectory" / "velocity"
    velo_dir.mkdir(parents=True, exist_ok=True)
    velo = a.copy()
    velo.layers["Ms"] = np.asarray(velo.X, dtype="float32").copy()
    velo.layers["velocity"] = np.asarray(velo.X, dtype="float32").copy()
    velo.write_h5ad(velo_dir / "whole_object.h5ad")

    captured: dict = {}
    _real_build_kernel = _cellrank.build_kernel

    def _spy_build_kernel(work, **kwargs):
        captured.update(kwargs)
        # Verify forwarding without depending on the dummy velocity layers
        # forming a valid transition matrix: build a clean connectivity kernel
        # (drop velocity_adata) so the downstream GPCCA chain still proceeds.
        clean = dict(kwargs)
        clean["velocity_adata"] = None
        return _real_build_kernel(work, **clean)

    monkeypatch.setattr(_cellrank, "build_kernel", _spy_build_kernel)

    cfg = _config(
        use_velocity=True,
        velocity_model="stochastic",
        time_key="some_stage",
        realtime_epsilon=0.25,
    )
    result = CellRankMethod().run(a, cfg, ctx, donor_col=None)

    assert not isinstance(result, MethodSkip)
    assert captured["velocity_adata"] is not None
    assert list(captured["velocity_adata"].obs_names) == list(a.obs_names)
    assert captured["velocity_model"] == "stochastic"
    assert captured["time_key"] == "some_stage"
    assert captured["realtime_epsilon"] == pytest.approx(0.25)


def test_cellrank_velocity_missing_h5ad_is_noted(tmp_path, monkeypatch):
    """use_velocity=True but no h5ad present → velocity_adata=None, no crash."""
    a = _make_adata()
    ctx = _context(a, tmp_path)

    captured: dict = {}
    _real_build_kernel = _cellrank.build_kernel

    def _spy_build_kernel(work, **kwargs):
        captured.update(kwargs)
        return _real_build_kernel(work, **kwargs)

    monkeypatch.setattr(_cellrank, "build_kernel", _spy_build_kernel)

    result = CellRankMethod().run(a, _config(use_velocity=True), ctx, donor_col=None)
    assert not isinstance(result, MethodSkip)
    assert captured["velocity_adata"] is None


def test_method_subsample_deterministic(tmp_path):
    a = _make_adata()
    ctx = _context(a, tmp_path)
    result = CellRankMethod().run(a, _config(max_cells=150, seed=0), ctx, donor_col=None)
    assert not isinstance(result, MethodSkip)
    uns = result.adata.uns["trajectory"]["cellrank"]
    assert uns["subsampled"] is True
    assert uns["n_cells_used"] == 150
    # Cells outside the sample are NaN in the fate-probability writeback.
    fp = result.adata.obsm["cellrank_fate_probabilities"]
    assert fp.shape[0] == a.n_obs
    assert np.isnan(fp).any()  # some rows unassigned
