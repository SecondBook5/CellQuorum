# tests/test_trajectory_velocity_whole_object.py
"""Whole-object velocity producer: save writer + VelocityMethod branch."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory import compute
from cellquorum.trajectory.save import write_whole_object_velocity_h5ad
from cellquorum.trajectory.velocity_method import VelocityMethod


def _adata(n=40, g=15):
    rng = np.random.default_rng(0)
    a = ad.AnnData(
        X=rng.poisson(1.0, (n, g)).astype("float32"),
        obs=pd.DataFrame(
            {"sample_id": ["s1"] * n, "cell_type": ["T"] * n},
            index=[f"s1_{i:04d}-1" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"g{j}" for j in range(g)]),
    )
    a.obsm["X_pca"] = rng.normal(0, 1, (n, 10)).astype("float32")
    return a


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp, adata, manifest):
        self.paths = _Paths(tmp)
        self._adata = adata
        self._manifest = manifest

    def require_adata(self):
        return self._adata

    def require_manifest(self):
        if self._manifest is None:
            raise RuntimeError("no manifest")
        return self._manifest


def test_write_whole_object_velocity_h5ad(tmp_path: Path):
    a = _adata()
    a.layers["Ms"] = a.X.copy()
    a.layers["velocity"] = a.X.copy()
    art, note = write_whole_object_velocity_h5ad(a, tmp_path)
    assert art is not None
    assert art.path.exists()
    assert art.path.name == "whole_object.h5ad"
    back = ad.read_h5ad(art.path)
    assert "Ms" in back.layers
    assert "velocity" in back.layers


def _cfg(**over):
    base = {
        "grouping_col": "cell_type",
        "sample_col": "sample_id",
        "loom_path_col": "loom_path",
        "groups": None,
        "use_rep": None,
        "use_rep_fallback": ["X_pca"],
        "mode": "dynamical",
        "min_shared_counts": 0,
        "n_top_genes": 3,
        "n_pcs": 2,
        "n_neighbors": 2,
        "min_cells": 1,
        "n_jobs": 1,
        "seed": 0,
        "whole_object": False,
        "generation": {"generate_missing": False},
    }
    base.update(over)
    return base


def _stamp_velocity(sub, **kwargs):
    """Cheap stand-in for compute.compute_velocity: stamp Ms + velocity layers."""
    sub.layers["Ms"] = np.asarray(sub.X, dtype="float32").copy()
    sub.layers["velocity"] = np.asarray(sub.X, dtype="float32").copy()
    sub.obs["velocity_pseudotime"] = np.linspace(0.0, 1.0, sub.n_obs)
    sub.obs["velocity_confidence"] = np.full(sub.n_obs, 0.5)


def _patch_looms(monkeypatch, layered: ad.AnnData):
    """Make reconcile_looms return a layered velo_adata without real looms."""
    monkeypatch.setattr(
        "cellquorum.trajectory.velocity_method.reconcile_looms",
        lambda adata, manifest, **k: (layered, ["stubbed looms"]),
    )


def test_velocity_method_whole_object_writes_h5ad(tmp_path, monkeypatch):
    a = _adata()
    # velo_adata carries spliced/unspliced conceptually; here the stub compute
    # stamps Ms+velocity so no real scVelo runs.
    velo = a.copy()
    _patch_looms(monkeypatch, velo)
    monkeypatch.setattr(compute, "compute_velocity", _stamp_velocity)
    monkeypatch.setattr(compute, "reproject_velocity", lambda adata, *, bases: [])

    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    ctx = _Ctx(tmp_path, a, manifest=manifest)

    n_vars_before = a.n_vars
    result = VelocityMethod().run(a, _cfg(whole_object=True), ctx)

    assert not isinstance(result, MethodSkip)
    # The whole-object velocity h5ad was written.
    whole = tmp_path / "results" / "trajectory" / "velocity" / "whole_object.h5ad"
    assert whole.exists(), "whole_object.h5ad not written"
    back = ad.read_h5ad(whole)
    assert "Ms" in back.layers and "velocity" in back.layers
    # The working atlas var set is unchanged (no _inplace_subset_var leak).
    assert a.n_vars == n_vars_before


def test_velocity_method_whole_object_off_writes_nothing(tmp_path, monkeypatch):
    a = _adata()
    velo = a.copy()
    _patch_looms(monkeypatch, velo)
    monkeypatch.setattr(compute, "compute_velocity", _stamp_velocity)
    monkeypatch.setattr(compute, "reproject_velocity", lambda adata, *, bases: [])

    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    ctx = _Ctx(tmp_path, a, manifest=manifest)

    VelocityMethod().run(a, _cfg(whole_object=False), ctx)
    whole = tmp_path / "results" / "trajectory" / "velocity" / "whole_object.h5ad"
    assert not whole.exists(), "whole_object.h5ad written despite whole_object=False"
