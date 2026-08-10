"""End-to-end chain: whole-object velocity producer → CellRank VelocityKernel.

Proves the cross-method contract wired across Tasks 2/4/6: VelocityMethod with
``whole_object=True`` writes ``whole_object.h5ad``, and CellRankMethod with
``use_velocity=True`` loads it and folds a VelocityKernel into the combined
kernel (recorded in ``uns["trajectory"]["cellrank"]["kernel"]["kernels"]``).

Import-gated on scvelo + cellrank. Real scVelo on synthetic data is fragile, so
``compute.compute_velocity`` is monkeypatched to stamp ``Ms``+``velocity``
layers (the whole_object.h5ad is a real on-disk object, just cheap to produce),
and ``reconcile_looms`` is stubbed so no real velocyto loom is required.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

pytest.importorskip("scvelo")
pytest.importorskip("cellrank")

from cellquorum.methods.base import MethodSkip  # noqa: E402
from cellquorum.trajectory import compute  # noqa: E402
from cellquorum.trajectory.cellrank_method import CellRankMethod  # noqa: E402
from cellquorum.trajectory.velocity_method import VelocityMethod  # noqa: E402


def _make_adata(n=120):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, 40)).astype("float32")
    a = ad.AnnData(X)
    a.obs_names = [f"s1_{i:04d}-1" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(40)]
    a.obs["sample_id"] = "s1"
    a.obs["cell_type"] = pd.Categorical(
        ["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n - 2 * (n // 3))
    )
    a.obs["pseudotime"] = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=20)
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    return a


class _Paths:
    def __init__(self, tmp):
        self.results = str(tmp / "results")
        Path(self.results).mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp, adata, manifest):
        self.paths = _Paths(tmp)
        self._adata = adata
        self._manifest = manifest
        self.config = None

    def require_adata(self):
        return self._adata

    def require_manifest(self):
        if self._manifest is None:
            raise RuntimeError("no manifest")
        return self._manifest


def _stamp_velocity(sub, **kwargs):
    """Cheap stand-in for compute.compute_velocity: stamp Ms + velocity layers."""
    sub.layers["Ms"] = np.asarray(sub.X, dtype="float32").copy()
    sub.layers["velocity"] = np.asarray(sub.X, dtype="float32").copy()
    sub.obs["velocity_pseudotime"] = np.linspace(0.0, 1.0, sub.n_obs)
    sub.obs["velocity_confidence"] = np.full(sub.n_obs, 0.5)


def _velocity_config(**over):
    base = {
        "grouping_col": "cell_type",
        "sample_col": "sample_id",
        "loom_path_col": "loom_path",
        "groups": None,
        "use_rep": "X_pca",
        "use_rep_fallback": ["X_pca"],
        "mode": "dynamical",
        "min_shared_counts": 0,
        "n_top_genes": 5,
        "n_pcs": 5,
        "n_neighbors": 5,
        "min_cells": 1,
        "n_jobs": 1,
        "seed": 0,
        "whole_object": True,
        "generation": {"generate_missing": False},
    }
    base.update(over)
    return base


def _cellrank_config(**over):
    base = {
        "cluster_key": "cell_type",
        "pseudotime_key": None,
        "cytotrace_key": None,
        "use_velocity": True,
        "velocity_model": "deterministic",
        "time_key": None,
        "realtime_epsilon": 0.1,
        "use_rep": "X_pca",
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


def test_velocity_to_cellrank_velocity_kernel_chain(tmp_path, monkeypatch):
    a = _make_adata()

    # Stub loom I/O + velocity compute so the whole_object.h5ad is real but cheap.
    velo_layered = a.copy()
    monkeypatch.setattr(
        "cellquorum.trajectory.velocity_method.reconcile_looms",
        lambda adata, manifest, **k: (velo_layered, ["stubbed looms"]),
    )
    monkeypatch.setattr(compute, "compute_velocity", _stamp_velocity)
    monkeypatch.setattr(compute, "reproject_velocity", lambda adata, *, bases: [])

    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    ctx = _Ctx(tmp_path, a, manifest=manifest)

    # 1. Producer: whole-object velocity → whole_object.h5ad.
    velo_result = VelocityMethod().run(a, _velocity_config(), ctx, donor_col=None)
    assert not isinstance(velo_result, MethodSkip)
    whole = Path(ctx.paths.results) / "trajectory" / "velocity" / "whole_object.h5ad"
    assert whole.exists(), "producer did not write whole_object.h5ad"

    # 2. Consumer: CellRank loads it and builds a VelocityKernel.
    cr_result = CellRankMethod().run(a, _cellrank_config(), ctx, donor_col=None)
    assert not isinstance(cr_result, MethodSkip), getattr(cr_result, "reason", "")

    kernels = cr_result.adata.uns["trajectory"]["cellrank"]["kernel"]["kernels"]
    assert "velocity" in kernels, f"VelocityKernel not folded in; kernels={kernels}"
    weights = cr_result.adata.uns["trajectory"]["cellrank"]["kernel"]["weights"]
    assert weights["velocity"] == pytest.approx(0.8)
    assert weights["connectivity"] == pytest.approx(0.2)
