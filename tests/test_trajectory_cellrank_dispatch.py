"""End-to-end dispatch of the cellrank method through TrajectoryStage."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellquorum.core.stage import StageResult
from cellquorum.stages.trajectory.config import CellRankConfig, TrajectoryConfig
from cellquorum.stages.trajectory.stage import TrajectoryStage

pytest.importorskip("cellrank")


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


def test_stage_dispatches_cellrank(tmp_path):
    a = _make_adata()
    traj = TrajectoryConfig(
        methods=[{"method": "cellrank"}],
        cellrank=CellRankConfig(
            pseudotime_key="pseudotime",
            use_rep_fallback=["X_pca"],
            n_neighbors=15,
            n_components=10,
            n_states=3,
            n_terminal_states=2,
            seed=0,
        ),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    paths = types.SimpleNamespace(results=str(tmp_path))
    context = types.SimpleNamespace(
        require_adata=lambda: a, config=config, paths=paths, donor_col=None
    )

    result = TrajectoryStage().run(context)
    assert isinstance(result, StageResult)
    assert "cellrank_macrostates" in result.adata.obs
    assert "cellrank_fate_probabilities" in result.adata.obsm
