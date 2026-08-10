"""End-to-end: DPT produces pseudotime, CellRank consumes it, through the stage."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

pytest.importorskip("cellrank")

from cellquorum.core.stage import StageResult  # noqa: E402
from cellquorum.trajectory.config import (  # noqa: E402
    CellRankConfig,
    DptConfig,
    TrajectoryConfig,
)
from cellquorum.trajectory.stage import TrajectoryStage  # noqa: E402


def _make_adata(n=250):
    """Build synthetic data: smooth pseudotime trajectory + clusters + stem score."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, n)
    # Low-D smooth trajectory in gene space.
    base = np.outer(t, rng.normal(size=30)) + rng.normal(scale=0.1, size=(n, 30))
    a = ad.AnnData(base.astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(30)]
    a.obs["cell_type"] = pd.Categorical(["A"] * 80 + ["B"] * 90 + ["C"] * 80)
    a.obs["stem_score"] = 1.0 - t  # Stem root at low t.
    sc.pp.pca(a, n_comps=15)
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    return a


def test_dpt_then_cellrank_chain(tmp_path):
    """DPT writes dpt_pseudotime; CellRank's PseudotimeKernel consumes it."""
    a = _make_adata()
    traj = TrajectoryConfig(
        methods=[{"method": "dpt"}, {"method": "cellrank"}],
        dpt=DptConfig(
            use_rep="X_pca",
            use_rep_fallback=["X_pca"],
            n_comps=10,
            root_marker_score_key="stem_score",
            seed=0,
        ),
        cellrank=CellRankConfig(
            pseudotime_key="dpt_pseudotime",
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
    # Producer wrote the pseudotime column…
    assert "dpt_pseudotime" in result.adata.obs
    # …and the consumer used it (PseudotimeKernel over dpt_pseudotime).
    assert "cellrank_macrostates" in result.adata.obs
    assert "cellrank_fate_probabilities" in result.adata.obsm
