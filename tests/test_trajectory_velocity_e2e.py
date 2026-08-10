"""Skippable real-loom integration test (mirrors the SoupX integration gate).

Runs the full velocity pipeline on a real loom when TRAJECTORY_TEST_LOOM (a loom
path) and TRAJECTORY_TEST_H5AD (a matching atlas .h5ad) env vars are set and
scvelo/loompy are installed. Skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

scvelo = pytest.importorskip("scvelo")
loompy = pytest.importorskip("loompy")

LOOM = os.environ.get("TRAJECTORY_TEST_LOOM")
H5AD = os.environ.get("TRAJECTORY_TEST_H5AD")
SAMPLE = os.environ.get("TRAJECTORY_TEST_SAMPLE", "s1")

pytestmark = pytest.mark.skipif(
    not (LOOM and H5AD and Path(LOOM).exists() and Path(H5AD).exists()),
    reason="set TRAJECTORY_TEST_LOOM + TRAJECTORY_TEST_H5AD to run the real-data test",
)


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
        return self._manifest


def test_real_loom_velocity_pipeline(tmp_path):
    from cellquorum.trajectory.velocity_method import VelocityMethod

    adata = ad.read_h5ad(H5AD)
    grouping = "cell_type" if "cell_type" in adata.obs else adata.obs.columns[0]
    manifest = pd.DataFrame({"sample_id": [SAMPLE], "loom_path": [LOOM]})
    cfg = {
        "grouping_col": grouping,
        "sample_col": "sample_id",
        "loom_path_col": "loom_path",
        "groups": None,
        "use_rep": None,
        "use_rep_fallback": ["X_scANVI", "X_scVI", "X_pca"],
        "mode": "dynamical",
        "min_shared_counts": 20,
        "n_top_genes": 2000,
        "n_pcs": 30,
        "n_neighbors": 30,
        "min_cells": 30,
        "n_jobs": 1,
        "seed": 1337,
        "generation": {"generate_missing": False},
    }
    VelocityMethod().run(adata, cfg, _Ctx(tmp_path, adata, manifest))
    # At least one group should have produced a velocity object with a graph.
    files = sorted((tmp_path / "results" / "trajectory" / "velocity").glob("*.h5ad"))
    assert files, "no velocity h5ad produced"
    obj = ad.read_h5ad(files[0])
    assert "velocity_graph" in obj.uns
