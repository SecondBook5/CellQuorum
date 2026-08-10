# tests/test_trajectory_velocity_method.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.trajectory.velocity_method import VelocityMethod


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


def _atlas():
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    names = [f"s1_{bc}-1" for bc in ("AAAA", "CCCC")]
    a = ad.AnnData(
        X=np.ones((2, 3), dtype="float32"),
        obs=pd.DataFrame({"sample_id": ["s1", "s1"], "cell_type": ["T", "T"]}, index=names),
    )
    a.var_names = genes
    a.obsm["X_pca"] = np.ones((2, 3))
    return a


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
        "generation": {"generate_missing": False},
    }
    base.update(over)
    return base


def test_velocity_skips_when_no_manifest(tmp_path):
    ctx = _Ctx(tmp_path, _atlas(), manifest=None)
    result = VelocityMethod().run(_atlas(), _cfg(), ctx)
    assert isinstance(result, MethodSkip)
    assert "spliced" in result.reason or "manifest" in result.reason.lower()


def test_velocity_skips_when_no_loom_and_generation_off(tmp_path):
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "absent.loom")]})
    ctx = _Ctx(tmp_path, _atlas(), manifest=manifest)
    result = VelocityMethod().run(_atlas(), _cfg(), ctx)
    assert isinstance(result, MethodSkip)


def test_velocity_end_to_end_with_looms(tmp_path):
    import pytest

    pytest.importorskip("scvelo")
    pytest.importorskip("loompy")
    import loompy

    # Build a loom whose barcodes match the atlas, enough genes/cells for scVelo.
    atlas_genes = [f"GENE_{i}" for i in range(50)]
    barcodes = [f"{b}{i:04d}" for i, b in enumerate(["AAAA"] * 40)]
    names = [f"s1_{bc}-1" for bc in barcodes]
    rng = np.random.default_rng(0)
    a = ad.AnnData(
        X=rng.random((40, 50)).astype("float32"),
        obs=pd.DataFrame({"sample_id": ["s1"] * 40, "cell_type": ["T"] * 40}, index=names),
    )
    a.var_names = atlas_genes
    a.obsm["X_pca"] = rng.random((40, 10))
    a.obsm["X_umap"] = rng.random((40, 2))

    loom = tmp_path / "s1.loom"
    n_g, n_c = len(atlas_genes), len(barcodes)
    loompy.create(
        str(loom),
        layers={
            "": rng.integers(0, 5, size=(n_g, n_c)).astype("float32"),
            "spliced": rng.integers(0, 5, size=(n_g, n_c)).astype("float32"),
            "unspliced": rng.integers(0, 3, size=(n_g, n_c)).astype("float32"),
        },
        row_attrs={"Gene": np.array(atlas_genes, dtype=object)},
        col_attrs={"CellID": np.array([f"s1:{bc}x" for bc in barcodes], dtype=object)},
    )
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(loom)]})
    ctx = _Ctx(tmp_path, a, manifest=manifest)
    result = VelocityMethod().run(a, _cfg(n_top_genes=30, n_pcs=5, n_neighbors=5, min_cells=5), ctx)
    # A StageResult (not a skip) with per-group metrics recorded.
    assert not isinstance(result, MethodSkip)
    assert "per_group" in result.metrics
