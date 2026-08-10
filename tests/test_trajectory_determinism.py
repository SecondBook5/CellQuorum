from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scvelo")
pytest.importorskip("loompy")

from cellquorum.trajectory.velocity_method import VelocityMethod  # noqa: E402


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


def _build(tmp_path, seed):
    import loompy

    tmp_path.mkdir(parents=True, exist_ok=True)
    genes = [f"GENE_{i}" for i in range(60)]
    barcodes = [f"BC{i:04d}" for i in range(40)]
    names = [f"s1_{bc}-1" for bc in barcodes]
    rng = np.random.default_rng(0)
    a = ad.AnnData(
        X=rng.random((40, 60)).astype("float32"),
        obs=pd.DataFrame({"sample_id": ["s1"] * 40, "cell_type": ["T"] * 40}, index=names),
    )
    a.var_names = genes
    a.obsm["X_pca"] = rng.random((40, 10))
    loom = tmp_path / f"s1_{seed}.loom"
    n_g, n_c = len(genes), len(barcodes)
    loompy.create(
        str(loom),
        layers={
            "": rng.integers(0, 5, size=(n_g, n_c)).astype("float32"),
            "spliced": rng.integers(0, 5, size=(n_g, n_c)).astype("float32"),
            "unspliced": rng.integers(0, 3, size=(n_g, n_c)).astype("float32"),
        },
        row_attrs={"Gene": np.array(genes, dtype=object)},
        col_attrs={"CellID": np.array([f"s1:{bc}x" for bc in barcodes], dtype=object)},
    )
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(loom)]})
    return a, manifest


def _cfg():
    return {
        "grouping_col": "cell_type",
        "sample_col": "sample_id",
        "loom_path_col": "loom_path",
        "groups": None,
        "use_rep": None,
        "use_rep_fallback": ["X_pca"],
        "mode": "dynamical",
        "min_shared_counts": 0,
        "n_top_genes": 40,
        "n_pcs": 5,
        "n_neighbors": 5,
        "min_cells": 5,
        "n_jobs": 1,
        "seed": 7,
        "generation": {"generate_missing": False},
    }


def test_same_seed_same_pseudotime(tmp_path):
    a1, m1 = _build(tmp_path / "a", 1)
    a2, m2 = _build(tmp_path / "b", 2)
    r1 = VelocityMethod().run(a1, _cfg(), _Ctx(tmp_path / "a", a1, m1))
    r2 = VelocityMethod().run(a2, _cfg(), _Ctx(tmp_path / "b", a2, m2))

    # Guard against a silent skip masquerading as determinism: the "T" group must
    # have actually run velocity, not been skipped, on both invocations.
    for r in (r1, r2):
        statuses = [g["status"] for g in r.metrics["per_group"]]
        assert "success" in statuses, f"velocity did not run: {statuses}"

    # Both succeeded and recorded per-group metrics; pseudotime is written on the
    # per-group h5ad. Re-read the written objects and compare.
    import glob

    def _pt(root):
        files = sorted(glob.glob(str(root / "results" / "trajectory" / "velocity" / "*.h5ad")))
        assert files
        obj = ad.read_h5ad(files[0])
        return np.asarray(obj.obs["velocity_pseudotime"])

    # scvelo 0.3.4's EM fit has an inherent ~1e-6 float-nondeterminism floor that
    # no public seed controls (only the process-global numpy seed, which we set in
    # compute.compute_velocity). atol=1e-3 sits comfortably above that noise while
    # still catching a genuine unseeded regression (which drifts by ~1e-1). See the
    # velocity-determinism caveat in the trajectory README.
    np.testing.assert_allclose(_pt(tmp_path / "a"), _pt(tmp_path / "b"), rtol=0, atol=1e-3)
