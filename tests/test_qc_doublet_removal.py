"""QC honors doublets.remove: consensus doublets are dropped when configured."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.qc.stage import QCStage


class _Paths:
    def __init__(self, tmp):
        self.results = tmp


class _Ctx:
    def __init__(self, adata, config, paths):
        self._adata = adata
        self.config = config
        self.paths = paths
        self.backend_registry = None
        self.run_id = "test"
        self.random_seed = 0

    def require_adata(self):
        return self._adata


def _counts_adata(n=60):
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, size=(n, 8)).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=[f"G{i}" for i in range(8)]))
    a.layers["counts"] = x.copy()
    a.obs["sample_id"] = "S1"
    return a


def test_doublets_removed_when_remove_true(tmp_path):
    a = _counts_adata()
    config = {
        "qc": {
            "enabled": True,
            "mode": "flag_no_drop",
            "metrics": {"layer": "counts"},
            "doublets": {
                "enabled": True,
                "method": "scrublet",
                "methods": ["scrublet"],
                "consensus": "any",
                "remove": True,
            },
            "ambient": {"enabled": False, "method": "none"},
        }
    }
    ctx = _Ctx(a, config, _Paths(tmp_path))
    # Force a deterministic predicted_doublet column by pre-seeding it: the
    # remover must act on whatever detect_doublets leaves in obs. We simulate a
    # detector result by monkeypatching detect_doublets to flag the first 5 cells.
    import cellquorum.stages.qc.doublets as doublets_mod

    def _fake_detect(adata, cfg, backend, sample_key=None, n_jobs=1):
        flags = np.zeros(adata.n_obs, dtype=bool)
        flags[:5] = True
        adata.obs["predicted_doublet"] = flags
        adata.obs["doublet_score"] = np.linspace(0, 1, adata.n_obs)
        return {"n_doublets": 5, "method": "scrublet"}

    orig = doublets_mod.detect_doublets
    try:
        # detect_doublets is imported lazily inside the stage from the module,
        # so patch the source module attribute.
        doublets_mod.detect_doublets = _fake_detect
        result = QCStage().run(ctx)
    finally:
        doublets_mod.detect_doublets = orig

    assert result.adata.n_obs == 55
    assert bool(result.adata.obs.get("predicted_doublet", pd.Series([], dtype=bool)).any()) is False
    assert result.metrics["doublets"]["n_removed"] == 5


def test_doublets_kept_when_remove_false(tmp_path):
    a = _counts_adata()
    config = {
        "qc": {
            "enabled": True,
            "mode": "flag_no_drop",
            "metrics": {"layer": "counts"},
            "doublets": {
                "enabled": True,
                "method": "scrublet",
                "methods": ["scrublet"],
                "consensus": "any",
                "remove": False,
            },
            "ambient": {"enabled": False, "method": "none"},
        }
    }
    ctx = _Ctx(a, config, _Paths(tmp_path))
    import cellquorum.stages.qc.doublets as doublets_mod

    def _fake_detect(adata, cfg, backend, sample_key=None, n_jobs=1):
        flags = np.zeros(adata.n_obs, dtype=bool)
        flags[:5] = True
        adata.obs["predicted_doublet"] = flags
        adata.obs["doublet_score"] = np.linspace(0, 1, adata.n_obs)
        return {"n_doublets": 5, "method": "scrublet"}

    orig = doublets_mod.detect_doublets
    try:
        doublets_mod.detect_doublets = _fake_detect
        result = QCStage().run(ctx)
    finally:
        doublets_mod.detect_doublets = orig

    # remove=False: all cells retained, flag preserved.
    assert result.adata.n_obs == 60
    assert int(result.adata.obs["predicted_doublet"].sum()) == 5
