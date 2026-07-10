"""QC stage invokes doublet + cell-cycle add-ons when enabled."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.qc.cell_cycle import TIROSH_G2M_GENES, TIROSH_S_GENES
from cellquorum.qc.config import QCConfig
from cellquorum.qc.stage import QCStage


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    # Include more Tirosh genes + background genes for proper scoring.
    s = TIROSH_S_GENES[:15]
    g2m = TIROSH_G2M_GENES[:15]
    genes = s + g2m + [f"G{i}" for i in range(200)]  # More genes for control set.
    n = 120
    x = rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.layers["counts"] = x.copy()
    a.layers["cellquorum_normalized"] = np.log1p(x)
    return a


class _Paths:
    def __init__(self, tmp):
        from pathlib import Path

        self.results = Path(tmp)
        self.figures = Path(tmp)


class _Config:
    """Match real context.config structure: config.qc reads QCConfig."""

    def __init__(self, qc):
        self.qc = qc
        self.stages = type("S", (), {"qc": True})()


class _Ctx:
    def __init__(self, adata, tmp, qc_config):
        self._adata = adata
        self.paths = _Paths(tmp)
        self.config = _Config(qc_config)
        self.run_id = "test-run"
        self.random_seed = 42

    def require_adata(self):
        return self._adata


def test_qc_stage_runs_cell_cycle_when_enabled(tmp_path):
    a = _adata()
    qc = QCConfig(mode="report_only", cell_cycle={"enabled": True})
    ctx = _Ctx(a, tmp_path, qc)
    result = QCStage().run(ctx)
    assert "phase" in result.adata.obs


def test_qc_stage_flags_doublets_when_enabled(tmp_path):
    a = _adata()
    qc = QCConfig(mode="report_only", doublets={"enabled": True, "methods": ["scrublet"]})
    ctx = _Ctx(a, tmp_path, qc)
    result = QCStage().run(ctx)
    assert "predicted_doublet" in result.adata.obs
    assert result.adata.n_obs == 120  # flag-only, no removal
