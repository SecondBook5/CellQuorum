from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.config.design import DesignConfig
from cellquorum.enrichment.stage import EnrichmentStage


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.scratch = tmp / "scratch"
        self.results.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)


class _Cfg:
    organism = "human"
    # No DE table on disk, no decoupler mock → every method must SKIP, not crash.
    enrichment = {"enabled": True}
    cohort = None
    design = DesignConfig(
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        control="Normal",
        paired=False,
    )


class _Ctx:
    def __init__(self, tmp, adata):
        self.config = _Cfg()
        self.paths = _Paths(tmp)
        self.adata = adata
        self.backend_registry = None

    def require_adata(self):
        return self.adata


def _adata():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "patient_id": ["d1"] * 5 + ["d2"] * 5,
            "condition": ["Normal"] * 5 + ["Disease"] * 5,
            "cell_type": ["T0"] * 10,
        }
    )
    a = ad.AnnData(X=rng.normal(size=(10, 6)), obs=obs)
    a.var_names = [f"G{i}" for i in range(6)]
    return a


def test_stage_dispatches_four_methods_without_crashing(tmp_path):
    a = _adata()
    ctx = _Ctx(tmp_path, a)
    result = EnrichmentStage().run(ctx)
    assert result.metrics["n_methods"] == 4
    assert len(result.metrics["per_method"]) == 4
    # No method skipped for case/control-unset (design bridge delivered them).
    for entry in result.metrics["per_method"]:
        if entry.get("skipped"):
            reason = entry.get("reason", "").lower()
            assert "case" not in reason or "control" not in reason
