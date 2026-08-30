"""Tests for the IntegrationStage dispatch + output guard."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.stages.integration.harmony import HarmonyMethod
from cellquorum.stages.integration.stage import IntegrationStage
from cellquorum.methods.registry import MethodRegistry


def _adata(n=120, n_pcs=10, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.normal(size=(n, 20)).astype(np.float32))
    pca = rng.normal(size=(n, n_pcs)).astype(np.float32)
    batch = np.array(["A", "B"] * (n // 2))
    a.obsm["X_pca"] = pca
    a.obs["patient_id"] = batch
    return a


class _Ctx:
    def __init__(self, adata, config):
        self._adata = adata
        self.config = config

    def require_adata(self):
        return self._adata


def test_integration_stage_runs_harmony_and_validates_output():
    reg = MethodRegistry()
    reg.register(HarmonyMethod)
    stage = IntegrationStage(registry=reg)
    a = _adata()
    ctx = _Ctx(
        a,
        {
            "integration": {
                "method": "harmony",
                "batch_key": "patient_id",
                "input_rep": "X_pca",
                "output_rep": "X_pca_harmony",
                "random_state": 0,
            }
        },
    )
    result = stage.run(ctx)
    assert "X_pca_harmony" in result.adata.obsm


def test_integration_stage_disabled_skips():
    reg = MethodRegistry()
    reg.register(HarmonyMethod)
    stage = IntegrationStage(registry=reg)
    a = _adata()
    ctx = _Ctx(a, {"integration": {"enabled": False}})
    result = stage.run(ctx)
    assert result.metrics.get("skipped") is True
    assert "X_pca_harmony" not in result.adata.obsm
