"""Tests for the method-dispatching Stage base class."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.methods.registry import MethodRegistry
from cellquorum.methods.stage_base import MethodDispatchStage


class _CountMethod(AnalysisMethod):
    name = "counter"
    stage_category = "demo_stage"
    backend = "python"

    def input_contract(self, config):
        return DataContract()

    def min_donors(self):
        return 5

    def _run(self, adata, config, context):
        return StageResult(adata=adata, metrics={"n": adata.n_obs})


class _DemoStage(MethodDispatchStage):
    name = "demo_stage"
    stage_category = "demo_stage"

    def _select_method_name(self, config):
        return config.get("method", "counter")


class _Ctx:
    def __init__(self, adata):
        self._adata = adata
        self.config = {"demo_stage": {"method": "counter"}}
        self.donor_col = "patient_id"

    def require_adata(self):
        return self._adata


def _adata(n_donors):
    donors = [f"P{i}" for i in range(n_donors)]
    obs = pd.DataFrame({"patient_id": donors}, index=[f"c{i}" for i in range(n_donors)])
    return ad.AnnData(X=np.zeros((n_donors, 2), dtype=np.float32), obs=obs)


def test_dispatch_runs_registered_method():
    reg = MethodRegistry()
    reg.register(_CountMethod)
    stage = _DemoStage(registry=reg)
    ctx = _Ctx(_adata(6))  # 6 >= min_donors 5
    result = stage.run(ctx)
    assert isinstance(result, StageResult)
    assert result.metrics["n"] == 6


def test_dispatch_converts_skip_to_skipped_result():
    reg = MethodRegistry()
    reg.register(_CountMethod)
    stage = _DemoStage(registry=reg)
    ctx = _Ctx(_adata(2))  # 2 < min_donors 5 => MethodSkip
    result = stage.run(ctx)
    # Skip is surfaced as a StageResult carrying a warning, not an exception.
    assert isinstance(result, StageResult)
    assert any("skipped" in w for w in result.warnings)
