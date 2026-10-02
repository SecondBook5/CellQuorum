"""Tests for multi-method dispatch in MethodDispatchStage."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.methods.registry import MethodRegistry
from cellquorum.methods.stage_base import MethodDispatchStage


class _WriteColMethod(AnalysisMethod):
    """Test method: writes obs[key_added] = its name."""

    name = "writer_a"
    stage_category = "toy"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata, config, context) -> StageResult:
        key = config.get("key_added", "label")
        adata.obs[key] = self.name
        return StageResult(adata=adata, metrics={"key_added": key})


class _WriteColMethodB(_WriteColMethod):
    name = "writer_b"


class _ToyStage(MethodDispatchStage):
    name = "toy"
    stage_category = "toy"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "writer_a")


class _Ctx:
    def __init__(self, adata, config):
        self._adata = adata
        self.config = config

    def require_adata(self):
        return self._adata


def _adata():
    a = ad.AnnData(X=np.ones((5, 2), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(5)]
    return a


def _registry():
    reg = MethodRegistry()
    reg.register(_WriteColMethod)
    reg.register(_WriteColMethodB)
    return reg


def test_multi_method_runs_each_in_order():
    stage = _ToyStage(registry=_registry())
    a = _adata()
    ctx = _Ctx(
        a,
        {
            "toy": {
                "methods": [
                    {"method": "writer_a", "key_added": "col_a"},
                    {"method": "writer_b", "key_added": "col_b"},
                ]
            }
        },
    )
    result = stage.run(ctx)
    assert "col_a" in result.adata.obs.columns
    assert "col_b" in result.adata.obs.columns
    assert result.adata.obs["col_a"].iloc[0] == "writer_a"
    assert result.adata.obs["col_b"].iloc[0] == "writer_b"
    assert result.metrics.get("n_methods") == 2
    # Verify per-method metrics include the key_added values.
    per_method = result.metrics.get("per_method")
    assert isinstance(per_method, list)
    assert len(per_method) == 2
    assert per_method[0]["key_added"] == "col_a"
    assert per_method[1]["key_added"] == "col_b"


def test_single_method_path_unchanged():
    stage = _ToyStage(registry=_registry())
    a = _adata()
    ctx = _Ctx(a, {"toy": {"method": "writer_a", "key_added": "only"}})
    result = stage.run(ctx)
    assert "only" in result.adata.obs.columns
    assert "n_methods" not in result.metrics


def test_multi_method_tolerates_a_skip():
    class _Skipper(_WriteColMethod):
        name = "skipper"

        def _run(self, adata, config, context):
            return MethodSkip(reason="nothing to do")

    reg = _registry()
    reg.register(_Skipper)
    stage = _ToyStage(registry=reg)
    a = _adata()
    ctx = _Ctx(
        a,
        {
            "toy": {
                "methods": [
                    {"method": "skipper"},
                    {"method": "writer_b", "key_added": "col_b"},
                ]
            }
        },
    )
    result = stage.run(ctx)
    # A per-method skip must not abort the remaining methods.
    assert "col_b" in result.adata.obs.columns
    assert "skipper" in " ".join(result.warnings)


def test_empty_methods_list_uses_scalar_path():
    """An empty methods list (the pydantic default) must use the scalar method path."""
    stage = _ToyStage(registry=_registry())
    a = _adata()
    ctx = _Ctx(a, {"toy": {"method": "writer_a", "methods": [], "key_added": "scalar"}})
    result = stage.run(ctx)
    # Empty methods list must NOT skip; it must run the scalar method.
    assert "scalar" in result.adata.obs.columns
    assert result.adata.obs["scalar"].iloc[0] == "writer_a"
    # Must NOT have multi-method metrics.
    assert "n_methods" not in result.metrics
