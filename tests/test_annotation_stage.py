"""Tests for AnnotationStage dispatch + output guard."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.annotation.marker_vote import MarkerVoteMethod
from cellquorum.annotation.stage import AnnotationStage
from cellquorum.core.contracts import set_layer_tag
from cellquorum.methods.registry import MethodRegistry


def _clustered_adata(seed=0):
    rng = np.random.default_rng(seed)
    n = 100
    x = rng.random((n, 6)).astype(np.float32) * 0.1
    x[: n // 2, :3] += 2.0
    x[n // 2 :, 3:] += 2.0
    a = ad.AnnData(X=x, var=pd.DataFrame(index=["A1", "A2", "A3", "B1", "B2", "B3"]))
    a.layers["cellquorum_normalized"] = x.copy()
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    a.obs["leiden"] = pd.Categorical(["0"] * (n // 2) + ["1"] * (n // 2))
    return a


class _Ctx:
    def __init__(self, adata, config):
        self._adata = adata
        self.config = config

    def require_adata(self):
        return self._adata


def test_annotation_stage_assigns_and_validates():
    reg = MethodRegistry()
    reg.register(MarkerVoteMethod)
    stage = AnnotationStage(registry=reg)
    a = _clustered_adata()
    ctx = _Ctx(
        a,
        {
            "annotation": {
                "method": "marker_vote",
                "cluster_key": "leiden",
                "score_layer": "cellquorum_normalized",
                "key_added": "cell_type",
                "random_state": 0,
                "marker_panels": {"TypeA": ["A1", "A2", "A3"], "TypeB": ["B1", "B2", "B3"]},
            }
        },
    )
    result = stage.run(ctx)
    assert "cell_type" in result.adata.obs


def test_annotation_stage_disabled_skips():
    reg = MethodRegistry()
    reg.register(MarkerVoteMethod)
    stage = AnnotationStage(registry=reg)
    a = _clustered_adata()
    ctx = _Ctx(a, {"annotation": {"enabled": False}})
    result = stage.run(ctx)
    assert result.metrics.get("skipped") is True
