"""Multi-method wiring for the real integration and annotation stages."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.annotation.marker_vote import MarkerVoteMethod
from cellquorum.stages.annotation.stage import AnnotationStage
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


def test_annotation_multi_method_writes_two_columns():
    reg = MethodRegistry()
    reg.register(MarkerVoteMethod)
    stage = AnnotationStage(registry=reg)
    a = _clustered_adata()
    panels = {"TypeA": ["A1", "A2", "A3"], "TypeB": ["B1", "B2", "B3"]}
    ctx = _Ctx(
        a,
        {
            "annotation": {
                "score_layer": "cellquorum_normalized",
                "cluster_key": "leiden",
                "marker_panels": panels,
                "methods": [
                    {"method": "marker_vote", "key_added": "cell_type_markers"},
                    {"method": "marker_vote", "key_added": "cell_type_markers_2"},
                ],
            }
        },
    )
    result = stage.run(ctx)
    assert "cell_type_markers" in result.adata.obs.columns
    assert "cell_type_markers_2" in result.adata.obs.columns
    assert result.metrics["n_methods"] == 2
