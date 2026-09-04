"""Tests for the marker-vote annotation method."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.contracts import CellQuorumContractError, set_layer_tag
from cellquorum.stages.annotation.marker_vote import MarkerVoteMethod


def _clustered_adata(seed=0):
    # Two clusters; cluster 0 expresses gene set A, cluster 1 expresses set B.
    rng = np.random.default_rng(seed)
    n = 100
    genes = ["A1", "A2", "A3", "B1", "B2", "B3"]
    x = rng.random((n, 6)).astype(np.float32) * 0.1
    x[: n // 2, :3] += 2.0  # cluster 0 -> A genes
    x[n // 2 :, 3:] += 2.0  # cluster 1 -> B genes
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.layers["cellquorum_normalized"] = x.copy()
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    a.obs["leiden"] = pd.Categorical(["0"] * (n // 2) + ["1"] * (n // 2))
    return a


def test_marker_vote_assigns_types_by_argmax():
    m = MarkerVoteMethod()
    a = _clustered_adata()
    cfg = {
        "cluster_key": "leiden",
        "score_layer": "cellquorum_normalized",
        "key_added": "cell_type",
        "random_state": 0,
        "marker_panels": {"TypeA": ["A1", "A2", "A3"], "TypeB": ["B1", "B2", "B3"]},
    }
    result = m.run(a, cfg, context=None)
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    ct = result.adata.obs["cell_type"]
    # Cluster 0 -> TypeA, cluster 1 -> TypeB.
    assert ct[a.obs["leiden"] == "0"].iloc[0] == "TypeA"
    assert ct[a.obs["leiden"] == "1"].iloc[0] == "TypeB"


def test_marker_vote_skips_when_cluster_key_absent():
    """MarkerVote skips gracefully when the cluster obs column is missing."""
    m = MarkerVoteMethod()
    a = _clustered_adata()
    del a.obs["leiden"]
    cfg = {
        "cluster_key": "leiden",
        "score_layer": "cellquorum_normalized",
        "marker_panels": {"TypeA": ["A1"]},
    }
    result = m.run(a, cfg, context=None)
    from cellquorum.methods.base import MethodSkip

    assert isinstance(result, MethodSkip)
    assert "leiden" in result.reason


def test_marker_vote_raises_when_layer_absent():
    """MarkerVote raises via contract when the score layer is missing."""
    m = MarkerVoteMethod()
    a = _clustered_adata()
    del a.layers["cellquorum_normalized"]
    cfg = {
        "cluster_key": "leiden",
        "score_layer": "cellquorum_normalized",
        "marker_panels": {"TypeA": ["A1"]},
    }
    with pytest.raises(CellQuorumContractError, match="cellquorum_normalized"):
        m.run(a, cfg, context=None)
