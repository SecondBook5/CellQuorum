"""Tests for the Leiden clustering method and stage."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.clustering.neighbors_leiden import LeidenMethod
from cellquorum.clustering.stage import ClusteringStage
from cellquorum.contracts import CellQuorumContractError
from cellquorum.methods.registry import MethodRegistry


def _adata_with_pca(n_cells=200, n_pcs=10, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.normal(size=(n_cells, 20)).astype(np.float32))
    # Two separated blobs in PCA space so Leiden finds >1 cluster.
    pca = rng.normal(size=(n_cells, n_pcs)).astype(np.float32)
    pca[: n_cells // 2, 0] += 10.0
    a.obsm["X_pca"] = pca
    return a


class _Ctx:
    def __init__(self, adata, config):
        self._adata = adata
        self.config = config

    def require_adata(self):
        return self._adata


def test_leiden_method_adds_clusters():
    m = LeidenMethod()
    a = _adata_with_pca()
    result = m.run(
        a,
        {"n_neighbors": 15, "resolution": 1.0, "random_state": 0, "key_added": "leiden"},
        context=_Ctx(a, {}),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert "leiden" in result.adata.obs
    assert result.metrics["n_clusters"] >= 2


def test_leiden_method_requires_pca():
    m = LeidenMethod()
    a = ad.AnnData(X=np.zeros((10, 5), dtype=np.float32))  # no X_pca
    with pytest.raises(CellQuorumContractError, match="X_pca"):
        m.run(
            a,
            {"n_neighbors": 5, "resolution": 1.0, "random_state": 0, "key_added": "leiden"},
            context=_Ctx(a, {}),
        )


def test_clustering_stage_dispatches_and_validates():
    reg = MethodRegistry()
    reg.register(LeidenMethod)
    stage = ClusteringStage(registry=reg)
    a = _adata_with_pca()
    ctx = _Ctx(
        a,
        {
            "clustering": {
                "method": "leiden",
                "n_neighbors": 15,
                "resolution": 1.0,
                "random_state": 0,
                "key_added": "leiden",
            }
        },
    )
    result = stage.run(ctx)
    assert "leiden" in result.adata.obs
