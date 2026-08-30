"""Tests for the Leiden clustering method and stage."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.stages.clustering.neighbors_leiden import LeidenMethod
from cellquorum.stages.clustering.stage import ClusteringStage
from cellquorum.core.contracts import CellQuorumContractError
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


def test_clustering_stage_honors_enabled_false():
    reg = MethodRegistry()
    reg.register(LeidenMethod)
    stage = ClusteringStage(registry=reg)
    a = _adata_with_pca()
    ctx = _Ctx(
        a,
        {
            "clustering": {
                "enabled": False,
                "method": "leiden",
            }
        },
    )
    result = stage.run(ctx)
    assert result.metrics.get("skipped") is True
    assert result.metrics.get("reason") == "disabled by config"
    assert any("disabled" in w for w in result.warnings)


def test_clustering_auto_couples_to_last_integration_method_output_rep():
    """When integration runs a methods list, clustering couples to the last method's output_rep."""
    from pydantic import BaseModel

    class MockIntegrationConfig(BaseModel):
        enabled: bool = True
        methods: list[dict] = []
        output_rep: str = "X_pca_harmony"

    class MockClusteringConfig(BaseModel):
        enabled: bool = True
        method: str = "leiden"
        n_neighbors: int = 15
        resolution: float = 1.0
        random_state: int = 0
        key_added: str = "leiden"
        use_rep: str = "X_pca"

        model_fields_set: set = set()

    class MockStages(BaseModel):
        integration: bool = True
        clustering: bool = True

    class MockConfig(BaseModel):
        stages: MockStages
        integration: MockIntegrationConfig
        clustering: MockClusteringConfig

    reg = MethodRegistry()
    reg.register(LeidenMethod)
    stage = ClusteringStage(registry=reg)
    a = _adata_with_pca()
    # Add both harmony and scvi embeddings.
    a.obsm["X_pca_harmony"] = a.obsm["X_pca"] + 0.1
    a.obsm["X_scvi"] = a.obsm["X_pca"] + 0.2

    # Configure integration with a methods list where scvi is LAST.
    integration_cfg = MockIntegrationConfig(
        methods=[
            {"method": "harmony", "output_rep": "X_pca_harmony"},
            {"method": "scvi", "output_rep": "X_scvi"},
        ]
    )
    clustering_cfg = MockClusteringConfig()
    config = MockConfig(
        stages=MockStages(),
        integration=integration_cfg,
        clustering=clustering_cfg,
    )

    ctx = _Ctx(a, config)
    result = stage.run(ctx)
    # Clustering should have auto-coupled to X_scvi (the last method's output_rep).
    assert any(
        "X_scvi" in note for note in result.notes
    ), f"Expected X_scvi in notes: {result.notes}"
    assert "leiden" in result.adata.obs
