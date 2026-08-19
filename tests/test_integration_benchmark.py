"""integration_benchmark: scib metrics over embeddings, read-only, label-fallback."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.integration.benchmark.scib_benchmark import ScibBenchmarkMethod


def _adata(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.poisson(1.0, size=(n, 20)).astype("float32"))
    # two batches, three cell types
    a.obs["batch"] = np.where(np.arange(n) % 2 == 0, "b0", "b1")
    a.obs["cell_type"] = rng.choice(["T", "KC", "Fib"], size=n)
    a.obsm["X_pca"] = rng.normal(size=(n, 10)).astype("float32")
    a.obsm["X_pca_harmony"] = rng.normal(size=(n, 10)).astype("float32")
    return a


def test_benchmark_records_batch_and_bio_metrics():
    a = _adata()
    cfg = {
        "method": "scib_benchmark",
        "batch_key": "batch",
        "label_key": "cell_type",
        "pre_embedding": "X_pca",
        "embeddings": ["X_pca_harmony"],
        "n_neighbors": 30,
        "mode": "full",
        "batch_weight": 0.4,
        "bio_weight": 0.6,
    }
    obsm_before = set(a.obsm.keys())
    obs_before = set(a.obs.columns)
    result = ScibBenchmarkMethod().run(a, cfg, context=None)
    m = result.metrics
    # per-embedding entry with an aggregate score present
    assert "X_pca_harmony" in m["embeddings"]
    assert "aggregate" in m["embeddings"]["X_pca_harmony"]
    # at least one batch metric + one bio metric computed
    emb = m["embeddings"]["X_pca_harmony"]
    assert "ilisi" in emb["batch"]
    assert "clisi" in emb["bio"]
    # kbet should be finite (not NaN from tuple-return bug)
    if "kbet" in emb["batch"]:
        assert np.isfinite(emb["batch"]["kbet"]), "kbet should be finite, not nan"
    # READ-ONLY: no obsm/obs mutation
    assert set(a.obsm.keys()) == obsm_before
    assert set(a.obs.columns) == obs_before


def test_benchmark_falls_back_to_batch_only_without_label():
    a = _adata()
    del a.obs["cell_type"]
    cfg = {
        "method": "scib_benchmark",
        "batch_key": "batch",
        "label_key": "cell_type",
        "label_key_fallback": "cell_type",
        "pre_embedding": "X_pca",
        "embeddings": ["X_pca_harmony"],
        "n_neighbors": 30,
        "mode": "full",
    }
    result = ScibBenchmarkMethod().run(a, cfg, context=None)
    emb = result.metrics["embeddings"]["X_pca_harmony"]
    # bio family empty/skipped, batch family present, note recorded
    assert emb["bio"] == {} or all(np.isnan(list(emb["bio"].values())))
    assert "ilisi" in emb["batch"]


def test_benchmark_fails_loud_when_pre_embedding_missing():
    a = _adata()
    del a.obsm["X_pca"]
    cfg = {
        "method": "scib_benchmark",
        "batch_key": "batch",
        "pre_embedding": "X_pca",
        "embeddings": ["X_pca_harmony"],
        "n_neighbors": 30,
    }
    with pytest.raises(CellQuorumContractError):
        ScibBenchmarkMethod().run(a, cfg, context=None)
