# tests/stages/clustering/test_neighbors_leiden_force_cpu.py
"""force_cpu must bypass GPU routing entirely, not just prefer CPU.

GPU-accelerated Leiden (rapids_singlecell/cuGraph) doesn't guarantee bit-exact
reproducibility across runs even with the same seed -- parallel floating-point
reductions on GPU don't sum in a fixed order. Confirmed live: two nominally
identical subclustering re-runs (same seed, same input) produced different
cluster boundaries. force_cpu is the deterministic opt-out for exactly that.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.stages.clustering.neighbors_leiden import run_neighbors_leiden_routed


def _blob_adata(n: int = 60, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(n, 5)).astype(np.float32)
    adata = ad.AnnData(X=rng.normal(size=(n, 10)).astype(np.float32))
    adata.obsm["X_pca"] = coords
    return adata


def test_force_cpu_never_calls_gpu_routing_even_when_gpu_would_be_used(monkeypatch):
    """With force_cpu=True, resolve_compute is never consulted -- not "prefer

    CPU," but "GPU routing doesn't exist for this call."
    """
    called = {"resolve_compute": False}

    def _fake_resolve_compute(context):
        called["resolve_compute"] = True
        return {"use_gpu": True, "fallback_to_cpu": True}

    monkeypatch.setattr("cellquorum.backends.compute.resolve_compute", _fake_resolve_compute)

    adata = _blob_adata()
    compute_used, fallback_note = run_neighbors_leiden_routed(
        adata,
        context=object(),
        n_neighbors=10,
        use_rep="X_pca",
        resolution=1.0,
        random_state=0,
        key_added="leiden",
        force_cpu=True,
    )

    assert called["resolve_compute"] is False
    assert compute_used == "cpu"
    assert fallback_note is None
    assert "leiden" in adata.obs.columns


def test_force_cpu_defaults_to_false_and_preserves_existing_routing(monkeypatch):
    """Every existing caller omits force_cpu -- must keep routing through

    resolve_compute exactly as before (regression guard for the shared
    GPU-first/CPU-fallback contract LeidenMethod and the bootstrap sweep
    both depend on not drifting).
    """
    called = {"resolve_compute": False}

    def _fake_resolve_compute(context):
        called["resolve_compute"] = True
        return {"use_gpu": False, "fallback_to_cpu": True}

    monkeypatch.setattr("cellquorum.backends.compute.resolve_compute", _fake_resolve_compute)

    adata = _blob_adata()
    run_neighbors_leiden_routed(
        adata,
        context=object(),
        n_neighbors=10,
        use_rep="X_pca",
        resolution=1.0,
        random_state=0,
        key_added="leiden",
    )

    assert called["resolve_compute"] is True
