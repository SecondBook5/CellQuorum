"""GPU-path tests for Leiden clustering (skip when rapids unavailable)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.clustering.neighbors_leiden import LeidenMethod
from cellquorum.compute.router import gpu_compute_available

pytestmark = pytest.mark.skipif(
    not gpu_compute_available(), reason="rapids-singlecell/cupy unavailable"
)


class _GpuCtx:
    def __init__(self):
        self.config = type(
            "C",
            (),
            {
                "compute": type(
                    "K", (), {"backend": "gpu", "prefer_gpu": True, "fallback_to_cpu": True}
                )()
            },
        )()

    def require_adata(self):
        raise AssertionError("method receives adata directly")


def _adata_with_pca(seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.normal(size=(200, 20)).astype(np.float32))
    pca = rng.normal(size=(200, 10)).astype(np.float32)
    pca[:100, 0] += 10.0
    a.obsm["X_pca"] = pca
    return a


def test_leiden_gpu_produces_clusters():
    m = LeidenMethod()
    a = _adata_with_pca()
    result = m.run(
        a,
        {
            "use_rep": "X_pca",
            "n_neighbors": 15,
            "resolution": 1.0,
            "random_state": 0,
            "key_added": "leiden",
        },
        context=_GpuCtx(),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert "leiden" in result.adata.obs
    assert result.metrics["n_clusters"] >= 2
    assert result.metrics.get("compute") == "gpu"
