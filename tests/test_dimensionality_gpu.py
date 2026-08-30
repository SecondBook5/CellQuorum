"""GPU-path tests for PCA (skip when rapids-singlecell unavailable)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.backends.compute import gpu_compute_available
from cellquorum.stages.preprocessing.dimensionality.pca import PCAMethod

pytestmark = pytest.mark.skipif(
    not gpu_compute_available(), reason="rapids-singlecell/cupy unavailable"
)


class _GpuCtx:
    """Context forcing GPU, with a figures dir."""

    def __init__(self, tmp):
        from pathlib import Path

        self.config = type(
            "C",
            (),
            {
                "compute": type(
                    "K", (), {"backend": "gpu", "prefer_gpu": True, "fallback_to_cpu": True}
                )()
            },
        )()
        self.paths = type("P", (), {"figures": Path(tmp)})()

    def require_adata(self):
        raise AssertionError("method receives adata directly")


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 50)).astype(np.float32)
    x[:100, :5] += 5.0
    a = ad.AnnData(X=x)
    a.layers["cellquorum_normalized"] = x.copy()
    from cellquorum.core.contracts import set_layer_tag

    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    return a


def test_pca_gpu_produces_same_output_keys(tmp_path):
    m = PCAMethod()
    a = _adata()
    result = m.run(
        a,
        {"input_layer": "cellquorum_normalized", "n_pcs": 10, "max_pcs": 50, "random_state": 0},
        context=_GpuCtx(tmp_path),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    # Same output contract as the CPU path.
    assert result.adata.obsm["X_pca"].shape[1] == 10
    assert result.metrics["n_pcs"] == 10
    # Records which compute path ran.
    assert result.metrics.get("compute") == "gpu"
