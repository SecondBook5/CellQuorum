"""GPU-path test for PFlog1pPF normalization (skips when rapids/cupy absent)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellquorum.compute.router import gpu_compute_available
from cellquorum.preprocessing.config import NormalizationConfig
from cellquorum.preprocessing.normalization import normalize_adata

pytestmark = pytest.mark.skipif(
    not gpu_compute_available(), reason="rapids-singlecell/cupy unavailable"
)


def _counts(seed=0):
    rng = np.random.default_rng(seed)
    return ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, size=(200, 60)).astype(np.float32)))


def test_gpu_normalization_matches_cpu():
    cfg = NormalizationConfig()  # default pf_log1p_pf recipe
    a = _counts()

    cpu = normalize_adata(a.copy(), cfg, use_gpu=False).adata
    gpu = normalize_adata(a.copy(), cfg, use_gpu=True).adata

    cpu_x = cpu.layers[cfg.output_layer]
    gpu_x = gpu.layers[cfg.output_layer]
    cpu_x = cpu_x.toarray() if sp.issparse(cpu_x) else np.asarray(cpu_x)
    gpu_x = gpu_x.toarray() if sp.issparse(gpu_x) else np.asarray(gpu_x)
    # GPU result matches CPU to float tolerance (verified ~7e-8 at plan time).
    assert np.allclose(cpu_x, gpu_x, atol=1e-5)
