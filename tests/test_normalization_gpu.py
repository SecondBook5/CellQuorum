"""PFlog1pPF (scclr) is env-internal: the use_gpu flag does not affect it.

The old CPU-vs-GPU shifted-CLR parity test is obsolete — PFlog1pPF now runs the
real scclr transform in an isolated subprocess environment, which manages its
own compute. There is no in-process cupy branch to compare against. This test
just confirms the scclr recipe produces the same sparse result regardless of the
``use_gpu`` flag (it is ignored for this recipe). Skips when the scclr env is
absent.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellquorum.preprocessing.config import NormalizationConfig
from cellquorum.preprocessing.normalization import normalize_adata


def _scclr_backend_or_skip():
    from cellquorum.backends.scclr_backend import build_scclr_backend

    backend = build_scclr_backend()
    if not backend.status().available:
        pytest.skip("scclr environment unavailable (isolated micromamba env not built)")
    return backend


def _counts(seed=0):
    # NB-distributed so scclr's target="auto" overdispersion estimate is valid.
    rng = np.random.default_rng(seed)
    return ad.AnnData(
        X=sp.csr_matrix(rng.negative_binomial(2, 0.15, size=(80, 30)).astype("float32"))
    )


def test_scclr_normalization_ignores_use_gpu_flag(tmp_path):
    backend = _scclr_backend_or_skip()
    cfg = NormalizationConfig()  # default cellquorum_pf_log1p_pf_v1 (scclr)
    a = _counts()

    off = normalize_adata(a.copy(), cfg, use_gpu=False, backend=backend, scratch_dir=tmp_path).adata
    on = normalize_adata(a.copy(), cfg, use_gpu=True, backend=backend, scratch_dir=tmp_path).adata

    off_x = off.layers[cfg.output_layer]
    on_x = on.layers[cfg.output_layer]
    assert sp.issparse(off_x) and sp.issparse(on_x)
    assert np.allclose(off_x.toarray(), on_x.toarray(), atol=1e-6)
