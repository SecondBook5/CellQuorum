"""PCA routing: scclr-normalized layer -> scclr sparse PCA; else scanpy PCA."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import set_layer_tag
from cellquorum.preprocessing.dimensionality.pca import PCAMethod


class _Paths:
    def __init__(self, tmp: Path):
        self.figures = tmp
        self.scratch = tmp


class _Ctx:
    def __init__(self, adata, tmp, *, backend_registry=None):
        self._adata = adata
        self.paths = _Paths(tmp)
        self.config = {}
        self.backend_registry = backend_registry

    def require_adata(self):
        return self._adata


def _scclr_registry_or_skip():
    from cellquorum.backends.registry import build_default_backend_registry

    registry = build_default_backend_registry()
    if not registry.get("scclr").status().available:
        pytest.skip("scclr environment unavailable (isolated micromamba env not built)")
    return registry


def _scclr_normalized_adata(tmp: Path, registry):
    """Build an adata whose normalized layer was produced by the scclr backend."""

    from cellquorum.preprocessing.config import NormalizationConfig
    from cellquorum.preprocessing.normalization import normalize_adata

    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(2, 0.15, size=(80, 30)).astype(np.float32)
    adata = ad.AnnData(X=counts)
    cfg = NormalizationConfig(
        recipe="cellquorum_pf_log1p_pf_v1", output_layer="cellquorum_normalized"
    )
    result = normalize_adata(adata, cfg, backend=registry.get("scclr"), scratch_dir=tmp)
    return result.adata


def test_scclr_layer_routes_to_scclr_pca(tmp_path):
    """A scclr-normalized layer (with row_center) uses scclr's sparse PCA."""
    registry = _scclr_registry_or_skip()
    adata = _scclr_normalized_adata(tmp_path, registry)
    # The scclr normalization must have left a row_center obs column.
    assert "cellquorum_normalized_row_center" in adata.obs.columns

    result = PCAMethod().run(
        adata,
        {"n_pcs": 10, "max_pcs": 20, "input_layer": "cellquorum_normalized"},
        context=_Ctx(adata, tmp_path, backend_registry=registry),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert result.metrics["compute"] == "scclr"
    assert result.adata.obsm["X_pca"].shape[0] == 80
    assert result.adata.obsm["X_pca"].shape[1] == 10


def test_standard_layer_routes_to_scanpy_pca(tmp_path):
    """A standard lognorm layer (no row_center) uses the scanpy PCA path."""
    rng = np.random.default_rng(1)
    lognorm = rng.normal(loc=2.0, scale=1.5, size=(120, 40)).astype(np.float32)
    lognorm[:60, :5] += 3.0
    adata = ad.AnnData(X=np.abs(lognorm))
    adata.layers["cellquorum_normalized"] = np.abs(lognorm)
    set_layer_tag(
        adata, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_log1p_cp10k_v1"
    )

    result = PCAMethod().run(
        adata,
        {"n_pcs": 8, "max_pcs": 20, "input_layer": "cellquorum_normalized"},
        context=_Ctx(adata, tmp_path),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    # No row_center -> standard scanpy path (cpu/gpu), never "scclr".
    assert result.metrics["compute"] != "scclr"
    assert result.adata.obsm["X_pca"].shape == (120, 8)
