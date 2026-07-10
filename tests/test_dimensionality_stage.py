"""Tests for the PCA dimensionality method and stage."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.dimensionality.pca import PCAMethod
from cellquorum.dimensionality.stage import DimensionalityStage
from cellquorum.methods.registry import MethodRegistry


def _adata(n_cells=200, n_genes=50, seed=0):
    rng = np.random.default_rng(seed)
    # Structured data so PCA has a real signal: two blobs.
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    x[: n_cells // 2, :5] += 5.0
    a = ad.AnnData(X=x)
    # Add a tagged normalized layer (non-integer log-like values).
    lognorm = rng.normal(loc=2.0, scale=1.5, size=(n_cells, n_genes)).astype(np.float32)
    lognorm[: n_cells // 2, :5] += 3.0
    a.layers["cellquorum_normalized"] = lognorm
    from cellquorum.contracts import set_layer_tag

    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    return a


class _Paths:
    def __init__(self, tmp):
        self.figures = Path(tmp)


class _Ctx:
    def __init__(self, adata, tmp, config):
        self._adata = adata
        self.paths = _Paths(tmp)
        self.config = config

    def require_adata(self):
        return self._adata


def test_pca_method_fixed_n_pcs(tmp_path):
    m = PCAMethod()
    a = _adata()
    result = m.run(
        a,
        {"n_pcs": 10, "max_pcs": 50, "random_state": 0, "use_highly_variable": False},
        context=_Ctx(a, tmp_path, {}),
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert result.adata.obsm["X_pca"].shape[1] == 10
    assert result.metrics["n_pcs"] == 10
    assert result.metrics["n_pcs_mode"] == "fixed"


def test_pca_method_auto_n_pcs_records_choice(tmp_path):
    m = PCAMethod()
    a = _adata()
    result = m.run(
        a,
        {"n_pcs": "auto", "max_pcs": 30, "random_state": 0, "use_highly_variable": False},
        context=_Ctx(a, tmp_path, {}),
    )
    assert result.metrics["n_pcs_mode"] == "auto"
    assert 1 <= result.metrics["n_pcs"] <= 30
    assert result.adata.obsm["X_pca"].shape[1] == result.metrics["n_pcs"]


def test_pca_writes_scree_artifact(tmp_path):
    m = PCAMethod()
    a = _adata()
    result = m.run(
        a,
        {"n_pcs": 10, "max_pcs": 50, "random_state": 0, "use_highly_variable": False},
        context=_Ctx(a, tmp_path, {}),
    )
    scree = [art for art in result.artifacts if "scree" in art.name]
    assert scree, "expected a scree artifact"
    assert Path(scree[0].path).is_file()


def test_dimensionality_stage_dispatches_and_validates(tmp_path):
    reg = MethodRegistry()
    reg.register(PCAMethod)
    stage = DimensionalityStage(registry=reg)
    a = _adata()
    ctx = _Ctx(
        a,
        tmp_path,
        {
            "dimensionality": {
                "method": "pca",
                "n_pcs": 8,
                "max_pcs": 50,
                "random_state": 0,
                "use_highly_variable": False,
            }
        },
    )
    result = stage.run(ctx)
    assert result.adata.obsm["X_pca"].shape[1] == 8


def test_dimensionality_stage_honors_enabled_false(tmp_path):
    reg = MethodRegistry()
    reg.register(PCAMethod)
    stage = DimensionalityStage(registry=reg)
    a = _adata()
    ctx = _Ctx(
        a,
        tmp_path,
        {
            "dimensionality": {
                "enabled": False,
                "method": "pca",
                "n_pcs": 8,
            }
        },
    )
    result = stage.run(ctx)
    assert result.metrics.get("skipped") is True
    assert result.metrics.get("reason") == "disabled by config"
    assert any("disabled" in w for w in result.warnings)
