"""Tests for the discovery stage (consensus-NMF method + dispatch)."""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import set_layer_tag
from cellquorum.discovery.nmf_method import NmfMethod
from cellquorum.discovery.stage import DiscoveryStage
from cellquorum.methods.base import MethodSkip

LAYER = "cellquorum_normalized"


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    """Dict-config context: resolve_stage_config reads config['discovery']."""

    def __init__(self, tmp, adata, stage_config):
        self.config = {"discovery": stage_config}
        self.paths = _Paths(tmp)
        self.adata = adata

    def require_adata(self):
        return self.adata


def _adata(n_cells: int = 60, n_genes: int = 30, hvg: bool = False) -> ad.AnnData:
    """Log-normalized synthetic data with latent block structure so NMF finds
    coherent programs. Small positive floats keep the lognorm contract happy."""
    rng = np.random.default_rng(0)
    # Three latent programs over disjoint gene blocks, mixed per cell.
    k_latent = 3
    block = n_genes // k_latent
    w = rng.random((n_cells, k_latent))
    h = np.zeros((k_latent, n_genes))
    for j in range(k_latent):
        h[j, j * block : (j + 1) * block] = rng.random(block) + 0.5
    x = (w @ h) + rng.random((n_cells, n_genes)) * 0.1
    x = x.astype(float)
    obs = pd.DataFrame(
        {"cell_type": (["T0"] * (n_cells // 2)) + (["T1"] * (n_cells - n_cells // 2))},
        index=[f"c{i}" for i in range(n_cells)],
    )
    a = ad.AnnData(X=x, obs=obs)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.layers[LAYER] = x.copy()
    set_layer_tag(a, LAYER, kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    if hvg:
        mask = np.zeros(n_genes, dtype=bool)
        mask[: 3 * block] = True  # first three blocks are "variable"
        a.var["highly_variable"] = mask
    return a


def _config(**overrides) -> dict:
    base = {
        "method": "nmf",
        "layer": LAYER,
        "n_components": 3,
        "n_runs": 5,
        "n_top_genes": 5,
        "cell_type_col": "cell_type",
        "use_hvg": False,
    }
    base.update(overrides)
    return base


def test_nmf_writes_usage_obsm_and_loadings_table(tmp_path):
    a = _adata()
    cfg = _config()
    out = NmfMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert not isinstance(out, MethodSkip)
    # Usage matrix: cells x programs.
    assert a.obsm["X_cnmf"].shape == (a.n_obs, 3)
    # Usage is non-negative (NMF projection).
    assert a.obsm["X_cnmf"].min() >= 0.0
    # Metadata recorded.
    assert a.uns["cnmf"]["n_components"] == 3
    assert len(a.uns["cnmf"]["programs"]) == 3
    assert len(a.uns["cnmf"]["stability"]) == 3
    # Top-genes loadings table.
    df = pd.read_csv(tmp_path / "results" / "discovery_nmf_top_genes.csv")
    assert list(df.columns) == ["program", "rank", "gene", "loading"]
    assert set(df["program"]) == set(a.uns["cnmf"]["programs"])
    assert df.groupby("program").size().max() == 5  # n_top_genes


def test_nmf_records_clipped_negative_fraction(tmp_path):
    a = _adata()
    # Inject negatives (shifted-CLR-style) so the clip path is exercised.
    a.layers[LAYER][0, 0] = -0.5
    cfg = _config()
    out = NmfMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert not isinstance(out, MethodSkip)
    assert out.metrics["clipped_negative_fraction"] > 0.0


def test_nmf_respects_hvg_subset(tmp_path):
    a = _adata(hvg=True)
    cfg = _config(use_hvg=True)
    out = NmfMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert not isinstance(out, MethodSkip)
    assert out.metrics["used_hvg"] is True
    # Spectra genes are restricted to the highly-variable set.
    assert len(a.uns["cnmf"]["genes"]) == int(a.var["highly_variable"].sum())


def test_nmf_skips_when_rank_too_large(tmp_path):
    a = _adata(n_cells=60, n_genes=30)
    cfg = _config(n_components=40)  # k >= n_genes
    out = NmfMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert isinstance(out, MethodSkip)
    assert "too large" in out.reason.lower()


def test_nmf_skips_when_sklearn_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
    a = _adata()
    cfg = _config()
    out = NmfMethod()._run(a, cfg, _Ctx(tmp_path, a, cfg))
    assert isinstance(out, MethodSkip)
    assert "scikit-learn" in out.reason.lower()


def test_stage_dispatch_runs_nmf_by_default(tmp_path):
    a = _adata()
    ctx = _Ctx(tmp_path, a, _config())
    result = DiscoveryStage().run(ctx)
    # Single-method path: default 'nmf'.
    assert result.metrics["method"] == "nmf"
    assert "X_cnmf" in a.obsm


def test_stage_disabled_returns_recorded_skip(tmp_path):
    a = _adata()
    ctx = _Ctx(tmp_path, a, _config(enabled=False))
    result = DiscoveryStage().run(ctx)
    assert result.status == "skipped"
    assert "X_cnmf" not in a.obsm
