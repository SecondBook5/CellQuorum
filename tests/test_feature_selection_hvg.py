"""HVGMethod: writes var['highly_variable'], honors layer discipline + excludes."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.contracts import CellQuorumContractError
from cellquorum.contracts.layer_tags import set_layer_tag
from cellquorum.feature_selection.hvg import HVGMethod


def _counts_adata(n=200, g=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g - 3)] + ["MT-CO1", "RPS2", "HBB"]
    a.layers["counts"] = X.copy()
    # A PFlog1pPF-like centered lognorm layer (has negatives).
    a.layers["cellquorum_normalized"] = (X - X.mean(0)).astype("float32")
    # Tag the layers using the real helper so contracts see the correct format.
    set_layer_tag(a, "counts", kind="counts")
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    return a


def test_seurat_v3_flags_highly_variable_from_counts():
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert "highly_variable" in result.adata.var.columns
    assert int(result.adata.var["highly_variable"].sum()) >= 1


def test_exclude_patterns_remove_genes_from_hvg():
    a = _counts_adata()
    cfg = {
        "method": "seurat_v3",
        "n_top_genes": 50,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": ["^MT-", "^RP[SL]", "^HB[AB]"],
    }
    result = HVGMethod().run(a, cfg, context=None)
    hv = result.adata.var["highly_variable"]
    assert not bool(hv.get("MT-CO1", False))
    assert not bool(hv.get("RPS2", False))
    assert not bool(hv.get("HBB", False))


def test_seurat_v3_contract_rejects_centered_layer_as_counts():
    # Point counts_layer at the centered (negative) layer -> counts contract must fail.
    a = _counts_adata()
    cfg = {"method": "seurat_v3", "n_top_genes": 20, "counts_layer": "cellquorum_normalized"}
    with pytest.raises(CellQuorumContractError):
        HVGMethod().run(a, cfg, context=None)


def test_pearson_residuals_flags_hvg_from_counts():
    a = _counts_adata()
    cfg = {
        "method": "pearson_residuals",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert "highly_variable" in result.adata.var.columns
    assert int(result.adata.var["highly_variable"].sum()) >= 1


def test_seurat_flavor_flags_hvg_from_lognorm():
    a = _counts_adata()
    cfg = {
        "method": "seurat",
        "n_top_genes": 20,
        "counts_layer": "counts",
        "lognorm_layer": "cellquorum_normalized",
        "exclude_gene_patterns": [],
    }
    result = HVGMethod().run(a, cfg, context=None)
    assert "highly_variable" in result.adata.var.columns
    assert int(result.adata.var["highly_variable"].sum()) >= 1
