"""CellTypistMethod: counts-layer contract + graceful skip on missing model."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.annotation.celltypist_method import CellTypistMethod
from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip


def _counts_adata(n=120, g=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.layers["counts"] = X.copy()
    a.layers["cellquorum_normalized"] = (X - X.mean(0)).astype("float32")
    # Tag the layers using the real helper so contracts see the correct format.
    set_layer_tag(a, "counts", kind="counts")
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    return a


def test_missing_model_skips_gracefully():
    a = _counts_adata()
    cfg = {
        "method": "celltypist",
        "model": "___nonexistent_model___.pkl",
        "counts_layer": "counts",
        "key_added": "cell_type",
        "majority_voting": False,
    }
    out = CellTypistMethod().run(a, cfg, context=None)
    assert isinstance(out, MethodSkip)


def test_contract_rejects_centered_layer_as_counts():
    a = _counts_adata()
    cfg = {
        "method": "celltypist",
        "model": "Adult_Human_Skin.pkl",
        "counts_layer": "cellquorum_normalized",
        "key_added": "cell_type",
    }
    with pytest.raises(CellQuorumContractError):
        CellTypistMethod().run(a, cfg, context=None)
