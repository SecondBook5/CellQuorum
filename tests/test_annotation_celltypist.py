"""CellTypistMethod: counts-layer contract + graceful skip on missing model."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.annotation.celltypist_method import CellTypistMethod


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


def test_labels_are_assigned_by_cell_identity_not_row_position(monkeypatch):
    """CellTypist's output must be realigned by obs_name, not trusted-by-position.

    Verified empirically (a real cached model, majority_voting=True) that celltypist
    currently does preserve input row order -- but the code should not silently depend
    on that staying true forever. This pins the defensive behavior directly: even when
    the returned DataFrames come back in a different order than the input, cells must
    get the label actually predicted for them, not whatever landed in their row slot.
    """
    import celltypist

    a = _counts_adata(n=6, g=10)
    a.obs_names = [f"cell_{i}" for i in range(6)]

    class _StubModel:
        pass

    class _StubResult:
        def __init__(self, obs_names):
            # Deliberately reversed order relative to the input.
            shuffled = list(reversed(obs_names))
            self.predicted_labels = pd.DataFrame(
                {"predicted_labels": [f"type_{name}" for name in shuffled]}, index=shuffled
            )
            self.probability_matrix = pd.DataFrame(
                {"type_x": [0.9] * len(shuffled)}, index=shuffled
            )

    monkeypatch.setattr(celltypist.models.Model, "load", staticmethod(lambda model: _StubModel()))
    monkeypatch.setattr(
        celltypist, "annotate", lambda work, **kw: _StubResult(list(work.obs_names))
    )

    cfg = {
        "method": "celltypist",
        "model": "any_model.pkl",
        "counts_layer": "counts",
        "key_added": "cell_type",
        "majority_voting": False,
    }
    result = CellTypistMethod().run(a, cfg, context=None)

    for cell_name in a.obs_names:
        assert result.adata.obs.loc[cell_name, "cell_type"] == f"type_{cell_name}", (
            f"{cell_name} got the wrong label -- labels were assigned by row position, "
            f"not by cell identity"
        )
