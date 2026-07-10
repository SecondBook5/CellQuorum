"""Clustering can run on a configured embedding (e.g. the integration output)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.clustering.neighbors_leiden import LeidenMethod
from cellquorum.contracts import CellQuorumContractError


def _adata_with_rep(rep, n=160, n_pcs=10, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.normal(size=(n, 20)).astype(np.float32))
    emb = rng.normal(size=(n, n_pcs)).astype(np.float32)
    emb[: n // 2, 0] += 10.0
    a.obsm[rep] = emb
    return a


def test_leiden_clusters_on_configured_use_rep():
    m = LeidenMethod()
    a = _adata_with_rep("X_pca_harmony")
    result = m.run(
        a,
        {
            "use_rep": "X_pca_harmony",
            "n_neighbors": 15,
            "resolution": 1.0,
            "random_state": 0,
            "key_added": "leiden",
        },
        context=None,
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert result.metrics["n_clusters"] >= 2


def test_leiden_requires_the_configured_rep():
    m = LeidenMethod()
    a = _adata_with_rep("X_pca")  # only X_pca present
    with pytest.raises(CellQuorumContractError, match="X_pca_harmony"):
        m.run(
            a,
            {
                "use_rep": "X_pca_harmony",
                "n_neighbors": 5,
                "resolution": 1.0,
                "random_state": 0,
                "key_added": "leiden",
            },
            context=None,
        )


def test_leiden_defaults_to_x_pca():
    m = LeidenMethod()
    a = _adata_with_rep("X_pca")
    result = m.run(
        a,
        {"n_neighbors": 15, "resolution": 1.0, "random_state": 0, "key_added": "leiden"},
        context=None,
    )
    assert "leiden" in result.adata.obs
