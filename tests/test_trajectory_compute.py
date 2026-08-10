from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.trajectory import compute


def _adata_with_reps():
    rng = np.random.default_rng(0)
    X = rng.random((10, 5)).astype("float32")
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]))
    a.obsm["X_pca"] = rng.random((10, 4))
    a.obsm["X_umap"] = rng.random((10, 2))
    return a


def test_resolve_use_rep_prefers_configured():
    a = _adata_with_reps()
    a.obsm["X_scANVI"] = np.zeros((10, 3))
    assert compute.resolve_use_rep(a, "X_scANVI", ["X_pca"]) == "X_scANVI"


def test_resolve_use_rep_falls_through_chain():
    a = _adata_with_reps()  # only X_pca and X_umap present
    assert compute.resolve_use_rep(a, None, ["X_scANVI", "X_scVI", "X_pca"]) == "X_pca"


def test_resolve_use_rep_none_when_absent():
    a = ad.AnnData(X=np.ones((3, 2)))
    assert compute.resolve_use_rep(a, None, ["X_scANVI"]) is None


def test_embedding_bases_excludes_pca_and_sorts():
    a = _adata_with_reps()
    a.obsm["X_phate"] = np.zeros((10, 2))
    a.obsm["X_diffmap"] = np.zeros((10, 2))
    bases = compute.embedding_bases(a)
    assert bases == ["diffmap", "phate", "umap"]  # sorted, no pca


def test_compute_velocity_unavailable_raises_typed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scvelo":
            raise ImportError("no scvelo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    a = _adata_with_reps()
    import pytest

    with pytest.raises(compute.ScveloUnavailable):
        compute.compute_velocity(
            a,
            mode="dynamical",
            use_rep="X_pca",
            min_shared_counts=0,
            n_top_genes=5,
            n_pcs=3,
            n_neighbors=3,
            n_jobs=1,
            seed=0,
        )
