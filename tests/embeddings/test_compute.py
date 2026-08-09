import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from cellquorum.embeddings import compute


def _adata_with_neighbors(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 20)).astype("float32")
    a = ad.AnnData(X=X)
    a.obsm["X_pca_harmony"] = rng.normal(size=(60, 10)).astype("float32")
    a.obs["cell_type"] = (["A"] * 30) + (["B"] * 30)
    a.obs["leiden"] = (["0"] * 20) + (["1"] * 20) + (["2"] * 20)
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    a.obs["leiden"] = a.obs["leiden"].astype("category")
    sc.pp.neighbors(a, use_rep="X_pca_harmony", n_neighbors=15, random_state=0)
    return a


def test_compute_umap_writes_obsm_and_is_deterministic():
    a1 = _adata_with_neighbors()
    a2 = _adata_with_neighbors()
    compute.compute_umap(a1, min_dist=0.3, random_state=1337)
    compute.compute_umap(a2, min_dist=0.3, random_state=1337)
    assert a1.obsm["X_umap"].shape == (60, 2)
    np.testing.assert_allclose(a1.obsm["X_umap"], a2.obsm["X_umap"])


def test_compute_umap_missing_neighbors_raises():
    a = ad.AnnData(X=np.zeros((10, 5), dtype="float32"))
    with pytest.raises(compute.NeighborsMissing):
        compute.compute_umap(a, min_dist=0.3, random_state=0)


def test_compute_phate_writes_obsm():
    a = _adata_with_neighbors()
    compute.compute_phate(a, use_rep="X_pca_harmony", knn=15, decay=40, random_state=0)
    assert a.obsm["X_phate"].shape == (60, 2)


def test_compute_phate_missing_rep_raises():
    a = _adata_with_neighbors()
    with pytest.raises(compute.RepMissing):
        compute.compute_phate(a, use_rep="X_absent", knn=15, decay=40, random_state=0)


def test_compute_phate_unavailable_raises(monkeypatch):
    a = _adata_with_neighbors()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "phate":
            raise ImportError("no phate")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(compute.PhateUnavailable):
        compute.compute_phate(a, use_rep="X_pca_harmony", knn=15, decay=40, random_state=0)


def test_compute_paga_writes_uns():
    a = _adata_with_neighbors()
    compute.compute_paga(a, groupby="cell_type")
    assert "paga" in a.uns
    assert "connectivities" in a.uns["paga"]


def test_compute_paga_missing_group_raises():
    a = _adata_with_neighbors()
    with pytest.raises(compute.GroupMissing):
        compute.compute_paga(a, groupby="not_a_col")


def test_resolve_paga_groupby_precedence():
    a = _adata_with_neighbors()
    # configured wins when present
    assert (
        compute.resolve_paga_groupby(a, "leiden", cell_type_key="cell_type", cluster_key="leiden")
        == "leiden"
    )
    # None -> cell_type when present
    assert (
        compute.resolve_paga_groupby(a, None, cell_type_key="cell_type", cluster_key="leiden")
        == "cell_type"
    )
    # cell_type absent -> cluster
    del a.obs["cell_type"]
    assert (
        compute.resolve_paga_groupby(a, None, cell_type_key="cell_type", cluster_key="leiden")
        == "leiden"
    )
    # neither -> None
    del a.obs["leiden"]
    assert (
        compute.resolve_paga_groupby(a, None, cell_type_key="cell_type", cluster_key="leiden")
        is None
    )
