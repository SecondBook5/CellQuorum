import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.embeddings.phate_method import PhateMethod
from cellquorum.embeddings.umap_method import UmapMethod


class _Ctx:
    random_seed = 1337


def _adata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((50, 8)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(50, 6)).astype("float32")
    sc.pp.neighbors(a, use_rep="X_pca_harmony", random_state=0)
    return a


def _cfg():
    return {
        "use_rep": "X_pca_harmony",
        "umap_min_dist": 0.3,
        "phate_knn": 15,
        "phate_decay": 40,
        "random_state": 1337,
    }


def test_umap_seed_reproducible():
    a1, a2 = _adata(), _adata()
    UmapMethod().run(a1, _cfg(), _Ctx())
    UmapMethod().run(a2, _cfg(), _Ctx())
    np.testing.assert_allclose(a1.obsm["X_umap"], a2.obsm["X_umap"])


def test_phate_seed_reproducible():
    a1, a2 = _adata(), _adata()
    PhateMethod().run(a1, _cfg(), _Ctx())
    PhateMethod().run(a2, _cfg(), _Ctx())
    np.testing.assert_allclose(a1.obsm["X_phate"], a2.obsm["X_phate"])
