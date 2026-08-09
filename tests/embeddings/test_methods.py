from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.embeddings.categorical_method import CategoricalEmbeddingMethod
from cellquorum.embeddings.config import MagicConfig, OverlayConfig
from cellquorum.embeddings.overlay_method import ContinuousOverlayMethod
from cellquorum.embeddings.paga_method import PagaMethod
from cellquorum.embeddings.phate_method import PhateMethod
from cellquorum.embeddings.umap_method import UmapMethod
from cellquorum.methods.base import MethodSkip


class _Ctx:
    random_seed = 1337


def _adata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((50, 8)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(50, 6)).astype("float32")
    a.obs["cell_type"] = ["A"] * 25 + ["B"] * 25
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    sc.pp.neighbors(a, use_rep="X_pca_harmony", random_state=0)
    return a


def _cfg(**kw):
    base = {
        "use_rep": "X_pca_harmony",
        "umap_min_dist": 0.3,
        "phate_knn": 15,
        "phate_decay": 40,
        "random_state": 1337,
        "paga_groupby": None,
        "cell_type_key": "cell_type",
        "cluster_key": "leiden",
    }
    base.update(kw)
    return base


def test_umap_method_writes_obsm():
    a = _adata()
    res = UmapMethod().run(a, _cfg(), _Ctx())
    assert not isinstance(res, MethodSkip)
    assert "X_umap" in res.adata.obsm


def test_umap_method_skips_without_neighbors():
    a = ad.AnnData(X=np.zeros((10, 4), dtype="float32"))
    res = UmapMethod().run(a, _cfg(), _Ctx())
    assert isinstance(res, MethodSkip)


def test_phate_method_writes_obsm():
    a = _adata()
    res = PhateMethod().run(a, _cfg(), _Ctx())
    assert not isinstance(res, MethodSkip)
    assert "X_phate" in res.adata.obsm


def test_paga_method_resolves_cell_type():
    a = _adata()
    res = PagaMethod().run(a, _cfg(), _Ctx())
    assert not isinstance(res, MethodSkip)
    assert "paga" in res.adata.uns
    assert res.metrics["groupby"] == "cell_type"


def test_paga_method_skips_when_no_group():
    a = _adata()
    del a.obs["cell_type"]
    res = PagaMethod().run(a, _cfg(cluster_key="leiden"), _Ctx())
    assert isinstance(res, MethodSkip)


# --- Task 6: Render methods ---


class _CtxPaths:
    random_seed = 1337

    def __init__(self, tmp):
        self.paths = type("P", (), {"figures": str(tmp), "results": str(tmp)})()


def _adata_rendered():
    a = _adata()
    sc.tl.umap(a, random_state=0)
    sc.tl.paga(a, groups="cell_type")
    a.var_names = [f"GENE_{i}" for i in range(a.n_vars)]
    return a


def test_categorical_method_renders_figures(tmp_path):
    a = _adata_rendered()
    cfg = _cfg(embeddings=["umap"], figure_formats=["png"], dpi=80)
    res = CategoricalEmbeddingMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_figures"] >= 1
    assert any(p.suffix == ".png" for art in res.artifacts for p in [Path(art.path)])


def test_overlay_method_renders_gene(tmp_path):
    a = _adata_rendered()
    cfg = _cfg(
        embeddings=["umap"],
        figure_formats=["png"],
        dpi=80,
        overlay=OverlayConfig(genes=["GENE_0"]),
        magic=MagicConfig(enabled=False),
    )
    res = ContinuousOverlayMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_figures"] >= 1


def test_overlay_method_skips_when_nothing_requested(tmp_path):
    a = _adata_rendered()
    cfg = _cfg(embeddings=["umap"], overlay=OverlayConfig(), magic=MagicConfig(enabled=False))
    res = ContinuousOverlayMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert isinstance(res, MethodSkip)


def test_categorical_method_skips_without_embedding(tmp_path):
    a = _adata()  # no X_umap computed
    cfg = _cfg(embeddings=["umap"])
    res = CategoricalEmbeddingMethod().run(a, cfg, _CtxPaths(tmp_path))
    assert isinstance(res, MethodSkip)
