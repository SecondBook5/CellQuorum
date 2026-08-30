import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.stages.integration.embeddings.stage import EmbeddingsStage
from cellquorum.methods.registry import METHOD_REGISTRY


def test_all_methods_registered():
    for name in ["umap", "phate", "paga", "categorical_embedding", "continuous_overlay"]:
        assert METHOD_REGISTRY.has("embeddings", name)


def test_stage_augment_injects_method_list_and_keys():
    stage = EmbeddingsStage()

    class _Cfg:
        class embeddings:
            enabled = True
            use_rep = "X_pca_harmony"
            umap_min_dist = 0.3
            phate_knn = 15
            phate_decay = 40
            paga_groupby = None
            paga_threshold = 0.2
            random_state = 0
            embeddings = ["umap", "phate"]
            figure_formats = ["png"]
            dpi = 80
            overlay = {"genes": []}
            magic = {"enabled": False}

        class annotation:
            key_added = "cell_type"

        class clustering:
            key_added = "leiden"

    class _Ctx:
        config = _Cfg()

    augmented = stage._augment_config(_Ctx(), {})
    method_names = [m["method"] for m in augmented["methods"]]
    assert method_names == ["umap", "phate", "paga", "categorical_embedding", "continuous_overlay"]
    assert augmented["cell_type_key"] == "cell_type"
    assert augmented["cluster_key"] == "leiden"
    assert augmented["use_rep"] == "X_pca_harmony"


def test_stage_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((50, 8)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(50, 6)).astype("float32")
    a.obs["cell_type"] = ["A"] * 25 + ["B"] * 25
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    a.var_names = [f"GENE_{i}" for i in range(a.n_vars)]
    sc.pp.neighbors(a, use_rep="X_pca_harmony", random_state=0)

    from cellquorum.stages.integration.embeddings.config import EmbeddingsConfig

    class _Cfg:
        embeddings = EmbeddingsConfig(figure_formats=["png"], dpi=80, embeddings=["umap"])

        class annotation:
            key_added = "cell_type"

        class clustering:
            key_added = "leiden"

    class _Ctx:
        config = _Cfg()
        random_seed = 1337
        paths = type("P", (), {"figures": str(tmp_path), "results": str(tmp_path)})()

        def require_adata(self):
            return a

    result = EmbeddingsStage().run(_Ctx())
    assert "X_umap" in result.adata.obsm
    assert "paga" in result.adata.uns
    assert result.metrics["n_methods"] == 5
