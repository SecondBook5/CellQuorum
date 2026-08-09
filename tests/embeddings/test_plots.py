import anndata as ad
import numpy as np
import scanpy as sc
from matplotlib.figure import Figure

from cellquorum.embeddings import plots
from cellquorum.embeddings.save import apply_theme, figure_artifacts, save_figure


def _adata_umap_paga():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((60, 8)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(60, 6)).astype("float32")
    a.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    sc.pp.neighbors(a, use_rep="X_pca_harmony", random_state=0)
    sc.tl.umap(a, random_state=0)
    sc.tl.paga(a, groups="cell_type")
    return a


def test_embedding_registry_keys():
    assert plots.EMBEDDING_REGISTRY["umap"]["obsm"] == "X_umap"
    assert plots.EMBEDDING_REGISTRY["phate"]["obsm"] == "X_phate"
    assert plots.EMBEDDING_REGISTRY["umap"]["axis"] == ("UMAP1", "UMAP2")


def test_categorical_embedding_with_paga_returns_figure():
    a = _adata_umap_paga()
    fig = plots.categorical_embedding(
        a, "cell_type", basis="X_umap", axis_labels=("UMAP1", "UMAP2"), paga_threshold=0.0
    )
    assert isinstance(fig, Figure)
    # PAGA overlay drew line(s): at least one Line2D on the axes.
    ax = fig.axes[0]
    assert len(ax.get_lines()) >= 1


def test_continuous_overlay_returns_figure():
    coords = np.random.default_rng(0).random((50, 2))
    values = np.random.default_rng(1).random(50)
    fig = plots.continuous_overlay(coords, values, title="GENE_A", axis_labels=("UMAP1", "UMAP2"))
    assert isinstance(fig, Figure)


def test_save_figure_writes_both_formats(tmp_path):
    apply_theme()
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "figs", "demo", formats=("pdf", "png"), dpi=100)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() for p in paths)
    arts = figure_artifacts(paths, name="embedding_figure", description="demo")
    assert all(a.kind == "figure" for a in arts)
