import anndata as ad
import numpy as np
import scanpy as sc
from matplotlib.figure import Figure

from cellquorum.stages.integration.embeddings import plots
from cellquorum.stages.integration.embeddings.plots import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import apply_theme


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


def test_categorical_embedding_without_paga_renders_scatter_only():
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((60, 8)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(60, 6)).astype("float32")
    a.obsm["X_umap"] = rng.normal(size=(60, 2)).astype("float32")
    a.obs["cell_type"] = ["A"] * 30 + ["B"] * 30
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    # Note: NO paga computed in a.uns["paga"]
    fig = plots.categorical_embedding(
        a, "cell_type", basis="X_umap", axis_labels=("UMAP1", "UMAP2")
    )
    assert isinstance(fig, Figure)
    # Scatter still rendered (collections are the scatter point collections)
    ax = fig.axes[0]
    assert len(ax.collections) >= 1
    # But NO PAGA edge lines were drawn
    assert len(ax.get_lines()) == 0


def test_categorical_embedding_cluster_labels_rename_legend_and_inplot_text():
    """Bare cluster ids ("0", "1") are meaningless on a figure without a separate

    legend to cross-reference -- cluster_labels renames the in-plot text and the
    legend, without touching which cells belong to which group (grouping, PAGA
    connectivity indexing, and the palette all stay keyed on the raw id).
    """
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((60, 8)).astype("float32"))
    a.obsm["X_umap"] = rng.normal(size=(60, 2)).astype("float32")
    a.obs["leiden"] = ["0"] * 30 + ["1"] * 30
    a.obs["leiden"] = a.obs["leiden"].astype("category")
    fig = plots.categorical_embedding(
        a,
        "leiden",
        basis="X_umap",
        axis_labels=("UMAP1", "UMAP2"),
        cluster_labels={"0": "Papillary", "1": "Reticular"},
    )
    ax = fig.axes[0]
    in_plot_texts = {t.get_text() for t in ax.texts}
    assert "Papillary" in in_plot_texts
    assert "Reticular" in in_plot_texts
    assert "0" not in in_plot_texts
    assert "1" not in in_plot_texts
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(label.startswith("Papillary") for label in legend_labels)
    assert any(label.startswith("Reticular") for label in legend_labels)


def test_continuous_overlay_returns_figure():
    coords = np.random.default_rng(0).random((50, 2))
    values = np.random.default_rng(1).random(50)
    fig = plots.continuous_overlay(coords, values, title="GENE_A", axis_labels=("UMAP1", "UMAP2"))
    assert isinstance(fig, Figure)


def _score_by_condition_adata():
    rng = np.random.default_rng(0)
    n = 60
    a = ad.AnnData(X=rng.random((n, 8)).astype("float32"))
    a.obsm["X_umap"] = rng.normal(size=(n, 2)).astype("float32")
    a.obs["fibroblast_subcluster"] = ["0"] * 30 + ["1"] * 30
    a.obs["fibroblast_subcluster"] = a.obs["fibroblast_subcluster"].astype("category")
    a.obs["condition"] = (["Normal"] * 15 + ["Lymphedema"] * 15) * 2
    return a


def test_plot_score_by_condition_returns_three_panels():
    a = _score_by_condition_adata()
    score = np.random.default_rng(2).random(a.n_obs)
    fig = plots.plot_score_by_condition(
        a,
        score=score,
        score_label="proliferating (UCell)",
        cluster_key="fibroblast_subcluster",
        basis="X_umap",
        axis_labels=("UMAP1", "UMAP2"),
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
    )
    assert isinstance(fig, Figure)
    # 3 content panels (state, control, case) + 1 shared colorbar axes.
    assert len(fig.axes) == 4


def test_plot_score_by_condition_shares_color_scale_across_condition_panels():
    """Panels 2 and 3 use the SAME vmin/vmax -- otherwise a dark red dot in one

    panel and a dark red dot in the other can mean different score values,
    silently defeating the whole point of a shared-scale comparison figure.
    """
    a = _score_by_condition_adata()
    score = np.random.default_rng(2).random(a.n_obs)
    fig = plots.plot_score_by_condition(
        a,
        score=score,
        score_label="proliferating (UCell)",
        cluster_key="fibroblast_subcluster",
        basis="X_umap",
        axis_labels=("UMAP1", "UMAP2"),
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
    )
    control_scatter = next(c for c in fig.axes[1].collections if hasattr(c, "get_clim"))
    case_scatter = next(c for c in fig.axes[2].collections if hasattr(c, "get_clim"))
    assert control_scatter.get_clim() == case_scatter.get_clim()


def test_plot_score_by_condition_cluster_labels_rename_state_panel_text():
    a = _score_by_condition_adata()
    score = np.random.default_rng(2).random(a.n_obs)
    fig = plots.plot_score_by_condition(
        a,
        score=score,
        score_label="proliferating (UCell)",
        cluster_key="fibroblast_subcluster",
        basis="X_umap",
        axis_labels=("UMAP1", "UMAP2"),
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
        cluster_labels={"0": "Papillary", "1": "Reticular"},
    )
    state_texts = {t.get_text() for t in fig.axes[0].texts}
    assert "Papillary" in state_texts
    assert "Reticular" in state_texts


def test_save_figure_writes_both_formats(tmp_path):
    apply_theme("notebook")
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "figs", "demo", formats=("pdf", "png"), dpi=100)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() for p in paths)
    arts = figure_artifacts(paths, name="embedding_figure", description="demo")
    assert all(a.kind == "figure" for a in arts)


def test_paga_overlay_edges_use_categorical_order():
    """PAGA edges must connect the correct centroids per categorical order, not sorted."""
    rng = np.random.default_rng(42)
    # 3 groups with non-alphabetical categorical order: ['c','a','b'] (sorted is ['a','b','c'])
    a = ad.AnnData(X=rng.random((90, 8)).astype("float32"))
    # Place each group at a distinct, well-separated location.
    umap_coords = np.vstack(
        [
            np.c_[rng.normal(0, 0.1, 30), rng.normal(0, 0.1, 30)],  # group 'c' at (0,0)
            np.c_[rng.normal(5, 0.1, 30), rng.normal(0, 0.1, 30)],  # group 'a' at (5,0)
            np.c_[rng.normal(0, 0.1, 30), rng.normal(5, 0.1, 30)],  # group 'b' at (0,5)
        ]
    )
    a.obsm["X_umap"] = umap_coords.astype("float32")
    # Categorical with order ['c','a','b'] (NOT sorted ['a','b','c'])
    a.obs["leiden"] = ["c"] * 30 + ["a"] * 30 + ["b"] * 30
    a.obs["leiden"] = a.obs["leiden"].astype("category").cat.reorder_categories(["c", "a", "b"])
    # Build connectivity matrix in categorical order: rows/cols are ['c','a','b']
    # Strong edge only between 'a' (idx=1) and 'b' (idx=2) in categorical order.
    # Sorted order would be ['a','b','c'], so idx[1]='b', idx[2]='c' — wrong!
    conn = np.zeros((3, 3), dtype="float32")
    conn[1, 2] = 0.9  # 'a' <-> 'b' in categorical order
    conn[2, 1] = 0.9
    a.uns["paga"] = {"connectivities": conn}

    fig = plots.categorical_embedding(
        a, "leiden", basis="X_umap", axis_labels=("U1", "U2"), paga_threshold=0.5
    )
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 1, "Expected exactly one PAGA edge above threshold"
    line = lines[0]
    x_data, y_data = line.get_data()
    # Centroids: 'c' at ~(0,0), 'a' at ~(5,0), 'b' at ~(0,5)
    # The edge MUST connect 'a' (5,0) and 'b' (0,5), NOT 'b' and 'c'.
    # Check both endpoints: one near (5,0), one near (0,5).
    endpoints = [(x_data[0], y_data[0]), (x_data[1], y_data[1])]
    near_a = any(abs(x - 5) < 1 and abs(y - 0) < 1 for x, y in endpoints)
    near_b = any(abs(x - 0) < 1 and abs(y - 5) < 1 for x, y in endpoints)
    assert near_a and near_b, f"Edge endpoints {endpoints} don't connect 'a' (5,0) and 'b' (0,5)"


def test_categorical_embedding_declared_but_empty_category():
    """A declared-but-empty category must not crash or misalign PAGA edges.

    scanpy indexes uns['paga']['connectivities'] by cat.codes, so the matrix
    aligns to the FULL category order — filtering to present categories would
    re-introduce an index mismatch (IndexError or misplaced edges). This uses
    the REAL sc.tl.paga so the connectivity indexing is ground truth.
    """
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((60, 6)).astype("float32"))
    a.obsm["X_pca_harmony"] = rng.normal(size=(60, 5)).astype("float32")
    # Categories B and D are declared but have zero cells.
    a.obs["leiden"] = ["A"] * 30 + ["C"] * 30
    a.obs["leiden"] = a.obs["leiden"].astype("category").cat.set_categories(["A", "B", "C", "D"])
    sc.pp.neighbors(a, use_rep="X_pca_harmony", n_neighbors=10, random_state=0)
    sc.tl.umap(a, random_state=0)
    sc.tl.paga(a, groups="leiden")
    # Must render without an IndexError and draw the two present groups.
    fig = plots.categorical_embedding(a, "leiden", basis="X_umap", axis_labels=("U1", "U2"))
    ax = fig.axes[0]
    total_points = sum(int(c.get_offsets().shape[0]) for c in ax.collections)
    assert total_points >= 60, "present-group points must be drawn (empty cats add none)"


def test_categorical_embedding_integer_categorical_renders_points():
    """An integer-typed categorical group column must still plot its cells.

    Masking casts the column to str, so category values must be compared as
    strings; otherwise every mask is all-False and the figure is silently blank.
    """
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((60, 6)).astype("float32"))
    a.obsm["X_umap"] = rng.normal(size=(60, 2)).astype("float32")
    import pandas as pd

    a.obs["leiden"] = pd.Categorical([0] * 20 + [1] * 20 + [2] * 20, categories=[0, 1, 2])
    fig = plots.categorical_embedding(a, "leiden", basis="X_umap", axis_labels=("U1", "U2"))
    ax = fig.axes[0]
    total_points = sum(int(c.get_offsets().shape[0]) for c in ax.collections)
    assert total_points == 60, f"expected 60 plotted cells, got {total_points} (blank-figure bug)"


def test_missing_displayed_category_cannot_silently_remove_cells():
    import pandas as pd

    data = ad.AnnData(np.ones((3, 1)))
    data.obs["group"] = pd.Categorical(["A", "B", None])
    data.obsm["X_umap"] = np.array([[0, 0], [1, 1], [2, 2]])
    kwargs = dict(basis="X_umap", axis_labels=("UMAP1", "UMAP2"), paga_overlay=False)
    fig = plots.categorical_embedding(data, "group", **kwargs)
    assert sum(len(c.get_offsets()) for c in fig.axes[0].collections) == 3
    assert any(
        "Unassigned (missing)" in text.get_text() for text in fig.axes[0].get_legend().get_texts()
    )
    fig = plots.categorical_embedding(
        data, "group", cell_mask=np.array([True, True, False]), **kwargs
    )
    assert isinstance(fig, Figure)


def test_unassigned_display_keeps_paga_indices_and_source_labels():
    import pandas as pd

    data = ad.AnnData(np.ones((3, 1)))
    data.obs["group"] = pd.Categorical(["B", "A", None], categories=["B", "A"])
    data.obsm["X_umap"] = np.array([[0, 0], [1, 1], [5, 5]])
    data.uns["paga"] = {"connectivities": np.array([[0, 1], [1, 0]])}
    original = data.obs["group"].copy()
    fig = plots.categorical_embedding(data, "group", basis="X_umap", axis_labels=("x", "y"))
    pd.testing.assert_series_equal(data.obs["group"], original)
    edges = [line for line in fig.axes[0].lines if len(line.get_xdata()) == 2]
    assert any(set(line.get_xdata()) == {0, 1} for line in edges)
    assert all(5 not in line.get_xdata() for line in edges)
