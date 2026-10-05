"""Tests for annotation marker-panel figures (dotplot + feature grid)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.visualization.marker_panels import plot_marker_dotplot, plot_marker_feature_grid


def _adata(n=40, n_genes=8) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n, n_genes)).astype(float)
    a = ad.AnnData(X=X, var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
    a.layers["norm"] = X.copy()
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obs["cell_type"] = pd.Categorical(
        (["A"] * (n // 2)) + (["B"] * (n - n // 2)), categories=["A", "B"]
    )
    return a


def test_plot_marker_dotplot_writes_png_and_a_vector_twin(tmp_path):
    a = _adata()
    marker_panel = {"A": ["G0", "G1"], "B": ["G2", "G3"]}

    out = plot_marker_dotplot(
        a, marker_panel, groupby="cell_type", layer="norm", out_path=tmp_path / "dotplot.png"
    )

    assert out == tmp_path / "dotplot.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_marker_dotplot_swap_axes_puts_genes_on_rows(tmp_path):
    """swap_axes=True: genes as rows (grouped, boxed by panel), clusters as
    columns -- the layout used for a per-cluster marker-panel profile, as
    opposed to the default per-cell-type-marker layout."""
    a = _adata()
    marker_panel = {"A": ["G0", "G1"], "B": ["G2", "G3"]}

    out = plot_marker_dotplot(
        a,
        marker_panel,
        groupby="cell_type",
        layer="norm",
        out_path=tmp_path / "dotplot_swapped.png",
        swap_axes=True,
    )

    assert out is not None
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_marker_dotplot_drops_genes_absent_from_the_object(tmp_path):
    a = _adata()
    marker_panel = {"A": ["G0", "NOT_A_REAL_GENE"], "B": ["G2", "G3"]}

    out = plot_marker_dotplot(
        a, marker_panel, groupby="cell_type", layer="norm", out_path=tmp_path / "dotplot.png"
    )

    assert out is not None
    assert out.exists()


def test_plot_marker_dotplot_returns_none_when_no_panel_genes_present(tmp_path):
    a = _adata()
    marker_panel = {"A": ["NOPE1"], "B": ["NOPE2"]}

    out = plot_marker_dotplot(
        a, marker_panel, groupby="cell_type", layer="norm", out_path=tmp_path / "dotplot.png"
    )

    assert out is None
    assert not (tmp_path / "dotplot.png").exists()


def test_plot_marker_feature_grid_writes_png_and_a_vector_twin(tmp_path):
    a = _adata()
    markers = {"A": "G0", "B": "G2"}

    out = plot_marker_feature_grid(
        a, markers, basis="X_umap", layer="norm", out_path=tmp_path / "grid.png"
    )

    assert out == tmp_path / "grid.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_marker_feature_grid_skips_genes_absent_from_the_object(tmp_path):
    a = _adata()
    markers = {"A": "G0", "B": "NOT_A_REAL_GENE"}

    out = plot_marker_feature_grid(
        a, markers, basis="X_umap", layer="norm", out_path=tmp_path / "grid.png"
    )

    assert out is not None
    assert out.exists()


def test_plot_marker_feature_grid_returns_none_when_no_markers_present(tmp_path):
    a = _adata()
    markers = {"A": "NOPE1", "B": "NOPE2"}

    out = plot_marker_feature_grid(
        a, markers, basis="X_umap", layer="norm", out_path=tmp_path / "grid.png"
    )

    assert out is None
    assert not (tmp_path / "grid.png").exists()


def _categorized_adata(n=60, seed=0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(10)]
    X = rng.poisson(2.0, size=(n, len(genes))).astype(float)
    a = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    a.layers["norm"] = X.copy()
    a.obs["state"] = pd.Categorical(
        (["S1"] * (n // 3)) + (["S2"] * (n // 3)) + (["S3"] * (n - 2 * (n // 3))),
        categories=["S1", "S2", "S3"],
    )
    return a


def test_plot_categorized_dotplot_writes_png_and_a_vector_twin(tmp_path):
    from cellquorum.visualization.marker_panels import plot_categorized_dotplot

    a = _categorized_adata()
    categories = {
        "Category A": ["G0", "G1", "G2"],
        "Category B": ["G3", "G4"],
    }

    out = plot_categorized_dotplot(
        a, categories, groupby="state", layer="norm", out_path=tmp_path / "cat_dotplot.png"
    )

    assert out == tmp_path / "cat_dotplot.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_categorized_dotplot_drops_categories_with_no_present_genes(tmp_path):
    from cellquorum.visualization.marker_panels import plot_categorized_dotplot

    a = _categorized_adata()
    categories = {
        "Category A": ["G0", "G1"],
        "Ghost category": ["NOT_A_REAL_GENE", "ALSO_FAKE"],
    }

    out = plot_categorized_dotplot(
        a, categories, groupby="state", layer="norm", out_path=tmp_path / "cat_dotplot2.png"
    )

    assert out is not None
    assert out.exists()


def test_plot_categorized_dotplot_handles_a_gene_shared_across_categories(tmp_path):
    """A gene can legitimately appear in more than one category (e.g. COL1A1 in
    both a homeostatic/ECM panel and a myofibroblast panel) -- exactly the
    real Mehrara panel this figure was built for. That duplicates the gene's
    row label, which broke a .loc-based scalar lookup before this test."""
    from cellquorum.visualization.marker_panels import plot_categorized_dotplot

    a = _categorized_adata()
    categories = {
        "Category A": ["G0", "G1"],
        "Category B": ["G1", "G2"],  # G1 shared with Category A
    }

    out = plot_categorized_dotplot(
        a, categories, groupby="state", layer="norm", out_path=tmp_path / "cat_dotplot_dup.png"
    )

    assert out is not None
    assert out.exists()


def test_plot_categorized_dotplot_returns_none_when_no_gene_present(tmp_path):
    from cellquorum.visualization.marker_panels import plot_categorized_dotplot

    a = _categorized_adata()
    categories = {"Ghost": ["NOPE1", "NOPE2"]}

    out = plot_categorized_dotplot(
        a, categories, groupby="state", layer="norm", out_path=tmp_path / "cat_dotplot3.png"
    )

    assert out is None


# ---- lineage marker dotplot with dendrogram ----------------------------------------------


from cellquorum.visualization.marker_panels import (  # noqa: E402
    dendrogram_row_order,
    draw_lineage_marker_dotplot,
    marker_panel_statistics,
    plot_lineage_marker_dotplot,
    scale_columns_unit,
)


def _panel_adata() -> ad.AnnData:
    """Three groups, three genes with known means and detection rates."""
    X = np.zeros((9, 3), dtype=float)
    X[0:3, 0] = [2.0, 4.0, 0.0]  # group A: gene0 mean 2, detected in 2/3
    X[3:6, 1] = [1.0, 1.0, 1.0]  # group B: gene1 mean 1, detected in 3/3
    X[6:9, 2] = [3.0, 0.0, 0.0]  # group C: gene2 mean 1, detected in 1/3
    a = ad.AnnData(X=X, var=pd.DataFrame(index=["G0", "G1", "G2"]))
    a.layers["norm"] = X.copy()
    a.obs["ct"] = pd.Categorical(["A"] * 3 + ["B"] * 3 + ["C"] * 3, categories=["A", "B", "C"])
    return a


def test_marker_panel_statistics_reports_mean_and_percent_detected_per_group():
    s = marker_panel_statistics(
        _panel_adata(), {"A": ["G0"], "B": ["G1"], "C": ["G2"]}, groupby="ct", layer="norm"
    )
    assert s.groups == ["A", "B", "C"]
    assert s.genes == ["G0", "G1", "G2"]
    assert s.gene_group == ["A", "B", "C"]
    np.testing.assert_allclose(s.mean[0], [2.0, 0.0, 0.0])  # group A over G0, G1, G2
    np.testing.assert_allclose(s.pct[0], [200 / 3, 0.0, 0.0])
    np.testing.assert_allclose(s.pct[1], [0.0, 100.0, 0.0])
    np.testing.assert_allclose(s.pct[2], [0.0, 0.0, 100 / 3])


def test_marker_panel_statistics_drops_absent_genes_and_empty_groups():
    s = marker_panel_statistics(
        _panel_adata(),
        {"A": ["G0", "NOT_A_GENE"], "B": ["NOPE"], "D": ["G1"]},  # B: no usable gene; D: no cells
        groupby="ct",
        layer="norm",
    )
    assert s.genes == ["G0"] and s.gene_group == ["A"]
    assert s.groups == ["A", "B", "C"]  # rows are every observed group


def test_marker_panel_statistics_allows_a_gene_to_repeat_across_groups():
    s = marker_panel_statistics(
        _panel_adata(), {"A": ["G0", "G1"], "B": ["G1"]}, groupby="ct", layer="norm"
    )
    assert s.genes == ["G0", "G1", "G1"] and s.gene_group == ["A", "A", "B"]
    assert s.mean.shape == (3, 3)


def test_scale_columns_unit_maps_each_gene_to_zero_one_and_survives_a_constant_gene():
    m = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [2.0, 5.0, 8.0]])
    out = scale_columns_unit(m)
    np.testing.assert_allclose(out.min(axis=0), [0, 0, 0])
    np.testing.assert_allclose(out.max(axis=0), [1, 0, 1])  # constant column -> 0, never NaN
    assert np.isfinite(out).all()


def test_dendrogram_row_order_keeps_similar_groups_next_to_each_other():
    # rows 0 and 2 are nearly identical, rows 1 and 3 are nearly identical, and the two pairs differ
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.95, 0.05, 0.0], [0.0, 0.9, 1.0]])
    order, linkage_matrix = dendrogram_row_order(m)
    assert sorted(order) == [0, 1, 2, 3] and linkage_matrix.shape == (3, 4)
    pos = {row: i for i, row in enumerate(order)}
    assert abs(pos[0] - pos[2]) == 1 and abs(pos[1] - pos[3]) == 1


def _stats():
    return marker_panel_statistics(
        _panel_adata(), {"A": ["G0"], "B": ["G1"], "C": ["G2"]}, groupby="ct", layer="norm"
    )


def test_draw_lineage_marker_dotplot_colours_row_labels_from_the_given_palette():
    palette = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
    fig = draw_lineage_marker_dotplot(_stats(), palette=palette)
    texts = {
        t.get_text(): t.get_color() for ax in fig.axes for t in ax.texts if t.get_text() in palette
    }
    assert texts == palette


def test_draw_lineage_marker_dotplot_labels_do_not_sit_on_the_dendrogram():
    """Real flaw in the first version: row labels were drawn over the dendrogram. The dendrogram,
    the label column and the dot grid must occupy separate, non-overlapping horizontal bands."""
    fig = draw_lineage_marker_dotplot(
        _stats(), palette={"A": "#111111", "B": "#222222", "C": "#333333"}
    )
    fig.canvas.draw()
    by_name = {ax.get_label(): ax.get_position() for ax in fig.axes if ax.get_label()}
    assert by_name["dendrogram"].x1 <= by_name["row_labels"].x0 + 1e-6
    assert by_name["row_labels"].x1 <= by_name["dots"].x0 + 1e-6


def test_draw_lineage_marker_dotplot_size_legend_clears_the_gene_labels():
    """Real flaw: the bottom margin was a fixed 20% of the figure, so on a figure with few rows the
    size legend landed on top of the rotated gene labels."""
    rng = np.random.default_rng(1)
    genes = [f"LONGGENENAME{i}" for i in range(6)]
    a = ad.AnnData(X=rng.poisson(2.0, size=(30, 6)).astype(float), var=pd.DataFrame(index=genes))
    a.layers["norm"] = a.X.copy()
    a.obs["ct"] = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    stats = marker_panel_statistics(
        a, {"A": genes[:2], "B": genes[2:4], "C": genes[4:]}, groupby="ct", layer="norm"
    )
    fig = draw_lineage_marker_dotplot(
        stats, palette={"A": "#111111", "B": "#222222", "C": "#333333"}
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dots = next(ax for ax in fig.axes if ax.get_label() == "dots")
    labels_bottom = min(t.get_window_extent(renderer).y0 for t in dots.get_xticklabels())
    legend_top = fig.legends[0].get_window_extent(renderer).y1
    assert legend_top <= labels_bottom


def test_draw_lineage_marker_dotplot_header_uses_panel_labels():
    fig = draw_lineage_marker_dotplot(
        _stats(),
        palette={"A": "#111111", "B": "#222222", "C": "#333333"},
        panel_labels={"A": "Alpha"},
    )
    drawn = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "Alpha" in drawn and "B" in drawn  # unlabelled panels fall back to the group name


def test_plot_lineage_marker_dotplot_writes_png_and_pdf(tmp_path):
    out = plot_lineage_marker_dotplot(
        _panel_adata(),
        {"A": ["G0"], "B": ["G1"], "C": ["G2"]},
        groupby="ct",
        layer="norm",
        out_path=tmp_path / "lineage_dotplot.png",
    )
    assert out == tmp_path / "lineage_dotplot.png" and out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_lineage_marker_dotplot_returns_none_when_no_panel_gene_is_present(tmp_path):
    out = plot_lineage_marker_dotplot(
        _panel_adata(), {"A": ["ZZZ"]}, groupby="ct", layer="norm", out_path=tmp_path / "x.png"
    )
    assert out is None and not (tmp_path / "x.png").exists()


def test_dendrogram_row_order_clusters_on_correlation_not_magnitude():
    """The reference tree clusters on Pearson correlation of group-mean profiles, not on
    distance: two groups with the same expression shape but different magnitude belong together."""
    m = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.8], [10.0, 0.0, 2.0], [0.0, 9.0, 8.0]])
    order, _ = dendrogram_row_order(m)
    pos = {row: i for i, row in enumerate(order)}
    assert abs(pos[0] - pos[2]) == 1 and abs(pos[1] - pos[3]) == 1


def test_dendrogram_row_order_survives_a_constant_row():
    """Correlation is undefined for a flat profile; the tree must still be drawn."""
    m = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    order, z = dendrogram_row_order(m)
    assert sorted(order) == [0, 1, 2] and np.isfinite(z).all()
