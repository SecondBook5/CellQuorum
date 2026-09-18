"""Tests for biology-agnostic enrichment-viz plotting primitives."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from cellquorum.stages.comparative.enrichment.viz import plots
from cellquorum.stages.comparative.enrichment.viz.plots import _wrap_label


def test_signed_norm_straddles_zero_even_when_all_positive():
    norm = plots.signed_norm(np.array([0.1, 0.5, 2.0]))
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vmin < 0 < norm.vmax


def test_signed_norm_handles_all_zero():
    norm = plots.signed_norm(np.array([0.0, 0.0]))
    assert norm.vmin < 0 < norm.vmax


def test_pvalue_to_stars():
    assert plots.pvalue_to_stars(1e-5) == "****"
    assert plots.pvalue_to_stars(1e-4) == "***"
    assert plots.pvalue_to_stars(5e-3) == "**"
    assert plots.pvalue_to_stars(2e-2) == "*"
    assert plots.pvalue_to_stars(0.2) == ""
    assert plots.pvalue_to_stars(float("nan")) == ""


def test_select_top_bottom_is_deterministic_and_symmetric():
    df = pd.DataFrame({"src": list("abcdef"), "score": [3.0, -2.0, 1.0, -4.0, 5.0, 0.5]})
    out = plots.select_top_bottom(df, "score", k=2)
    # top-2 (5.0, 3.0) + bottom-2 (-4.0, -2.0); returned ascending by score.
    assert list(out["score"]) == [-4.0, -2.0, 3.0, 5.0]
    # deterministic across calls
    out2 = plots.select_top_bottom(df, "score", k=2)
    assert list(out2["src"]) == list(out["src"])


def test_select_top_bottom_no_duplicate_when_k_exceeds_rows():
    df = pd.DataFrame({"src": ["a", "b"], "score": [1.0, -1.0]})
    out = plots.select_top_bottom(df, "score", k=5)
    assert len(out) == 2  # no row counted twice


def test_select_top_bottom_breaks_ties_by_index():
    # Test with tied values - should break ties by index
    df = pd.DataFrame({"src": list("abcdef"), "score": [1.0, 1.0, -2.0, -2.0, 3.0, 0.0]})
    out = plots.select_top_bottom(df, "score", k=2)
    # bottom-2: indices 2,3 both have -2.0, should pick indices 2,3
    # top-2: indices 0,1 have 1.0, index 4 has 3.0; should pick indices 0,4 or 1,4 then 4
    # Actually: top-2 by value are [3.0 at idx 4, then tied 1.0 at idx 0,1]
    # So we get idx 4 (3.0) and idx 0 (1.0, lower index wins tie)
    # bottom-2 by value are [-2.0 at idx 2,3]
    # So we get idx 2,3 (both -2.0, lower indices)
    assert len(out) == 4
    # Verify determinism: shuffle input and check same output
    df_shuffled = df.iloc[[5, 2, 4, 0, 3, 1]].copy()
    out_shuffled = plots.select_top_bottom(df_shuffled, "score", k=2)
    # Both should select the same indices
    assert set(out.index) == set(out_shuffled.index)
    # And produce identical ordering
    pd.testing.assert_frame_equal(out, out_shuffled)


def test_diverging_bar_returns_axes():
    df = pd.DataFrame(
        {
            "source": [f"S{i}" for i in range(6)],
            "score": [3, -2, 1, -4, 5, -0.5],
            "padj": [1e-5, 0.2, 0.03, 1e-3, 1e-6, 0.5],
        }
    )
    ax = plots.diverging_bar(df, value_col="score", label_col="source", pvalue_col="padj", top_k=3)
    assert ax is not None
    # axvline(0) present
    assert any(np.allclose(line.get_xdata(), [0, 0]) for line in ax.get_lines())
    plt.close("all")


def test_activity_dotplot_returns_axes():
    df = pd.DataFrame(
        {
            "source": [f"S{i}" for i in range(4)],
            "score": [1.0, -1.0, 2.0, -0.5],
            "padj": [0.01, 0.2, 1e-4, 0.5],
        }
    )
    ax = plots.activity_dotplot(
        df, value_col="score", label_col="source", pvalue_col="padj", top_k=2
    )
    assert ax is not None
    plt.close("all")


def test_running_es_curve_returns_three_track_figure():
    n = 50
    df = pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "running_es": np.concatenate([np.linspace(0, 0.6, 25), np.linspace(0.6, -0.1, 25)]),
            "hit": ([1, 0, 0, 0, 1] * 10),
            "metric": np.linspace(3, -3, n),
        }
    )
    fig = plots.running_es_curve(df, title="S1")
    assert len(fig.axes) == 3
    plt.close("all")


def test_ora_barplot_facets_by_direction():
    df = pd.DataFrame(
        {
            "source": [f"S{i}" for i in range(4)],
            "direction": ["up", "up", "down", "down"],
            "count": [10, 5, 8, 3],
            "padj": [1e-3, 0.02, 1e-4, 0.04],
        }
    )
    fig = plots.ora_barplot(
        df, count_col="count", label_col="source", padj_col="padj", facet_col="direction", top_k=5
    )
    assert fig is not None
    plt.close("all")


def test_ora_barplot_facet_colorbars_do_not_overlap_the_next_facet():
    """Each facet's per-panel colorbar is added to the shared figure AFTER

    plt.subplots() has already fixed every axes' position -- without a
    layout engine, matplotlib doesn't reflow neighboring axes to make room,
    so a facet's colorbar is drawn straddling the boundary into the next
    facet's panel. Confirmed live: the real hallmark up/down figure has the
    "down" panel's padj colorbar tick labels rendered directly on top of the
    "up" panel's gene-set-name bar labels.
    """
    df = pd.DataFrame(
        {
            "source": [f"SOME_VERY_LONG_HALLMARK_GENE_SET_NAME_{i}" for i in range(6)],
            "direction": ["up", "up", "up", "down", "down", "down"],
            "count": [10, 5, 3, 8, 6, 3],
            "padj": [1e-3, 0.02, 0.03, 1e-4, 0.01, 0.04],
        }
    )
    fig = plots.ora_barplot(
        df, count_col="count", label_col="source", padj_col="padj", facet_col="direction", top_k=5
    )
    fig.canvas.draw()

    boxes = [ax.get_position() for ax in fig.axes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(
                boxes[j]
            ), f"axes {i} and {j} overlap: {boxes[i]} vs {boxes[j]}"
    plt.close("all")


def test_ora_barplot_long_labels_do_not_overflow_into_the_previous_facets_colorbar():
    """Y-tick labels are right-anchored text, not clipped to their axes box.

    A long gene-set name (Reactome/Hallmark names routinely run 30+ chars,
    e.g. "EPITHELIAL_MESENCHYMAL_TRANSITION") renders far enough left of its
    anchor point to land on top of the PREVIOUS facet's colorbar -- axes
    positions never overlap (already covered by a separate geometric test),
    but unwrapped label text does. Confirmed live against the real hallmark
    ORA barplot: this exact label overlaps the "down" panel's colorbar tick
    labels two columns to its left.
    """
    long_label = "EPITHELIAL_MESENCHYMAL_TRANSITION"
    df = pd.DataFrame(
        {
            "source": [long_label]
            + [f"UP_SET_{i}" for i in range(5)]
            + [f"DOWN_SET_{i}" for i in range(6)],
            "direction": ["up"] * 6 + ["down"] * 6,
            "count": [31, 15, 15, 14, 12, 12, 7, 6, 6, 5, 5, 4],
            "padj": [1e-4, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    fig = plots.ora_barplot(
        df, count_col="count", label_col="source", padj_col="padj", facet_col="direction", top_k=12
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Facet order is alphabetical ("down" before "up"): axes = [down_bars,
    # up_bars, down_cbar, up_cbar], per _facet_bar_or_dot's construction.
    down_bars_ax, up_bars_ax, down_cbar_ax, _up_cbar_ax = fig.axes
    down_cbar_right_edge = down_cbar_ax.get_position().x1

    # Now wrapped onto multiple lines (a separate, later fix), so it no longer
    # matches the raw source string exactly -- match on the unwrapped text.
    target = next(
        t
        for t in up_bars_ax.get_yticklabels()
        if t.get_text().replace("\n", "").replace(" …", "") == long_label
    )
    bbox_fig = target.get_window_extent(renderer).transformed(fig.transFigure.inverted())

    assert bbox_fig.x0 >= down_cbar_right_edge, (
        f"label {long_label!r} starts at x={bbox_fig.x0:.3f}, left of the "
        f"'down' colorbar's right edge at x={down_cbar_right_edge:.3f} -- it "
        "overlaps the previous facet"
    )
    plt.close("all")


def test_wrap_label_caps_even_an_extreme_label_at_two_lines():
    """The real reactome ORA table's longest label (141 chars) must never

    produce more than 2 lines -- textwrap alone would keep wrapping onto a
    4th or 5th line for something this long, eating vertical space the
    figure doesn't have.
    """
    extreme = (
        "REACTOME_TP53_REGULATES_TRANSCRIPTION_OF_SEVERAL_ADDITIONAL_CELL_DEATH_GENES_"
        "WHOSE_SPECIFIC_ROLES_IN_P53_DEPENDENT_APOPTOSIS_REMAIN_UNCERTAIN"
    )
    wrapped = _wrap_label(extreme)
    assert wrapped.count("\n") <= 1


def test_wrap_label_leaves_short_labels_untouched():
    assert _wrap_label("HYPOXIA") == "HYPOXIA"


def test_ora_barplot_wraps_reactome_length_labels_instead_of_overflowing():
    """constrained_layout alone (the earlier fix) only holds up to the ~34-char

    synthetic label used in the test above it -- real Reactome names run to
    80+ characters (confirmed against the actual pipeline output:
    REACTOME_TRANSPORT_OF_BILE_SALTS_AND_ORGANIC_ACIDS_METAL_IONS_AND_AMINE_COMPOUNDS
    is 81 chars), and at that length the "up" panel's labels bled across the
    shared middle colorbar into the "down" panel on the real hallmark/reactome
    ORA figure. Wrapping the label text onto multiple lines keeps each line
    short enough for constrained_layout to actually separate the facets.
    """
    long_label = "REACTOME_TRANSPORT_OF_BILE_SALTS_AND_ORGANIC_ACIDS_METAL_IONS_AND_AMINE_COMPOUNDS"
    df = pd.DataFrame(
        {
            "source": [long_label]
            + [f"UP_SET_{i}" for i in range(5)]
            + [f"DOWN_SET_{i}" for i in range(6)],
            "direction": ["up"] * 6 + ["down"] * 6,
            "count": [31, 15, 15, 14, 12, 12, 7, 6, 6, 5, 5, 4],
            "padj": [1e-4, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    fig = plots.ora_barplot(
        df, count_col="count", label_col="source", padj_col="padj", facet_col="direction", top_k=12
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    down_bars_ax, up_bars_ax, down_cbar_ax, _up_cbar_ax = fig.axes
    down_cbar_right_edge = down_cbar_ax.get_position().x1

    target = next(
        t for t in up_bars_ax.get_yticklabels() if long_label.split("_")[0] in t.get_text()
    )
    text = target.get_text()
    assert "\n" in text, "a label this long must be wrapped onto multiple lines"
    assert text.count("\n") <= 1, f"wrapped label must be capped at 2 lines, got: {text!r}"
    bbox_fig = target.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert bbox_fig.x0 >= down_cbar_right_edge, (
        f"wrapped label still starts at x={bbox_fig.x0:.3f}, left of the "
        f"'down' colorbar's right edge at x={down_cbar_right_edge:.3f}"
    )
    plt.close("all")


def test_ora_dotplot_returns_figure():
    df = pd.DataFrame(
        {
            "source": [f"S{i}" for i in range(4)],
            "direction": ["up", "up", "down", "down"],
            "count": [10, 5, 8, 3],
            "gene_ratio": [0.2, 0.1, 0.16, 0.06],
            "padj": [1e-3, 0.02, 1e-4, 0.04],
        }
    )
    fig = plots.ora_dotplot(
        df,
        ratio_col="gene_ratio",
        count_col="count",
        padj_col="padj",
        label_col="source",
        facet_col="direction",
        top_k=5,
    )
    assert fig is not None
    plt.close("all")


def test_annotated_clustermap_with_and_without_strip():
    mat = pd.DataFrame(
        np.random.default_rng(0).normal(size=(6, 5)),
        index=[f"src{i}" for i in range(6)],
        columns=[f"samp{j}" for j in range(5)],
    )
    grid = plots.annotated_clustermap(mat, col_colors=None, top_n=4)
    assert grid is not None
    plt.close("all")
    colors = pd.Series(["red", "red", "blue", "blue", "red"], index=mat.columns)
    grid2 = plots.annotated_clustermap(mat, col_colors=colors, top_n=None)
    assert grid2 is not None
    plt.close("all")


def test_cross_group_dotplot_returns_figure():
    df = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "source": ["S1", "S2", "S1", "S2"],
            "mean_score": [1.0, -0.5, 0.2, 2.0],
        }
    )
    fig = plots.cross_group_dotplot(
        df, row_col="source", col_col="cell_type", value_col="mean_score", top_k=5
    )
    assert fig is not None
    plt.close("all")


def test_cross_group_dotplot_col_labels_rename_the_x_axis_only():
    """Bare cluster ids ("0", "1") are meaningless without cross-referencing --

    col_labels renames the x-axis text for display, same convention as every
    other cluster_labels site in the codebase (de_heatmap, composition,
    categorical_embedding). Column selection/ordering still runs on the raw
    ids -- only the tick text changes.
    """
    df = pd.DataFrame(
        {
            "cell_type": ["0", "0", "1", "1"],
            "source": ["S1", "S2", "S1", "S2"],
            "mean_score": [1.0, -0.5, 0.2, 2.0],
        }
    )
    fig = plots.cross_group_dotplot(
        df,
        row_col="source",
        col_col="cell_type",
        value_col="mean_score",
        top_k=5,
        col_labels={"0": "Papillary", "1": "Reticular"},
    )
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Papillary", "Reticular"]
    plt.close("all")


def test_annotated_clustermap_figsize_is_capped_for_a_large_matrix():
    """A reactome-scale matrix (hundreds of pathways) must not blow the figure

    up to dozens of inches tall -- annotated_clustermap's own figsize formula
    grows without bound (0.3in per row), so a real large collection produces
    an unusably huge figure. A hard ceiling keeps it viewable regardless of
    how many rows top_n lets through.
    """
    mat = pd.DataFrame(
        np.random.default_rng(0).normal(size=(200, 6)),
        index=[f"src{i}" for i in range(200)],
        columns=[f"samp{j}" for j in range(6)],
    )
    grid = plots.annotated_clustermap(mat, col_colors=None, top_n=200)
    height_in = grid.fig.get_size_inches()[1]
    assert height_in <= 20.0
    plt.close("all")
