"""Tests for biology-agnostic enrichment-viz plotting primitives."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from cellquorum.stages.comparative.enrichment.viz import plots


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
