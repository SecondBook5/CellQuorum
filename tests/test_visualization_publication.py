"""Tests for reusable publication plotting primitives."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import (
    LE_RED,
    NORMAL_BLUE,
    categorical_embedding,
    condition_palette,
    pvalue_to_stars,
    set_publication_style,
    violin_with_stats,
)


def test_condition_palette_maps_case_and_control() -> None:
    # removed: biology-free consolidation (#152) — condition_palette is now the
    # canonical case/control-keyed mapping; extras fall back to the categorical
    # palette rather than a hardcoded disease-condition table.
    palette = condition_palette("Lymphedema", "Normal", others=["Other"])

    assert palette["Lymphedema"] == LE_RED
    assert palette["Normal"] == NORMAL_BLUE
    assert palette["Other"].startswith("#")


def test_publication_style_sets_editable_vector_fonts() -> None:
    """Publication style should preserve editable PDF/SVG text."""

    import matplotlib as mpl

    set_publication_style(dpi=250, small=True)

    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42
    assert mpl.rcParams["svg.fonttype"] == "none"
    assert mpl.rcParams["savefig.dpi"] == 250


def test_pvalue_to_stars() -> None:
    """P-value labels should match the reference figure convention."""

    assert pvalue_to_stars(0.00001) == "****"
    assert pvalue_to_stars(0.005) == "**"
    assert pvalue_to_stars(0.5) == "ns"


def test_violin_with_stats_and_categorical_embedding_render(tmp_path) -> None:
    """Core reusable figure helpers should render and save without errors."""

    import matplotlib.pyplot as plt

    set_publication_style(small=True)
    frame = pd.DataFrame(
        {
            "condition": ["Normal"] * 6 + ["Lymphedema"] * 6,
            "score": [1.0, 1.1, 1.2, 1.1, 1.3, 1.0, 2.0, 2.1, 2.2, 2.1, 2.3, 2.0],
        }
    )
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    violin_with_stats(
        ax,
        frame,
        "condition",
        "score",
        palette=condition_palette("Lymphedema", "Normal"),
        order=["Normal", "Lymphedema"],
    )
    violin_path = tmp_path / "violin.png"
    fig.savefig(violin_path)
    plt.close(fig)

    obs = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=[f"cell_{i}" for i in range(4)])
    adata = ad.AnnData(X=np.ones((4, 2)), obs=obs)
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.2], [2.0, 2.0], [2.2, 2.1]])
    embedding_fig = categorical_embedding(adata, "group", point_size=8)
    embedding_path = tmp_path / "embedding.png"
    embedding_fig.savefig(embedding_path)
    plt.close(embedding_fig)

    assert violin_path.exists()
    assert embedding_path.exists()


def test_violin_does_not_compute_inference_from_cells(monkeypatch):
    import matplotlib.pyplot as plt
    from scipy import stats

    def forbidden(*args, **kwargs):
        raise AssertionError("Renderers must not perform inference")

    monkeypatch.setattr(stats, "mannwhitneyu", forbidden)
    monkeypatch.setattr(stats, "wilcoxon", forbidden)
    data = pd.DataFrame({"condition": ["A"] * 10 + ["B"] * 10, "score": np.arange(20, dtype=float)})
    fig, ax = plt.subplots()
    violin_with_stats(ax, data, "condition", "score")
    assert not ax.texts
    plt.close(fig)


def test_publication_sizes_survive_seaborn_defaults() -> None:
    import matplotlib as mpl

    from cellquorum.visualization.figstyle import FONTSIZE

    with mpl.rc_context():
        set_publication_style(small=True)
        assert mpl.rcParams["font.size"] == 7
        assert mpl.rcParams["axes.titlesize"] == 8
        assert mpl.rcParams["axes.linewidth"] == 0.7
        set_publication_style(small=False)
        assert mpl.rcParams["axes.titlesize"] == FONTSIZE["title"]
        assert mpl.rcParams["xtick.labelsize"] == FONTSIZE["tick"]
        assert mpl.rcParams["axes.linewidth"] == 0.75


def test_render_restores_style_after_success_and_failure() -> None:
    import matplotlib as mpl

    from cellquorum.visualization.figstyle import render_figure

    with mpl.rc_context():
        mpl.rcParams["font.size"] = 17
        figures, warnings = [], []

        def successful():
            set_publication_style(small=True)

        def failed():
            set_publication_style(small=True)
            raise ValueError("test renderer failed")

        render_figure("success", successful, figures=figures, warnings=warnings)
        assert mpl.rcParams["font.size"] == 17
        render_figure("failure", failed, figures=figures, warnings=warnings)
        assert mpl.rcParams["font.size"] == 17
        assert warnings == ["failure figure failed: test renderer failed"]


def test_render_closes_owned_figures_and_preserves_existing(tmp_path) -> None:
    import matplotlib.pyplot as plt

    from cellquorum.visualization.figstyle import render_figure

    existing = plt.figure()
    baseline = set(plt.get_fignums())
    figures, warnings = [], []
    output = tmp_path / "panel.png"

    def successful():
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(output)
        return output

    def failed():
        plt.subplots()
        raise RuntimeError("drawing failed")

    try:
        render_figure("panel", successful, figures=figures, warnings=warnings)
        assert output.is_file()
        assert figures == [output]
        assert set(plt.get_fignums()) == baseline
        for _ in range(25):
            render_figure("panel", failed, figures=figures, warnings=warnings)
        assert set(plt.get_fignums()) == baseline
        assert len(warnings) == 25
        assert figures == [output]
    finally:
        plt.close(existing)


def test_embedding_order_cannot_hide_a_rare_population() -> None:
    import pytest

    data = ad.AnnData(
        np.ones((4, 1)),
        obs=pd.DataFrame({"population": ["Common"] * 3 + ["Rare"]}, index=list("abcd")),
    )
    data.obsm["X_umap"] = np.array([[0, 0], [0, 1], [1, 0], [5, 5]])
    with pytest.raises(ValueError, match="omits observed populations.*Rare"):
        categorical_embedding(data, "population", order=["Common"])
    with pytest.raises(ValueError, match="duplicates"):
        categorical_embedding(data, "population", order=["Common", "Common", "Rare"])
    with pytest.raises(ValueError, match="at least one color"):
        categorical_embedding(data, "population", palette=[])
