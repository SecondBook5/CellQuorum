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
