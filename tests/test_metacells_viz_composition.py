"""Tests for the metacell composition-bar figure, and its column-order lock-step
with the gene heatmap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.metacells.viz.composition import plot_metacell_composition_bar
from cellquorum.stages.metacells.viz.heatmap import ordered_metacell_columns


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metacell_id": ["MC1", "MC2", "MC3"],
            "dominant_group": ["Fibroblast", "Endothelial", "Fibroblast"],
            "n_cells": [80, 95, 60],
        }
    )


def _assignments() -> pd.DataFrame:
    rows = []
    for mc, n, dom in [
        ("MC1", 80, "Fibroblast"),
        ("MC2", 95, "Endothelial"),
        ("MC3", 60, "Fibroblast"),
    ]:
        for i in range(n):
            group = dom if i < n - 5 else "Other"
            rows.append({"cell_id": f"{mc}_{i}", "metacell_id": mc, "group": group})
    return pd.DataFrame(rows)


def test_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = plot_metacell_composition_bar(_assignments(), _nodes(), tmp_path)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]


def test_column_order_matches_heatmap_ordering() -> None:
    from cellquorum.stages.metacells.viz.composition import _build_composition_figure

    fig = _build_composition_figure(_assignments(), _nodes())
    ax = fig.axes[0]
    tick_order = [lbl.get_text() for lbl in ax.get_xticklabels()]
    assert tick_order == ordered_metacell_columns(_nodes())
