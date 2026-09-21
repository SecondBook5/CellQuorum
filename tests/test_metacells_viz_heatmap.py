"""Tests for the metacell gene x metacell heatmap figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.metacells.viz.heatmap import (
    ordered_metacell_columns,
    plot_metacell_gene_heatmap,
)


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metacell_id": ["MC1", "MC2", "MC3"],
            "dominant_group": ["Fibroblast", "Endothelial", "Fibroblast"],
            "n_cells": [80, 95, 60],
        }
    )


def _expr() -> pd.DataFrame:
    return pd.DataFrame(
        {"MC1": [1.0, 2.0], "MC2": [0.1, 0.2], "MC3": [1.5, 1.8]},
        index=["GENE_A", "GENE_B"],
    )


def test_ordered_columns_group_by_dominant_type() -> None:
    order = ordered_metacell_columns(_nodes())
    assert order.index("MC1") < order.index("MC2") or order.index("MC3") < order.index("MC2")
    # Both Fibroblast metacells sit together, Endothelial elsewhere.
    fibro_positions = [order.index("MC1"), order.index("MC3")]
    endo_position = order.index("MC2")
    assert (max(fibro_positions) < endo_position) or (min(fibro_positions) > endo_position)


def test_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = plot_metacell_gene_heatmap(_expr(), _nodes(), tmp_path)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]


def test_empty_expr_returns_empty(tmp_path: Path) -> None:
    assert plot_metacell_gene_heatmap(pd.DataFrame(), _nodes(), tmp_path) == []
