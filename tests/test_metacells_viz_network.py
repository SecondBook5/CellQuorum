"""Tests for the metacell similarity-network figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.metacells.viz.network import _build_network_figure, plot_metacell_network


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metacell_id": ["MC1", "MC2", "MC3"],
            "dominant_group": ["Fibroblast", "Endothelial", "Fibroblast"],
            "n_cells": [80, 95, 60],
        }
    )


def _edges() -> pd.DataFrame:
    return pd.DataFrame({"source": ["MC1", "MC2"], "target": ["MC2", "MC3"], "weight": [0.8, 0.5]})


def test_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = plot_metacell_network(_edges(), _nodes(), tmp_path)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0


def test_empty_edges_returns_empty(tmp_path: Path) -> None:
    assert (
        plot_metacell_network(
            pd.DataFrame(columns=["source", "target", "weight"]), _nodes(), tmp_path
        )
        == []
    )


def test_every_node_gets_a_label() -> None:
    fig = _build_network_figure(_edges(), _nodes())
    ax = fig.axes[0]
    labels = {t.get_text() for t in ax.texts}
    assert labels == {"MC1", "MC2", "MC3"}
