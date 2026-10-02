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


def test_large_graph_skips_individual_node_labels() -> None:
    """At real atlas scale (thousands of metacells), a text label on every
    node is a wall of illegible overlapping text -- confirmed live against a
    4297-metacell atlas_v3 run. Above max_labeled_nodes, rely on color +
    legend only, matching how the reference atlas figure itself has no
    per-point text labels in its network panel."""
    n = 200
    nodes = pd.DataFrame(
        {
            "metacell_id": [f"MC{i}" for i in range(n)],
            "dominant_group": ["A" if i % 2 else "B" for i in range(n)],
            "n_cells": [10] * n,
        }
    )
    edges = pd.DataFrame(
        {
            "source": [f"MC{i}" for i in range(n - 1)],
            "target": [f"MC{i + 1}" for i in range(n - 1)],
            "weight": [0.5] * (n - 1),
        }
    )
    fig = _build_network_figure(edges, nodes)
    ax = fig.axes[0]
    assert len(ax.texts) == 0
