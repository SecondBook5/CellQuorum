"""The enrichment map: pathways as nodes, joined when their leading edges share genes."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.enrichment import enrichment_map  # noqa: E402


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "score": [1.8, 1.5, -1.6, 0.9],
            "padj": [0.001, 0.03, 0.02, 0.5],
            "leading_edge": ["g1;g2;g3;g4", "g3;g4;g5;g6", "x1;x2", ""],
        }
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _edges(fig):
    return {
        ln.get_gid()
        for ln in fig.axes[0].lines + fig.axes[0].collections
        if (ln.get_gid() or "").startswith("edge|")
    }


def _nodes(fig):
    return {
        p.get_gid(): p for p in fig.axes[0].collections if (p.get_gid() or "").startswith("node|")
    }


def test_one_node_per_pathway() -> None:
    assert set(_nodes(enrichment_map(_table()))) == {"node|A", "node|B", "node|C", "node|D"}


def test_pathways_are_joined_only_when_their_leading_edges_overlap_enough() -> None:
    assert _edges(enrichment_map(_table(), min_jaccard=0.2)) == {"edge|A|B"}
    assert _edges(enrichment_map(_table(), min_jaccard=0.5)) == set()


def test_only_significant_pathways_are_labelled_and_outlined() -> None:
    fig = enrichment_map(_table(), alpha=0.05)
    labels = {t.get_text() for t in fig.axes[0].texts if t.get_gid() == "node_label"}
    assert labels == {"A", "B", "C"}
    nodes = _nodes(fig)
    assert nodes["node|A"].get_linewidths()[0] > nodes["node|D"].get_linewidths()[0]


def test_the_colour_follows_the_sign_of_the_score() -> None:
    fig = enrichment_map(_table())
    fig.canvas.draw()  # the colormap is applied at draw time
    nodes = _nodes(fig)
    up = nodes["node|A"].get_facecolor()[0]
    down = nodes["node|C"].get_facecolor()[0]
    assert up[0] > up[2]
    assert down[2] > down[0]


def test_the_layout_is_the_same_for_the_same_seed() -> None:
    first = _nodes(enrichment_map(_table(), seed=3))["node|A"].get_offsets()
    again = _nodes(enrichment_map(_table(), seed=3))["node|A"].get_offsets()
    assert np.allclose(first, again)


def test_an_empty_leading_edge_is_a_node_without_edges() -> None:
    fig = enrichment_map(_table(), min_jaccard=0.0)
    assert not any("D" in e.split("|")[1:] for e in _edges(fig))
