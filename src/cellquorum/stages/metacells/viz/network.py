"""Metacell similarity-network figure: force-directed layout, nodes colored by
dominant cell type, matching the house convention established in
`cell_cell_communication.viz._plots.curvature_network` (curved edges would be
overkill here -- this graph is typically far denser/more connected than a
regulon graph, so a plain spring layout with the house palette is the right
level of polish).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cellquorum.visualization import figstyle


def _build_network_figure(
    edges: pd.DataFrame, nodes: pd.DataFrame, *, max_labeled_nodes: int = 60
) -> Figure:
    """Draw the metacell network figure and return it unsaved (for geometry tests).

    Above `max_labeled_nodes`, individual node text labels are dropped
    entirely -- confirmed live against a real 4297-metacell atlas run that a
    label on every node is a wall of illegible overlapping text, not a
    readable figure. Color + the dominant-cell-type legend still identify
    every node's group at any scale; this matches the reference atlas
    figure's own convention of no per-point text labels in its network panel.
    """

    groups = sorted(nodes["dominant_group"].astype(str).unique())
    palette = (
        figstyle.MUTED_PALETTE
        if len(groups) <= len(figstyle.MUTED_PALETTE)
        else figstyle.distinct_palette(len(groups))
    )
    color_by_group = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(str(row["metacell_id"]))
    for _, row in edges.iterrows():
        G.add_edge(str(row["source"]), str(row["target"]), weight=float(row["weight"]))

    pos = nx.spring_layout(G, seed=0, weight="weight")

    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE["heatmap"])
    for u, v, d in G.edges(data=True):
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color="#999999",
            linewidth=0.5 + 1.5 * d["weight"],
            alpha=0.5,
            zorder=1,
        )

    node_group = dict(
        zip(nodes["metacell_id"].astype(str), nodes["dominant_group"].astype(str), strict=False)
    )
    node_size = dict(zip(nodes["metacell_id"].astype(str), nodes["n_cells"], strict=False))
    max_size = max(node_size.values()) if node_size else 1

    label_nodes = len(pos) <= max_labeled_nodes
    for n, (x, y) in pos.items():
        color = color_by_group.get(node_group.get(n, ""), "#888888")
        size = 60.0 + 200.0 * (node_size.get(n, 1) / max_size) if max_size else 100.0
        ax.scatter([x], [y], s=size, color=color, edgecolor="#222", linewidth=0.6, zorder=2)
        if label_nodes:
            ax.text(x, y, n, fontsize=6.5, ha="center", va="center", zorder=3, color=figstyle.TEXT)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_by_group[g],
            markeredgecolor="#222",
            markersize=8,
            label=g,
        )
        for g in groups
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=7.5,
        title="Dominant cell type",
        frameon=True,
    )

    ax.set_title("Metacell similarity network", fontsize=13, fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_metacell_network(
    edges: pd.DataFrame, nodes: pd.DataFrame, out_dir: str | Path, *, tag: str = ""
) -> list[Path]:
    """Render the metacell similarity-network figure.

    Args:
        edges: Columns source, target, weight (metacell-to-metacell similarity).
        nodes: Columns metacell_id, dominant_group, n_cells.
        out_dir: Output directory.
        tag: Optional filename tag.

    Returns:
        Written PNG+PDF paths, or [] when there are no edges to draw.
    """
    figstyle.apply_theme("compact")
    if edges is None or len(edges) == 0:
        return []
    out_dir = Path(out_dir)
    fig = _build_network_figure(edges, nodes)
    stem = f"metacell_network_{tag}" if tag else "metacell_network"
    return figstyle.save_figure(fig, out_dir, stem)


__all__ = ["plot_metacell_network"]
