"""Composition bar: % of cells by dominant group per metacell, columns in the
SAME order as the gene heatmap (heatmap.ordered_metacell_columns is the single
shared source of that order)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.metacells.viz.heatmap import ordered_metacell_columns
from cellquorum.visualization import figstyle


def _build_composition_figure(
    assignments: pd.DataFrame, nodes: pd.DataFrame, *, max_labeled_columns: int = 80
) -> Figure:
    """Draw the composition-bar figure and return it unsaved (for geometry tests).

    Above `max_labeled_columns`, x-axis tick labels are dropped entirely --
    confirmed live against a real 4297-metacell atlas run that a label on
    every column is a solid, illegible band of text, not a readable axis.
    """

    order = ordered_metacell_columns(nodes)
    groups = sorted(assignments["group"].astype(str).unique())
    palette = (
        figstyle.MUTED_PALETTE
        if len(groups) <= len(figstyle.MUTED_PALETTE)
        else figstyle.distinct_palette(len(groups))
    )
    color_by_group = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    pct = assignments.groupby(["metacell_id", "group"], observed=True).size().unstack(fill_value=0)
    pct = pct.reindex(order).fillna(0)
    pct = pct.div(pct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE["heatmap"])
    bottom = pd.Series(0.0, index=pct.index)
    for g in groups:
        if g not in pct.columns:
            continue
        ax.bar(pct.index, pct[g], bottom=bottom, color=color_by_group[g], width=0.9, label=g)
        bottom = bottom + pct[g]

    if len(order) <= max_labeled_columns:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    ax.set_ylabel("% of cells", fontsize=9)
    ax.set_title("Metacell composition", fontsize=13, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=7.5,
        title="Group",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def plot_metacell_composition_bar(
    assignments: pd.DataFrame, nodes: pd.DataFrame, out_dir: str | Path, *, tag: str = ""
) -> list[Path]:
    """Render the per-metacell composition bar chart.

    Args:
        assignments: Per-cell rows with metacell_id and group columns.
        nodes: Columns metacell_id, dominant_group (defines the shared column order).
        out_dir: Output directory.
        tag: Optional filename tag.

    Returns:
        Written PNG+PDF paths, or [] when assignments is empty.
    """
    figstyle.set_style()
    if assignments is None or assignments.empty:
        return []
    out_dir = Path(out_dir)
    fig = _build_composition_figure(assignments, nodes)
    stem = f"metacell_composition_bar_{tag}" if tag else "metacell_composition_bar"
    return figstyle.save_figure(fig, out_dir, stem)


__all__ = ["plot_metacell_composition_bar"]
