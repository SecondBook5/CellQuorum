"""Subclustering diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from cellquorum.visualization.style import (
    CELLQUORUM_BLUE,
    CELLQUORUM_RED,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    save_cellquorum_figure,
)

# Use Agg backend for headless plotting.
matplotlib.use("Agg")


def plot_group_recovery(
    counts: dict[str, int],
    min_cells: int,
    group_key: str,
    out_path: Path | str,
) -> Path:
    """
    Plot group-level cell counts with threshold line.

    This shows how many cells per group survived the group_filter,
    with kept/dropped groups colored by threshold.

    Args:
        counts: {group: cell_count} dict.
        min_cells: minimum cells threshold.
        group_key: obs column name for groups.
        out_path: output file path.

    Returns:
        Path to saved figure.
    """
    # Guard: skip if counts empty.
    if not counts:
        return Path(out_path)

    # Apply CellQuorum theme.
    apply_cellquorum_theme()

    # Sort groups by count (descending).
    sorted_groups = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    groups = [g for g, _ in sorted_groups]
    values = [c for _, c in sorted_groups]

    # Assign colors (kept = blue, dropped = red).
    colors = [CELLQUORUM_BLUE if v >= min_cells else CELLQUORUM_RED for v in values]

    # Create figure.
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars.
    x_pos = range(len(groups))
    ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor="none")

    # Plot threshold line.
    ax.axhline(
        min_cells,
        color=CELLQUORUM_RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {min_cells}",
    )

    # Axis labels.
    ax.set_xlabel(f"{group_key}", fontsize=11)
    ax.set_ylabel("Cell count", fontsize=11)
    ax.set_title(
        f"Group recovery: {group_key}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, rotation=45, ha="right")

    # Legend.
    ax.legend(loc="upper right", frameon=False)

    # Apply CellQuorum axis style.
    apply_cellquorum_axis_style(ax)

    # Save figure.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cellquorum_figure(fig, out_path)
    plt.close(fig)

    return out_path


__all__ = ["plot_group_recovery"]
