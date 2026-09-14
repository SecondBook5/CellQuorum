"""Two-column categorical-flow Sankey diagram.

General-purpose: any crosstab of one categorical call against another (an
automated classifier vs. a final consensus, a cluster vs. a cell type, a
before/after relabeling) draws the same way. Ribbons are cubic-bezier bands
between stacked node blocks, colored by the RIGHT-side category so agreement
reads as "the ribbon matches its destination node's color."
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from cellquorum.stages.integration.embeddings.plots import apply_theme
from cellquorum.visualization.figstyle import cell_type_palette, save_figure

if TYPE_CHECKING:
    import pandas as pd

_GREY = "#9aa0a6"


def _stack(
    order: list[str], totals: pd.Series, gap: float
) -> tuple[dict[str, tuple[float, float]], float]:
    """Return {category: (top, bottom)} stacked downward from a centered top."""
    height_used = sum(totals[k] for k in order) + gap * max(len(order) - 1, 0)
    top = height_used / 2.0
    positions: dict[str, tuple[float, float]] = {}
    y = top
    for k in order:
        h = totals[k]
        positions[k] = (y, y - h)
        y -= h + gap
    return positions, height_used


def two_column_sankey(
    crosstab: pd.DataFrame,
    *,
    left_title: str,
    right_title: str,
    out_path: Path,
    left_colors: dict[str, str] | None = None,
    right_colors: dict[str, str] | None = None,
    left_order: list[str] | None = None,
    right_order: list[str] | None = None,
    subtitle: str | None = None,
) -> Path:
    """
    Plot a two-column Sankey: crosstab.index (left) flowing to crosstab.columns (right).

    Args:
        crosstab: Rows = left-side categories, columns = right-side categories,
            values = flow counts (e.g. ``pd.crosstab(left, right)``).
        left_title: Label above the left column.
        right_title: Label above the right column.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
        left_colors: Category -> color for left nodes/labels. Falls back to
            ``right_colors`` (a category present on both sides keeps one
            color), then a generated categorical palette.
        right_colors: Category -> color for right nodes/ribbons (ribbons are
            always colored by their RIGHT-side destination).
        left_order: Top-to-bottom order for left categories. None sorts by
            each category's dominant right-side destination, then by size.
        right_order: Top-to-bottom order for right categories. None keeps
            ``crosstab.columns``'s own order.
        subtitle: Optional caption text below the diagram.

    Returns:
        The PNG path written.

    Raises:
        ValueError: If every cell of ``crosstab`` is zero (no flow to draw).
    """

    total = float(crosstab.to_numpy().sum())
    if total <= 0:
        raise ValueError("crosstab has no flow (every cell is zero) -- nothing to draw.")

    apply_theme()
    out_path = Path(out_path)

    right_order = list(right_order) if right_order is not None else list(crosstab.columns)
    right_colors = right_colors or cell_type_palette(right_order)
    left_colors = left_colors or right_colors

    if left_order is None:
        right_pos = {name: i for i, name in enumerate(right_order)}

        def _dominant_position(row_name: str) -> tuple[int, float]:
            row = crosstab.loc[row_name]
            dominant = row.idxmax()
            return right_pos.get(dominant, len(right_order)), -float(row.sum())

        left_order = sorted(crosstab.index, key=_dominant_position)
    else:
        left_order = list(left_order)

    left_totals = crosstab.sum(axis=1)
    right_totals = crosstab.sum(axis=0)

    gap = 0.018 * total
    left_pos, left_height = _stack(left_order, left_totals, gap)
    right_pos, right_height = _stack(right_order, right_totals, gap)

    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    x_left, x_right = 0.0, 1.0
    node_width = 0.028

    for name in left_order:
        top, bottom = left_pos[name]
        ax.add_patch(
            Rectangle(
                (x_left - node_width, bottom),
                node_width,
                top - bottom,
                color=left_colors.get(name, _GREY),
                lw=0,
                zorder=4,
                clip_on=False,
            )
        )
    for name in right_order:
        top, bottom = right_pos[name]
        ax.add_patch(
            Rectangle(
                (x_right, bottom),
                node_width,
                top - bottom,
                color=right_colors.get(name, _GREY),
                lw=0,
                zorder=4,
                clip_on=False,
            )
        )

    left_running_top = {name: left_pos[name][0] for name in left_order}
    right_running_top = {name: right_pos[name][0] for name in right_order}
    x_mid = (x_left + x_right) / 2
    for left_name in left_order:
        for right_name in right_order:
            count = crosstab.loc[left_name, right_name] if right_name in crosstab.columns else 0
            if count <= 0:
                continue
            y0_top = left_running_top[left_name]
            y0_bottom = y0_top - count
            left_running_top[left_name] = y0_bottom
            y1_top = right_running_top[right_name]
            y1_bottom = y1_top - count
            right_running_top[right_name] = y1_bottom

            color = right_colors.get(right_name, _GREY)
            agrees = left_name == right_name
            ax.add_patch(
                PathPatch(
                    MplPath(
                        [
                            (x_left, y0_top),
                            (x_mid, y0_top),
                            (x_mid, y1_top),
                            (x_right, y1_top),
                            (x_right, y1_bottom),
                            (x_mid, y1_bottom),
                            (x_mid, y0_bottom),
                            (x_left, y0_bottom),
                            (x_left, y0_top),
                        ],
                        [
                            MplPath.MOVETO,
                            MplPath.CURVE4,
                            MplPath.CURVE4,
                            MplPath.CURVE4,
                            MplPath.LINETO,
                            MplPath.CURVE4,
                            MplPath.CURVE4,
                            MplPath.CURVE4,
                            MplPath.CLOSEPOLY,
                        ],
                    ),
                    facecolor=color,
                    edgecolor="none",
                    alpha=(0.62 if agrees else 0.34),
                    zorder=2,
                )
            )

    for name in left_order:
        top, bottom = left_pos[name]
        pct = 100 * left_totals[name] / total
        ax.text(
            x_left - node_width - 0.012,
            (top + bottom) / 2,
            f"{name}  ({int(left_totals[name]):,} · {pct:.1f}%)",
            ha="right",
            va="center",
            fontsize=8.2,
            color=left_colors.get(name, "#5a5f66"),
            fontweight="bold",
        )
    for name in right_order:
        top, bottom = right_pos[name]
        pct = 100 * right_totals[name] / total
        ax.text(
            x_right + node_width + 0.012,
            (top + bottom) / 2,
            f"{name}  ({int(right_totals[name]):,} · {pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=8.2,
            color=right_colors.get(name, "#333"),
            fontweight="bold",
        )

    max_height = max(left_height, right_height)
    ax.text(
        x_left - node_width,
        max_height / 2 + 0.03 * total,
        left_title,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#333",
    )
    ax.text(
        x_right + node_width,
        max_height / 2 + 0.03 * total,
        right_title,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#333",
    )

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-max_height / 2 - 0.04 * total, max_height / 2 + 0.10 * total)
    ax.axis("off")
    fig.suptitle(f"{left_title} vs. {right_title}", fontsize=13, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(
            0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=7.5, color="#6b7075", wrap=True
        )

    paths = save_figure(fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


__all__ = ["two_column_sankey"]
