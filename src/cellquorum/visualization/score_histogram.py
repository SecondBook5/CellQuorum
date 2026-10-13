"""A score or index drawn as overlaid histograms, one per condition.

Both conditions are binned on the same edges and drawn translucent on one axis, so where they
overlap the colours mix and where one condition sits alone its own colour shows. The legend gives
the number of cells behind each histogram, because two conditions rarely have the same number and
raw counts then favour the larger one; ``stat="share"`` scales each to its own total instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellquorum.visualization.figstyle import LE_RED, NORMAL_COLOR, apply_theme

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

_STATS = ("count", "share")


def score_histogram(
    value: pd.Series,
    condition: pd.Series,
    *,
    case: str,
    control: str,
    value_label: str,
    title: str = "",
    bins: int = 25,
    stat: str = "count",
    colors: dict[str, str] | None = None,
) -> Figure:
    """One value for one set of cells, a histogram per condition on shared bins.

    Args:
        value: The value of each cell (an index, a score).
        condition: Condition label of each cell; only ``control`` and ``case`` are drawn.
        case, control: The two condition levels, drawn control first.
        value_label: Name of the value, the x axis label.
        title: Optional title.
        bins: Number of equal-width bins across the observed range.
        stat: ``"count"`` (cells per bin) or ``"share"`` (share of the condition's cells per bin).
        colors: Optional colour per condition level.

    Raises:
        ValueError: If ``stat`` is not recognised, or a condition has no finite values (named).
    """
    import matplotlib.pyplot as plt

    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}, got {stat!r}")
    apply_theme("compact")
    palette = {control: NORMAL_COLOR, case: LE_RED, **(colors or {})}
    x_all = np.asarray(value, dtype=float)
    cond = np.asarray(condition).astype(str)
    finite = np.isfinite(x_all)
    for level in (control, case):
        if not (finite & (cond == level)).any():
            raise ValueError(f"{level!r} has no cells with a value to draw")
    edges = np.linspace(float(x_all[finite].min()), float(x_all[finite].max()), bins + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    legend = []
    for level in (control, case):
        x = x_all[finite & (cond == level)]
        counts, _ = np.histogram(x, bins=edges)
        heights = counts / counts.sum() if stat == "share" else counts
        for left, width, height in zip(edges[:-1], np.diff(edges), heights, strict=True):
            bar = ax.bar(
                left,
                height,
                width=width,
                align="edge",
                color=palette[level],
                alpha=0.55,
                linewidth=0,
            )
            bar.patches[0].set_gid(f"bars|{level}")
        legend.append((level, len(x)))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[level], alpha=0.55, linewidth=0)
        for level, _ in legend
    ]
    ax.legend(handles, [f"{level} (n = {n})" for level, n in legend], frameon=False, fontsize=8)
    ax.set_xlabel(value_label)
    ax.set_ylabel("Cells" if stat == "count" else "Share of cells")
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


__all__ = ["score_histogram"]
