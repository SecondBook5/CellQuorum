"""A score along an ordering of cells, one smoothed curve per condition.

For a programme or a regulon whose activity is expected to change as cells move along an axis
(an EndoMT index, a pseudotime), this draws every cell as a faint point and, per condition, a
locally weighted (LOWESS) curve with a band of resampling uncertainty: the curve of each condition
is recomputed on cells resampled within that condition, and the band is the 2.5th to 97.5th
percentile of those curves at each position. Two conditions that follow the same curve are one
trajectory; two curves that part show where the conditions differ along the axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellquorum.visualization.figstyle import LE_RED, NORMAL_COLOR, apply_theme

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def _lowess(x: np.ndarray, y: np.ndarray, grid: np.ndarray, frac: float) -> np.ndarray:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fit = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = fit[:, 0], fit[:, 1]
    keep = np.concatenate([[True], np.diff(xs) > 0])
    return np.interp(grid, xs[keep], ys[keep])


def score_along_axis(
    axis: pd.Series,
    score: pd.Series,
    condition: pd.Series,
    *,
    case: str,
    control: str,
    axis_label: str,
    score_label: str,
    title: str = "",
    frac: float = 0.5,
    n_boot: int = 100,
    n_grid: int = 60,
    min_cells: int = 20,
    seed: int = 0,
    colors: dict[str, str] | None = None,
) -> Figure:
    """One score against one axis for one set of cells, a smoothed curve per condition.

    A condition's curve and band span only the positions its own cells cover.

    Args:
        axis: Position of each cell along the ordering (any scale; drawn as given).
        score: Value of the score in each cell (for example z-normalised within the cells shown).
        condition: Condition label of each cell; only ``control`` and ``case`` are drawn.
        case, control: The two condition levels, drawn control first.
        axis_label, score_label: Axis names.
        title: Optional title.
        frac: LOWESS span (the share of cells each local fit uses).
        n_boot: Resamples behind the uncertainty band.
        n_grid: Positions at which the curves and bands are evaluated.
        min_cells: A condition with fewer cells than this cannot be smoothed.
        seed: Seed of the resampling.
        colors: Optional colour per condition level.

    Raises:
        ValueError: If a condition has fewer than ``min_cells`` cells, naming it.
    """
    import matplotlib.pyplot as plt

    apply_theme("compact")
    palette = {control: NORMAL_COLOR, case: LE_RED, **(colors or {})}
    x_all = np.asarray(axis, dtype=float)
    y_all = np.asarray(score, dtype=float)
    cond = np.asarray(condition).astype(str)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    for level in (control, case):
        n = int((finite & (cond == level)).sum())
        if n < min_cells:
            raise ValueError(
                f"{level!r} has {n} cells, fewer than the {min_cells} needed to smooth"
            )

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cells = ax.scatter(
        x_all[finite],
        y_all[finite],
        s=5,
        alpha=0.18,
        c=[palette[c] for c in cond[finite]],
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    cells.set_gid("cells")
    for level in (control, case):
        m = finite & (cond == level)
        x, y = x_all[m], y_all[m]
        # Each condition is drawn only over the span its own cells cover (2.5th to 97.5th
        # percentile of position); beyond it the smoother would extrapolate a flat line.
        lo, hi = np.percentile(x, [2.5, 97.5])
        grid = np.linspace(float(lo), float(hi), n_grid)
        curve = _lowess(x, y, grid, frac)
        draws = np.empty((n_boot, n_grid))
        for b in range(n_boot):
            pick = rng.integers(0, len(x), len(x))
            draws[b] = _lowess(x[pick], y[pick], grid, frac)
        low, high = np.percentile(draws, [2.5, 97.5], axis=0)
        band = ax.fill_between(
            grid, low, high, color=palette[level], alpha=0.18, linewidth=0, zorder=2
        )
        band.set_gid(f"band|{level}")
        (line,) = ax.plot(grid, curve, color=palette[level], lw=2.0, zorder=3, label=level)
        line.set_gid(f"curve|{level}")
    ax.set_xlabel(axis_label)
    ax.set_ylabel(score_label)
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


__all__ = ["score_along_axis"]
