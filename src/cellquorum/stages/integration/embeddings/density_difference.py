"""Case-minus-control density on an embedding, the numbers and the drawing.

Where do the cells of one condition sit that the cells of the other do not? Each specimen gets
its own Gaussian kernel density on one shared grid, and that density is normalised to sum to 1,
so a specimen with 40,000 cells and one with 4,000 cells count the same. The figure is the mean
case-specimen density minus the mean control-specimen density: red where the case specimens put
more of their cells, blue where the controls do, blank where they agree.

Weighting by specimen rather than by cell is the point. Pooling cells lets whichever arm was
sequenced deeper, or simply yielded more cells, paint the picture, and a single large specimen
can pass for a condition effect. The figure is descriptive. With a handful of specimens per arm
it shows where each arm's cells sit; it is not a test, and the figure says so and states how
many specimens stand behind each colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from cellquorum.stages.integration.embeddings.plots import (
    _adaptive_point_size,
    _lighten_colorbar_outline,
    _style_axes,
    view_limits,
)
from cellquorum.visualization.figstyle import TEXT

# The window of the embedding the density grid covers when no view_percentile is configured.
# A grid spanning the full range spends its resolution on a few far-away cells.
_DEFAULT_WINDOW = (0.3, 99.7)


class DensityDifferenceError(ValueError):
    """The density difference cannot be formed from the cells and specimens given."""


@dataclass(frozen=True)
class DensityDifference:
    """The case-minus-control density on a grid, with what stands behind it.

    Attributes:
        x_edges: Grid x range, ``(low, high)``.
        y_edges: Grid y range, ``(low, high)``.
        difference: ``(grid, grid)`` array, case minus control, in units of the fraction of a
            specimen's cells per grid cell. Positive means case-enriched.
        limit: Symmetric colour limit, a high percentile of the absolute difference.
        control_specimens: Specimens that contributed to the control mean.
        case_specimens: Specimens that contributed to the case mean.
        dropped: ``(specimen, n_cells)`` for specimens too small to estimate a density from.
    """

    x_edges: tuple[float, float]
    y_edges: tuple[float, float]
    difference: np.ndarray
    limit: float
    control_specimens: tuple[str, ...]
    case_specimens: tuple[str, ...]
    dropped: tuple[tuple[str, int], ...]


def _specimen_arms(
    specimen: np.ndarray, condition: np.ndarray, case: str, control: str
) -> dict[str, str]:
    """Map each specimen to ``case`` or ``control``; a specimen in both is an error.

    A specimen is one biological replicate and belongs to one arm. If a "specimen" column holds
    cells of both conditions it is not a specimen column (a batch or a lane, say), and averaging
    over it would quietly turn the figure into something else, so this refuses.
    """
    frame = pd.DataFrame({"specimen": specimen, "condition": condition})
    frame = frame[frame["condition"].isin([case, control])]
    levels = frame.groupby("specimen")["condition"].nunique()
    mixed = sorted(levels.index[levels > 1].astype(str))
    if mixed:
        raise DensityDifferenceError(
            f"specimen column mixes '{case}' and '{control}' cells within {mixed[:5]}; "
            "it must identify one biological replicate, each in a single condition"
        )
    return {str(s): str(c) for s, c in frame.drop_duplicates("specimen").to_numpy()}


def _grid(
    xy: np.ndarray, view_percentile: tuple[float, float] | None, size: int
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """A ``size`` x ``size`` evaluation grid over the plotted window; returns points and ranges."""
    limits = view_limits(xy, view_percentile or _DEFAULT_WINDOW)
    assert limits is not None  # a window is always passed
    (x0, x1), (y0, y1), _ = limits
    gx, gy = np.meshgrid(np.linspace(x0, x1, size), np.linspace(y0, y1, size))
    return np.vstack([gx.ravel(), gy.ravel()]), (x0, x1), (y0, y1)


def _specimen_density(
    points: np.ndarray, grid: np.ndarray, size: int, bandwidth: float, max_cells: int, seed: int
) -> np.ndarray:
    """One specimen's density on the grid, normalised to sum to 1.

    The kernel width is ``bandwidth`` times the specimen's own spread, so it does not depend on
    how many cells the specimen has and a subsample keeps the same shape. Specimens larger than
    ``max_cells`` are subsampled, with a fixed seed, only for the density; evaluating a kernel
    sum over tens of thousands of cells at every grid point is the cost, and the density of a
    few thousand cells already has the same smooth shape at this bandwidth.
    """
    if len(points) > max_cells:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_cells, replace=False)]
    z = gaussian_kde(points.T, bw_method=bandwidth)(grid).reshape(size, size)
    return z / z.sum()


def density_difference(
    xy: np.ndarray,
    specimen: np.ndarray,
    condition: np.ndarray,
    *,
    case: str,
    control: str,
    view_percentile: tuple[float, float] | None = None,
    grid_size: int = 220,
    bandwidth: float = 0.08,
    min_cells_per_specimen: int = 50,
    max_cells_per_specimen: int = 8000,
    limit_percentile: float = 99.5,
    seed: int = 0,
) -> DensityDifference:
    """Case-minus-control density of cells on a 2-D embedding, every specimen weighted equally.

    Args:
        xy: ``(n_cells, 2)`` embedding coordinates.
        specimen: Per-cell specimen identifier.
        condition: Per-cell condition label.
        case: Condition drawn red.
        control: Condition drawn blue.
        view_percentile: Window of the embedding the grid covers; ``None`` uses 0.3 to 99.7.
        grid_size: Grid points per side.
        bandwidth: Gaussian kernel width as a fraction of each specimen's spread.
        min_cells_per_specimen: A specimen with fewer cells has no stable density and is left
            out, and reported in ``dropped``.
        max_cells_per_specimen: Cap on cells used to estimate one specimen's density.
        limit_percentile: Percentile of the absolute difference that sets the colour limit.
        seed: Seed for the subsampling.

    Raises:
        DensityDifferenceError: A specimen mixes conditions, or an arm has no usable specimen.
    """
    arms = _specimen_arms(specimen, condition, case, control)
    grid, x_edges, y_edges = _grid(xy, view_percentile, grid_size)
    densities: dict[str, list[np.ndarray]] = {case: [], control: []}
    members: dict[str, list[str]] = {case: [], control: []}
    dropped: list[tuple[str, int]] = []
    for name in sorted(arms):
        points = xy[specimen == name]
        if len(points) < min_cells_per_specimen:
            dropped.append((name, len(points)))
            continue
        densities[arms[name]].append(
            _specimen_density(points, grid, grid_size, bandwidth, max_cells_per_specimen, seed)
        )
        members[arms[name]].append(name)
    for arm in (control, case):
        if not densities[arm]:
            raise DensityDifferenceError(
                f"no '{arm}' specimen has at least {min_cells_per_specimen} cells"
            )
    difference = np.mean(densities[case], axis=0) - np.mean(densities[control], axis=0)
    limit = float(np.percentile(np.abs(difference), limit_percentile))
    return DensityDifference(
        x_edges=x_edges,
        y_edges=y_edges,
        difference=difference,
        limit=limit,
        control_specimens=tuple(members[control]),
        case_specimens=tuple(members[case]),
        dropped=tuple(dropped),
    )


def _label_groups(
    ax: Axes, xy: np.ndarray, groups: pd.Series, min_label_frac: float, names: dict[str, str]
) -> None:
    """Name each group at its median position, skipping groups too small to name honestly."""
    codes = groups.astype(str).to_numpy()
    for group in sorted(set(codes)):
        mask = codes == group
        if mask.mean() < min_label_frac:
            continue
        x, y = np.median(xy[mask], axis=0)
        ax.text(
            x,
            y,
            names.get(group, group),
            fontsize=6.5,
            ha="center",
            va="center",
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )


def plot_density_difference(
    xy: np.ndarray,
    result: DensityDifference,
    *,
    title: str,
    axis_labels: tuple[str, str],
    case_label: str,
    control_label: str,
    mask_fraction: float = 0.08,
    cmap: str = "RdBu_r",
    labels: pd.Series | None = None,
    label_names: dict[str, str] | None = None,
    min_label_frac: float = 0.001,
    view_percentile: tuple[float, float] | None = None,
) -> Figure:
    """Draw every cell in grey, then the case-minus-control density over it.

    Differences smaller than ``mask_fraction`` of the colour limit are left blank rather than
    tinted, so near-equal regions read as "no difference" instead of as a faint colour.
    """
    import matplotlib.pyplot as plt

    (x0, x1), (y0, y1) = result.x_edges, result.y_edges
    # The embedding is drawn with equal axes, so a canvas of the wrong shape leaves the cloud
    # small in the middle of empty page. Size the page from the cloud's own aspect ratio.
    height = float(np.clip(8.2 * (y1 - y0) / (x1 - x0) + 1.2, 5.0, 10.0))
    fig, ax = plt.subplots(figsize=(10.0, height))
    inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    shown = xy[inside]
    ax.scatter(
        shown[:, 0],
        shown[:, 1],
        s=_adaptive_point_size(len(shown)) * 0.1,
        c="#d9d9d9",
        linewidths=0,
        rasterized=True,
    )
    visible = np.ma.masked_where(
        np.abs(result.difference) < mask_fraction * result.limit, result.difference
    )
    image = ax.imshow(
        visible,
        extent=(x0, x1, y0, y1),
        origin="lower",
        cmap=cmap,
        vmin=-result.limit,
        vmax=result.limit,
        alpha=0.88,
        aspect="auto",
    )
    if labels is not None:
        _label_groups(ax, shown, labels[inside], min_label_frac, label_names or {})
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    _style_axes(ax, axis_labels)
    cbar = fig.colorbar(image, ax=ax, shrink=0.55, fraction=0.035, pad=0.01)
    cbar.set_ticks([-result.limit, 0.0, result.limit])
    cbar.set_ticklabels([f"{control_label}", "equal", f"{case_label}"])
    cbar.set_label("where cells sit (each specimen weighted equally)", fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=0)
    _lighten_colorbar_outline(cbar)
    ax.set_title(title, fontsize=10)
    fig.text(
        0.5,
        0.01,
        f"{control_label}: {len(result.control_specimens)} specimens   "
        f"{case_label}: {len(result.case_specimens)} specimens   "
        "descriptive, not a test",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#7A7A7A",
    )
    return fig


def strata(
    values: pd.Series | None, n_cells: int, min_cells: int
) -> Sequence[tuple[str, np.ndarray]]:
    """The cell masks to draw one figure each for: every level of ``values``, or all cells.

    Levels with fewer than ``min_cells`` cells are dropped; they hold too few cells for a
    density of anything. ``None`` means one figure over all cells.
    """
    if values is None:
        return [("all", np.ones(n_cells, dtype=bool))]
    text = values.astype(str).to_numpy()
    out = []
    for level in sorted(set(text)):
        mask = text == level
        if int(mask.sum()) >= min_cells:
            out.append((level, mask))
    return out
