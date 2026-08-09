"""Biology-agnostic embedding plots: categorical (with PAGA overlay) + continuous.

Ported from the house figure library: soft rasterized points, no frame, corner
axis-name arrows, per-group median labels, PAGA nodes at per-group centroids with
connectivity-weighted edges. Works on any basis (UMAP or PHATE).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Single source of truth: tag -> obsm key + axis labels.
EMBEDDING_REGISTRY: dict[str, dict] = {
    "umap": {"obsm": "X_umap", "axis": ("UMAP1", "UMAP2")},
    "phate": {"obsm": "X_phate", "axis": ("PHATE1", "PHATE2")},
}

_TEXT = "#202428"
_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _style_axes(ax: Axes, axis_labels: tuple[str, str]) -> None:
    """Remove frame/ticks, set equal aspect, draw corner axis-name arrows."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    dx = (xlim[1] - xlim[0]) * 0.16
    dy = (ylim[1] - ylim[0]) * 0.16
    arrow = {"arrowstyle": "-|>", "color": _TEXT, "lw": 1.0}
    ax.annotate("", xy=(x0 + dx, y0), xytext=(x0, y0), arrowprops=arrow)
    ax.annotate("", xy=(x0, y0 + dy), xytext=(x0, y0), arrowprops=arrow)
    ax.text(x0 + dx * 1.1, y0, axis_labels[0], fontsize=7, va="center")
    ax.text(x0, y0 + dy * 1.1, axis_labels[1], fontsize=7, ha="center", rotation=90)


def categorical_embedding(
    adata: ad.AnnData,
    group_key: str,
    *,
    basis: str,
    axis_labels: tuple[str, str],
    paga_threshold: float = 0.2,
    point_size: float = 6.0,
) -> Figure:
    """Per-group scatter on `basis`, with PAGA graph overlaid when present.

    PAGA nodes are drawn at per-group centroids in the embedding; edges are the
    upper-triangle connectivities above `paga_threshold`, width ~ normalized weight.
    Categories iterate in the categorical's category order (or sorted for non-categorical).
    """
    xy = np.asarray(adata.obsm[basis])[:, :2]
    groups = adata.obs[group_key].astype(str)
    # Determine category order: use categorical order if available, else sorted.
    orig_col = adata.obs[group_key]
    if hasattr(orig_col, "cat") and hasattr(orig_col.cat, "categories"):
        # Categorical: use category order, filtered to actually-present categories.
        present = set(orig_col.dropna().unique())
        cats = [c for c in orig_col.cat.categories if c in present]
    else:
        # Non-categorical: scanpy coerces to sorted categorical, so sorted is correct.
        cats = sorted(groups.unique())
    palette = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}

    fig = Figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111)
    for cat in cats:
        mask = (groups == cat).to_numpy()
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            c=palette[cat],
            alpha=0.8,
            linewidths=0,
            rasterized=True,
            label=cat,
        )
    # Per-group median labels.
    for cat in cats:
        mask = (groups == cat).to_numpy()
        cx, cy = np.median(xy[mask, 0]), np.median(xy[mask, 1])
        ax.annotate(
            cat,
            (cx, cy),
            fontsize=7,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
        )

    # PAGA overlay (nodes at centroids, thresholded connectivity edges).
    paga = adata.uns.get("paga")
    if paga is not None and "connectivities" in paga:
        conn = paga["connectivities"]
        conn = conn.toarray() if hasattr(conn, "toarray") else np.asarray(conn)
        pos = np.vstack([xy[(groups == c).to_numpy()].mean(axis=0) for c in cats])
        mx = conn.max() or 1.0
        n = conn.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                w = conn[i, j]
                if w > paga_threshold:
                    ax.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        color=_TEXT,
                        lw=0.4 + 3.0 * (w / mx),
                        alpha=0.7,
                        solid_capstyle="round",
                        zorder=5,
                    )
        for i, cat in enumerate(cats):
            ax.scatter(
                [pos[i, 0]],
                [pos[i, 1]],
                s=90,
                c=palette[cat],
                edgecolors="white",
                linewidths=1.2,
                zorder=6,
            )

    _style_axes(ax, axis_labels)
    return fig


def continuous_overlay(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    axis_labels: tuple[str, str],
    cmap: str = "magma",
    sort_high_on_top: bool = True,
) -> Figure:
    """Color a 2-D embedding scatter by a per-cell value vector."""
    coords = np.asarray(coords)[:, :2]
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort") if sort_high_on_top else np.arange(len(values))

    fig = Figure(figsize=(5.0, 5.0))
    ax = fig.add_subplot(111)
    sctr = ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=values[order],
        cmap=cmap,
        s=6.0,
        linewidths=0,
        rasterized=True,
    )
    fig.colorbar(sctr, ax=ax, shrink=0.55, aspect=18)
    ax.set_title(title)
    _style_axes(ax, axis_labels)
    return fig


__all__ = ["EMBEDDING_REGISTRY", "categorical_embedding", "continuous_overlay"]
