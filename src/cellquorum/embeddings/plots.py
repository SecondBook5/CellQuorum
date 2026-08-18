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

from cellquorum.visualization.figstyle import CATEGORICAL_PALETTE as _PALETTE
from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP as _SEQUENTIAL_CMAP
from cellquorum.visualization.figstyle import TEXT as _TEXT

# Single source of truth: tag -> obsm key + axis labels.
EMBEDDING_REGISTRY: dict[str, dict] = {
    "umap": {"obsm": "X_umap", "axis": ("UMAP1", "UMAP2")},
    "phate": {"obsm": "X_phate", "axis": ("PHATE1", "PHATE2")},
}


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
    orig_col = adata.obs[group_key]
    # Category order must match how scanpy indexes uns['paga']['connectivities'].
    # sc.tl.paga builds it from `cat.codes` (see scanpy _paga._compute_connectivities),
    # so connectivity index i corresponds to `cat.categories[i]` — the FULL declared
    # order, unfiltered. Filtering to present categories would shift indices and
    # misalign (or over-run) the matrix. For a non-categorical column scanpy coerces
    # to a sorted categorical, so sorted() reproduces its category order.
    if hasattr(orig_col, "cat") and hasattr(orig_col.cat, "categories"):
        cats = [str(c) for c in orig_col.cat.categories]
    else:
        cats = sorted(orig_col.astype(str).unique())
    # Compare against the stringified column so int/categorical dtypes still match.
    groups = orig_col.astype(str)
    palette = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(cats)}

    fig = Figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111)
    for cat in cats:
        mask = (groups == cat).to_numpy()
        if not mask.any():
            continue  # declared-but-empty category: nothing to draw
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
        if not mask.any():
            continue
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
        n = conn.shape[0]
        # `cats[i]` aligns with connectivity row/col i (both from cat.codes order).
        # A category with no cells yields a NaN centroid; guard so its edges/node
        # are skipped rather than drawn at a bogus position.
        pos = np.full((n, 2), np.nan)
        for i in range(min(n, len(cats))):
            mask = (groups == cats[i]).to_numpy()
            if mask.any():
                pos[i] = xy[mask].mean(axis=0)
        mx = conn.max() or 1.0
        for i in range(n):
            for j in range(i + 1, n):
                w = conn[i, j]
                if w > paga_threshold and not np.isnan(pos[i]).any() and not np.isnan(pos[j]).any():
                    ax.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        color=_TEXT,
                        lw=0.4 + 3.0 * (w / mx),
                        alpha=0.7,
                        solid_capstyle="round",
                        zorder=5,
                    )
        for i in range(min(n, len(cats))):
            if np.isnan(pos[i]).any():
                continue
            ax.scatter(
                [pos[i, 0]],
                [pos[i, 1]],
                s=90,
                c=palette[cats[i]],
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
    cmap: str = _SEQUENTIAL_CMAP,
    sort_high_on_top: bool = True,
    clip_pct: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Color a 2-D embedding scatter by a per-cell value vector.

    ``clip_pct`` view-clips the color scale to the [clip_pct, 100-clip_pct]
    percentiles when explicit ``vmin``/``vmax`` are not supplied (keeps outliers
    from flattening the ramp). For signed layers (e.g. MAGIC z-scores) pass
    ``cmap="RdBu_r", vmin=-2, vmax=2``.
    """
    coords = np.asarray(coords)[:, :2]
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort") if sort_high_on_top else np.arange(len(values))

    if vmin is None and vmax is None and clip_pct > 0 and np.isfinite(values).any():
        finite = values[np.isfinite(values)]
        vmin, vmax = np.percentile(finite, [clip_pct, 100 - clip_pct])

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
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(sctr, ax=ax, shrink=0.55, aspect=18)
    ax.set_title(title)
    _style_axes(ax, axis_labels)
    return fig


def magic_zscore_layer(
    adata: ad.AnnData, *, source_layer: str = "magic", out_layer: str = "magic_z"
) -> bool:
    """Write a per-gene z-scored layer from an existing MAGIC layer.

    Returns True if written, False if ``source_layer`` is absent (skip-not-crash).
    Does NOT compute MAGIC itself.
    """
    if source_layer not in adata.layers:
        return False
    m = adata.layers[source_layer]
    m = m.toarray() if hasattr(m, "toarray") else np.asarray(m, dtype=float)
    mu = m.mean(0, keepdims=True)
    sd = m.std(0, keepdims=True) + 1e-9
    adata.layers[out_layer] = (m - mu) / sd
    return True


__all__ = [
    "EMBEDDING_REGISTRY",
    "categorical_embedding",
    "continuous_overlay",
    "magic_zscore_layer",
]
