"""Biology-agnostic embedding plots: categorical (with PAGA overlay) + continuous.

Ported from the house figure library: soft rasterized points, no frame, corner
axis-name arrows, per-group median labels, PAGA nodes at per-group centroids with
connectivity-weighted edges. Works on any basis (UMAP or PHATE).

House-style figure saving integrated from save.py for the embeddings stage.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cellquorum.core.stage import StageArtifact
from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP as _SEQUENTIAL_CMAP
from cellquorum.visualization.figstyle import TEXT as _TEXT
from cellquorum.visualization.figstyle import apply_cellquorum_theme, distinct_palette

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


# PAGA edge color: a mid grey so the connectivity graph reads as a recessive
# scaffold under the named nodes, never a black hairball on top of the points.
_PAGA_EDGE = "#5a5a5a"


def _figsize_for(n_groups: int) -> tuple[float, float]:
    """Grow the canvas with group count so many named labels have room to repel."""
    if n_groups <= 12:
        return (5.2, 5.0)
    if n_groups <= 25:
        return (7.2, 6.8)
    return (9.2, 8.6)


def _repel_labels(ax: Axes, texts: list) -> None:
    """De-collide per-group text labels with leader lines (best-effort).

    Uses adjustText when importable; on any failure (package absent, no
    renderer) the labels simply stay at their centroids — the figure still
    renders, it is only less tidy. Never raises into the caller.
    """
    if len(texts) < 2:
        return
    try:
        from adjustText import adjust_text
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = ax.get_figure()
        # A bare Figure() has no renderer; adjustText needs one to measure
        # text extents. Attaching an Agg canvas provides it without pyplot.
        if not hasattr(fig.canvas, "get_renderer"):
            FigureCanvasAgg(fig)
        adjust_text(
            texts,
            ax=ax,
            arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.4},
        )
    except Exception:  # noqa: BLE001 — cosmetic; degrade to un-repelled labels
        pass


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
    # One distinct jewel/vivid color per category (a generator, not a cycled
    # fixed list): N categories get N distinct colors, so no two clusters share
    # a hue the way cycling a short list would.
    colors = distinct_palette(len(cats))
    palette = {c: colors[i] for i, c in enumerate(cats)}
    # Per-group centroid (per-axis median: robust to trailing arcs/stragglers and
    # always sits inside the point cloud). Computed once and reused for BOTH the
    # text label and the PAGA node, so every node sits exactly under its label.
    centroids: dict[str, np.ndarray] = {}
    for _cat in cats:
        _m = (groups == _cat).to_numpy()
        if _m.any():
            centroids[_cat] = np.array([np.median(xy[_m, 0]), np.median(xy[_m, 1])])

    fig = Figure(figsize=_figsize_for(len(cats)))
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

    # PAGA overlay (nodes at centroids, thresholded connectivity edges). Edges
    # are a recessive grey scaffold: width AND opacity both scale with the
    # normalized connectivity, so weak links fade toward invisible instead of
    # crowding the plot into a black hairball when there are many groups.
    node_size = 90.0 if len(cats) <= 15 else 55.0
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
            centroid = centroids.get(cats[i])
            if centroid is not None:
                pos[i] = centroid
        mx = conn.max() or 1.0
        for i in range(n):
            for j in range(i + 1, n):
                w = conn[i, j]
                if w > paga_threshold and not np.isnan(pos[i]).any() and not np.isnan(pos[j]).any():
                    wn = w / mx
                    ax.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        color=_PAGA_EDGE,
                        lw=0.2 + 1.8 * wn**1.5,
                        alpha=0.12 + 0.5 * wn**1.5,
                        solid_capstyle="round",
                        zorder=2,
                    )
        for i in range(min(n, len(cats))):
            if np.isnan(pos[i]).any():
                continue
            ax.scatter(
                [pos[i, 0]],
                [pos[i, 1]],
                s=node_size,
                c=palette[cats[i]],
                edgecolors="white",
                linewidths=1.2,
                zorder=6,
            )

    # Pad the view so repelled labels have somewhere to go, then finalize the
    # axis frame before placing labels (adjustText measures against final lims).
    ax.margins(0.10)
    _style_axes(ax, axis_labels)

    # Per-group NAMED labels on top, then de-collided with leader lines.
    texts = []
    for cat in cats:
        centroid = centroids.get(cat)
        if centroid is None:
            continue
        texts.append(
            ax.text(
                centroid[0],
                centroid[1],
                cat,
                fontsize=7,
                ha="center",
                va="center",
                zorder=10,
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.8},
            )
        )
    _repel_labels(ax, texts)
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


def apply_theme() -> None:
    """Apply the house theme plus embeddings vector-font overrides."""
    apply_cellquorum_theme()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Write ``fig`` to ``out_dir/stem.<fmt>`` for each format, then close it.

    Creates ``out_dir`` (and parents) if absent. Returns paths in ``formats`` order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def figure_artifacts(paths: list[Path], *, name: str, description: str) -> list[StageArtifact]:
    """Wrap saved figure paths as ``kind='figure'`` stage artifacts."""
    return [
        StageArtifact(name=name, path=path, kind="figure", description=description)
        for path in paths
    ]


__all__ = [
    "EMBEDDING_REGISTRY",
    "apply_theme",
    "categorical_embedding",
    "continuous_overlay",
    "figure_artifacts",
    "magic_zscore_layer",
    "save_figure",
]
