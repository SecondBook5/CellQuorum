"""Biology-agnostic embedding plots: categorical (with PAGA overlay) + continuous.

Ported from the house figure library: soft rasterized points, no frame, corner
axis-name arrows, per-group median labels, PAGA nodes at per-group centroids with
connectivity-weighted edges. Works on any basis (UMAP or PHATE).

House-style figure saving integrated from save.py for the embeddings stage.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cellquorum.visualization.figio import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import ORANGE_CMAP as _SEQUENTIAL_CMAP
from cellquorum.visualization.figstyle import TEXT as _TEXT
from cellquorum.visualization.figstyle import muted_palette_colors

# Single source of truth: tag -> obsm key + axis labels.
EMBEDDING_REGISTRY: dict[str, dict] = {
    "umap": {"obsm": "X_umap", "axis": ("UMAP1", "UMAP2")},
    "phate": {"obsm": "X_phate", "axis": ("PHATE1", "PHATE2")},
}


def _lighten_colorbar_outline(cbar: Any) -> None:
    """Give a colorbar the thin, light frame used beside the frameless scatter.

    The default heavy dark outline reads as a clunky box. ``cbar`` is typed ``Any`` because
    matplotlib's type stubs describe ``Colorbar.outline`` as a callable ``Spine``, which
    rejects the ordinary ``set_*`` calls even though they are valid at run time.
    """
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor("#BDBDBD")


def _style_axes(ax: Axes, axis_labels: tuple[str, str]) -> None:
    """Remove frame/ticks, set equal aspect, draw corner axis-name arrows."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
    # The arrows live in a margin opened below and left of the cells, so a cloud that reaches
    # the corner of its view never has an arrow or an axis name drawn across it.
    lo_x, hi_x = xlim[0] - 0.14 * span_x, xlim[1]
    lo_y, hi_y = ylim[0] - 0.14 * span_y, ylim[1]
    # Fit the limits to the shape of the box ourselves. Leaving that to matplotlib (equal aspect on
    # fixed limits) makes it throw one of the limits away whenever the box and the cloud differ in
    # shape, which moves the margin and clips the arrows.
    box = ax.get_position()
    fig_w, fig_h = ax.figure.bbox.width, ax.figure.bbox.height
    box_aspect = (box.height * fig_h) / (box.width * fig_w)
    need_h = (hi_x - lo_x) * box_aspect
    if need_h >= hi_y - lo_y:
        extra = need_h - (hi_y - lo_y)
        lo_y, hi_y = lo_y - extra / 2, hi_y + extra / 2
    else:
        extra = (hi_y - lo_y) / box_aspect - (hi_x - lo_x)
        lo_x, hi_x = lo_x - extra / 2, hi_x + extra / 2
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_aspect("equal", adjustable="box")
    # Positions are fractions of the axes, so the arrows stay in view if the layout changes later.
    width, height = hi_x - lo_x, hi_y - lo_y
    x0, y0 = 0.03 * span_x / width, 0.03 * span_y / height
    dx, dy = 0.12 * span_x / width, 0.12 * span_y / height
    arrow = {"arrowstyle": "-|>", "color": _TEXT, "lw": 1.0}
    for tip in ((x0 + dx, y0), (x0, y0 + dy)):
        ax.annotate(
            "",
            xy=tip,
            xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=arrow,
            annotation_clip=False,
        )
    ax.text(x0 + dx * 1.1, y0, axis_labels[0], fontsize=7, va="center", transform=ax.transAxes)
    ax.text(
        x0,
        y0 + dy * 1.1,
        axis_labels[1],
        fontsize=7,
        ha="center",
        rotation=90,
        transform=ax.transAxes,
    )


# PAGA edge color: a mid grey so the connectivity graph reads as a recessive
# scaffold under the named nodes, never a black hairball on top of the points.
_PAGA_EDGE = "#5a5a5a"


def _adaptive_point_size(n_cells: int) -> float:
    """Marker size that shrinks with cell count so a large cohort keeps its texture.

    A size tuned for ~20k cells renders a 167k-cell cohort as fat overlapping discs —
    the "cartooney" poster-paint look — instead of a scatter whose internal density is
    visible. Calibrated so ~167k cells land near 1.2 (validated by eye against the
    reference atlas figure), with a floor for pathological sizes and a ceiling so small
    datasets still show. Shared by the categorical panel and the continuous overlays so
    the two read as one figure set.
    """
    return float(np.clip(3.5 * (20_000.0 / max(n_cells, 1)) ** 0.5, 0.5, 6.0))


def _figsize_for(n_groups: int) -> tuple[float, float]:
    """Grow the canvas with group count so many named labels have room to repel."""
    if n_groups <= 12:
        return (5.2, 5.0)
    if n_groups <= 25:
        return (7.2, 6.8)
    return (9.2, 8.6)


def _angular_sweep(centroids: dict[str, np.ndarray], cats: list[str]) -> list[str]:
    """Order groups by the angle of their centroid about the global centroid.

    Assigning palette slots in this order is what makes SPATIALLY ADJACENT clusters
    get ADJACENT slots — and adjacent-slot separation is precisely what
    :data:`CATEGORICAL_PALETTE` guarantees across its full 18 slots (its overflow
    tier is adjacent-pair separated, not all-pairs). Assigning by abundance instead
    leaves the pairing to chance, which is how two touching blobs end up a few
    perceptual units apart — the one place on an embedding where colour has to do
    real work, because there is no gap to read the boundary from.
    """
    present = [c for c in cats if c in centroids]
    if len(present) < 3:
        return present
    points = np.array([centroids[c] for c in present], dtype=float)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return [present[i] for i in np.argsort(angles)]


def _palette_from_centroids(centroids: dict[str, np.ndarray], cats: list[str]) -> dict[str, str]:
    """Shared core: muted-palette colors dealt in angular-sweep order.

    Split out of ``categorical_embedding`` so it and
    ``compute_categorical_palette`` assign colors identically without either
    duplicating the other's centroid computation -- ``categorical_embedding``
    already has its own ``centroids`` dict (reused later for PAGA nodes and
    text labels), so it calls this directly; an external caller without
    precomputed centroids should use ``compute_categorical_palette`` instead.
    """
    colors = muted_palette_colors(len(cats))
    sweep = _angular_sweep(centroids, cats)
    palette = {c: colors[i] for i, c in enumerate(sweep)}
    for i, cat in enumerate(c for c in cats if c not in palette):
        palette[cat] = colors[(len(sweep) + i) % len(colors)]
    return palette


def compute_categorical_palette(
    xy: np.ndarray,
    groups: object,
    cats: list[str],
) -> dict[str, str]:
    """The exact category -> color mapping ``categorical_embedding`` uses.

    Extracted so any OTHER figure describing the same categories (a marker
    dotplot's column colors, say) can reuse the identical mapping instead of
    silently drifting from it -- two figures about the same clusters using
    different color assignments defeats the entire point of "color it so
    it's easily identifiable [against the other figure]".

    Args:
        xy: Cells x 2 embedding coordinates (same basis categorical_embedding
            would use), aligned to ``groups``.
        groups: Per-cell category labels (same length as ``xy``; any array-like
            works -- a plain list, numpy array, or pandas Series).
        cats: The full ordered category list to assign colors for.

    Returns:
        ``{category: hex_color}`` for every category with at least one cell in
        ``groups`` intersected with ``cats``, plus any declared-but-absent
        category (dealt a color so callers never KeyError on it).
    """
    groups_arr = np.asarray(groups)
    centroids: dict[str, np.ndarray] = {}
    for cat in cats:
        mask = groups_arr == cat
        if mask.any():
            centroids[cat] = np.array([np.median(xy[mask, 0]), np.median(xy[mask, 1])])
    return _palette_from_centroids(centroids, cats)


def view_limits(
    xy: np.ndarray, view_percentile: tuple[float, float] | None
) -> tuple[tuple[float, float], tuple[float, float], int] | None:
    """The plotted range of an embedding, trimmed to a percentile window.

    Returns the x range, the y range and how many cells fall outside them.

    One isolated cell (a cell the neighbour graph barely connects) can sit far from everything
    else and shrink the whole cloud into a corner. ``view_percentile = (low, high)`` sets each
    axis range to those percentiles of the coordinates plus 6 percent padding. The cells outside
    are not hidden silently: the count is returned so the figure can say so. ``None`` means the
    range covers every cell.
    """
    if view_percentile is None:
        return None
    low, high = view_percentile
    xs, ys = xy[:, 0], xy[:, 1]
    ranges = []
    for values in (xs, ys):
        lo, hi = np.percentile(values, [low, high])
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        ranges.append((float(lo - pad), float(hi + pad)))
    (x0, x1), (y0, y1) = ranges
    outside = int(((xs < x0) | (xs > x1) | (ys < y0) | (ys > y1)).sum())
    return (x0, x1), (y0, y1), outside


def _apply_view(ax: Axes, xy: np.ndarray, view_percentile: tuple[float, float] | None) -> None:
    """Trim an axes to :func:`view_limits` and note how many cells fall outside it."""
    limits = view_limits(xy, view_percentile)
    if limits is None:
        return
    (x0, x1), (y0, y1), outside = limits
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if outside:
        noun = "cell" if outside == 1 else "cells"
        ax.text(
            0.99,
            0.01,
            f"{outside} {noun} outside the plotted range",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#7A7A7A",
        )


def _name_label(ax: Axes, x: float, y: float, text: str, colour: str) -> Any:
    """A cluster's name on an embedding: bold text in a near-white box edged in the cluster colour.

    A faint tint of the cluster colour behind black text went unreadable over dark clusters and
    muddy over pastel ones; white reads over any cluster and the coloured edge still ties the
    name to its cloud. The thin white halo keeps the text legible where it overhangs points.
    """
    return ax.text(
        x,
        y,
        text,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=10,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": colour, "lw": 1.1, "alpha": 0.92},
        path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")],
    )


def column_palette(xy: np.ndarray, column: pd.Series) -> dict[str, str]:
    """The category -> colour mapping ``categorical_embedding`` draws for ``column``.

    Categories are taken the way the figure takes them (the declared order of a categorical
    column, else the sorted unique values), so a figure that colours the same labels --
    composition bars, a dotplot's headers -- gets the same colour for the same cell type as
    the UMAP does, whatever the dtype of the label column. ``xy`` must be the coordinates the
    UMAP figure uses: the palette is dealt in angular order of the group centroids.
    """
    if hasattr(column, "cat") and hasattr(column.cat, "categories"):
        cats = [str(c) for c in column.cat.categories]
    else:
        cats = sorted(column.dropna().astype(str).unique())
    return compute_categorical_palette(np.asarray(xy)[:, :2], column.astype(str).to_numpy(), cats)


def _label_anchor(points: np.ndarray) -> np.ndarray:
    """The cell of a group nearest the group's median, where its name is placed.

    The median of a lobed, ringed or crescent-shaped group can fall in empty space or on a
    neighbouring group, which then carries the wrong name. A cell of the group itself cannot.
    For a compact group this is the middle, as before.
    """
    points = np.asarray(points, dtype=float)
    middle = np.median(points, axis=0)
    return points[int(np.argmin(((points - middle) ** 2).sum(axis=1)))]


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
        if isinstance(fig, Figure) and not hasattr(fig.canvas, "get_renderer"):
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
    point_size: float | None = None,
    paga_overlay: bool = True,
    min_label_frac: float = 0.001,
    legend: bool = True,
    title: str = "",
    cell_mask: np.ndarray | None = None,
    cluster_labels: dict[str, str] | None = None,
    view_percentile: tuple[float, float] | None = None,
) -> Figure:
    """Per-group scatter on `basis`, with PAGA graph overlaid when present.

    PAGA nodes are drawn at per-group centroids in the embedding; edges are the
    upper-triangle connectivities above `paga_threshold`, width ~ normalized weight.
    Categories iterate in the categorical's category order (or sorted for non-categorical).

    ``point_size=None`` scales the marker with cell count, because a size tuned for
    20,000 cells paints a 200,000-cell cohort into flat blocks where the rare
    populations are the ones that disappear.

    ``min_label_frac`` suppresses the on-plot name (and the PAGA node) for any group
    holding less than that fraction of cells. Without it a 2-cell category is drawn
    with the same weight as a 70,000-cell lineage, and its label lands in the middle
    of a real cluster it has no claim to.

    ``legend`` adds a side legend carrying every group with its cell count, including
    the small ones ``min_label_frac`` leaves unnamed on the plot — so nothing drawn is
    unidentifiable, and the count that decides whether a population is real is on the
    figure rather than in a table beside it.

    ``cell_mask`` restricts the panel to a subset of cells (e.g. QC-core only) WITHOUT
    subsetting the object, so no multi-gigabyte copy is made to draw a figure. The
    category order is still taken from the unfiltered column, because that order is what
    ``uns['paga']['connectivities']`` is indexed by; only the drawn points, the counts,
    and the centroids come from the subset.

    ``cluster_labels`` renames the in-plot text and legend only (e.g. {"0":
    "Papillary"}) -- grouping, the palette, and PAGA connectivity indexing all
    stay keyed on the raw category value throughout.
    """
    cluster_labels = cluster_labels or {}
    xy = np.asarray(adata.obsm[basis])[:, :2]
    orig_col = adata.obs[group_key]
    if cell_mask is not None:
        cell_mask = np.asarray(cell_mask, dtype=bool)
        if cell_mask.shape[0] != adata.n_obs:
            raise ValueError(f"cell_mask has {cell_mask.shape[0]} entries for {adata.n_obs} cells.")
        if not cell_mask.any():
            raise ValueError("cell_mask selects no cells.")
        xy = xy[cell_mask]
        orig_col = orig_col[cell_mask]
    # Category order must match how scanpy indexes uns['paga']['connectivities'].
    # sc.tl.paga builds it from `cat.codes` (see scanpy _paga._compute_connectivities),
    # so connectivity index i corresponds to `cat.categories[i]` — the FULL declared
    # order, unfiltered. Filtering to present categories would shift indices and
    # misalign (or over-run) the matrix. For a non-categorical column scanpy coerces
    # to a sorted categorical, so sorted() reproduces its category order.
    if hasattr(orig_col, "cat") and hasattr(orig_col.cat, "categories"):
        cats = [str(c) for c in orig_col.cat.categories]
    else:
        cats = sorted(orig_col.dropna().astype(str).unique())
    # Compare against the stringified column so int/categorical dtypes still match.
    groups = orig_col.astype(str)
    missing_label = "Unassigned (missing)"
    while missing_label in cats:
        missing_label += " [missing]"
    missing_groups = orig_col.isna()
    groups = groups.where(~missing_groups, missing_label)
    # Per-group centroid (per-axis median: robust to trailing arcs/stragglers and
    # always sits inside the point cloud). Computed once and reused for the text
    # label, the PAGA node, and the palette sweep below, so every node sits exactly
    # under its label.
    anchors: dict[str, np.ndarray] = {}  # where names go; centroids stay for PAGA nodes + colours
    centroids: dict[str, np.ndarray] = {}
    for _cat in cats:
        _m = (groups == _cat).to_numpy()
        if _m.any():
            centroids[_cat] = np.array([np.median(xy[_m, 0]), np.median(xy[_m, 1])])
            anchors[_cat] = _label_anchor(xy[_m])

    # Colors come from the MUTED atlas palette, not the bright CATEGORICAL_PALETTE. A
    # 200,000-cell UMAP in saturated primaries reads as clip-art; the desaturated jewel
    # tones are what make it read as data (and match the published atlas figures). Still
    # a fixed, non-repeating authority — never the golden-angle generator directly, which
    # is what once painted a 15-category atlas in raw vivids.
    #
    # Slots are then dealt in ANGULAR SWEEP order, not category order, so spatial
    # neighbours land on adjacent slots. See `_angular_sweep`. Shared with
    # compute_categorical_palette (for external figures that want these same
    # colors) via `_palette_from_centroids`, so the two can never drift apart.
    palette = _palette_from_centroids(centroids, cats)

    if missing_groups.any():
        palette[missing_label] = "#999999"

    # Counts drive three separate decisions below: draw order, the label floor, and
    # which PAGA nodes are worth drawing.
    counts = groups.value_counts()
    n_obs = int(len(groups))
    label_floor = max(1, int(round(min_label_frac * n_obs)))

    # Marker area shrinks as the cohort grows, so a 200,000-cell atlas keeps its
    # internal structure visible instead of saturating into solid colour.
    size = _adaptive_point_size(n_obs) if point_size is None else point_size

    width, height = _figsize_for(len(cats))
    if legend:
        width += 2.0  # room for the side legend rather than squeezing the plot
    fig = Figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    # ABUNDANT FIRST, so rare populations are drawn last and stay visible. Iterating
    # `cats` instead put whichever category happened to sort last on top: on this
    # cohort the 70,000-cell fibroblast blob buried LEC and Plasma, the two
    # populations the analysis is about. `cats` order is still what the palette and
    # the PAGA connectivity matrix are indexed by, so it is kept for both.
    draw_order = [c for c in counts.index if c in palette]
    for cat in draw_order:
        mask = (groups == cat).to_numpy()
        if not mask.any():
            continue  # declared-but-empty category: nothing to draw
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            c=palette[cat],
            # 0.6, not full opacity: overlapping points then build visible density
            # within a cluster instead of flooding it to one flat poster colour.
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            label=cat,
        )

    # PAGA overlay (nodes at centroids, thresholded connectivity edges). Edges
    # are a recessive grey scaffold: width AND opacity both scale with the
    # normalized connectivity, so weak links fade toward invisible instead of
    # crowding the plot into a black hairball when there are many groups.
    node_size = 90.0 if len(cats) <= 15 else 55.0
    paga = adata.uns.get("paga") if paga_overlay else None
    if paga is not None and "connectivities" in paga:
        conn = paga["connectivities"]
        conn = conn.toarray() if hasattr(conn, "toarray") else np.asarray(conn)
        n = conn.shape[0]
        # `cats[i]` aligns with connectivity row/col i (both from cat.codes order).
        # A category with no cells yields a NaN centroid; guard so its edges/node
        # are skipped rather than drawn at a bogus position.
        # Below-floor categories are left NaN so neither their node nor their edges
        # are drawn: a node whose centroid is a handful of cells sits wherever those
        # cells happen to fall, and every edge it carries is drawn to that accident.
        pos = np.full((n, 2), np.nan)
        for i in range(min(n, len(cats))):
            centroid = centroids.get(cats[i])
            if centroid is not None and int(counts.get(cats[i], 0)) >= label_floor:
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
    _apply_view(ax, xy, view_percentile)
    _style_axes(ax, axis_labels)

    # Per-group NAMED labels on top, then de-collided with leader lines.
    texts = []
    for cat in cats:
        anchor = anchors.get(cat)
        if anchor is None or int(counts.get(cat, 0)) < label_floor:
            continue
        texts.append(
            _name_label(
                ax, anchor[0], anchor[1], cluster_labels.get(cat, cat), palette.get(cat, "#888888")
            )
        )
    _repel_labels(ax, texts)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Side legend, ordered by abundance and carrying counts. Built from proxy handles
    # rather than the scatter labels so the ORDER is abundance (what a reader scans
    # for) while the DRAW order stays rare-on-top (what keeps rare groups visible) —
    # the two orders are deliberately different and a shared handle list would force
    # them to be the same.
    if legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=5,
                markerfacecolor=palette[cat],
                markeredgecolor="none",
                label=f"{cluster_labels.get(cat, cat)} ({int(counts[cat]):,})",
            )
            for cat in counts.index
            if cat in palette
        ]
        if handles:
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=7.5,
                handletextpad=0.3,
                labelspacing=0.5,
                borderaxespad=0.0,
            )
    return fig


def continuous_overlay(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    axis_labels: tuple[str, str],
    cmap: str = _SEQUENTIAL_CMAP,
    sort_high_on_top: bool = True,
    clip_pct: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float | None = None,
    cbar_label: str | None = None,
    view_percentile: tuple[float, float] | None = None,
) -> Figure:
    """Color a 2-D embedding scatter by a per-cell value vector.

    ``clip_pct`` view-clips the color scale to the [clip_pct, 100-clip_pct]
    percentiles when explicit ``vmin``/``vmax`` are not supplied (keeps outliers
    from flattening the ramp). ``None`` picks a gentle default (2% each tail) for
    unsigned data, because a handful of very high-expressing cells otherwise
    compress every other cell into the dark end of the ramp and the panel reads as
    "off everywhere but a few dots". For signed layers (e.g. MAGIC z-scores) pass
    ``cmap="RdBu_r", vmin=-2, vmax=2``.

    ``point_size=None`` scales the marker with cell count, matching the categorical
    panel: a size tuned for ~20k cells renders a 167k-cell cohort as fat overlapping
    discs (the "cartooney" look) rather than a smooth density.

    ``view_percentile`` trims the plotted range to that percentile window and writes how many
    cells fall outside it (see :func:`view_limits`).

    ``cbar_label`` names the colour scale; without it the colour bar carries only numbers,
    so a caller drawing anything other than a self-explanatory score should pass one.
    """
    coords = np.asarray(coords)[:, :2]
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort") if sort_high_on_top else np.arange(len(values))

    # Gentle default clip so outliers do not flatten the ramp. Skipped when the caller
    # sets explicit vmin/vmax (signed layers), and harmless on near-constant data.
    if vmin is None and vmax is None:
        pct = 2.0 if clip_pct is None else clip_pct
        if pct > 0 and np.isfinite(values).any():
            finite = values[np.isfinite(values)]
            lo, hi = np.percentile(finite, [pct, 100 - pct])
            if hi > lo:  # a degenerate all-equal vector would give lo==hi
                vmin, vmax = lo, hi

    # Same n-adaptive sizing as the categorical panel, so the two read as one figure set.
    size = _adaptive_point_size(len(values)) if point_size is None else point_size

    fig = Figure(figsize=(5.6, 5.0))
    ax = fig.add_subplot(111)
    sctr = ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=values[order],
        cmap=cmap,
        s=size,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(sctr, ax=ax, shrink=0.55, aspect=22, pad=0.02)
    # Thin, light frame: the default heavy dark outline reads as a clunky box beside
    # the frameless scatter.
    _lighten_colorbar_outline(cbar)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.4)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title)
    _apply_view(ax, coords, view_percentile)
    _style_axes(ax, axis_labels)
    return fig


def plot_score_by_condition(
    adata: ad.AnnData,
    *,
    score: np.ndarray,
    score_label: str,
    cluster_key: str,
    basis: str,
    axis_labels: tuple[str, str],
    condition_col: str,
    case: str,
    control: str,
    case_label: str | None = None,
    control_label: str | None = None,
    cluster_labels: dict[str, str] | None = None,
    cmap: str = _SEQUENTIAL_CMAP,
    clip_pct: float = 2.0,
    point_size: float | None = None,
    view_percentile: tuple[float, float] | None = None,
    show_state_panel: bool = True,
) -> Figure:
    """3-panel figure: cluster-state UMAP, then a per-cell score split by condition.

    ``show_state_panel=False`` drops the state map and keeps only the control and
    case panels (control on the left) with their shared colourbar.

    Panel 1 is a plain categorical scatter (no PAGA, no side legend -- just
    in-plot cluster labels, matching the reference "state map" layout this
    was built to reproduce). Panels 2 and 3 color the SAME score by the
    control and case condition respectively, on ONE shared color scale
    (computed once, across all cells, before either panel is drawn) with a
    single shared colorbar -- two independently-clipped scales would let the
    same color mean different score values in each panel, defeating the
    point of putting them side by side.

    ``cluster_labels`` renames panel 1's in-plot text only; grouping and the
    palette stay keyed on the raw cluster value, same convention as
    ``categorical_embedding``.
    """
    cluster_labels = cluster_labels or {}
    control_label = control_label or control
    case_label = case_label or case

    coords = np.asarray(adata.obsm[basis])[:, :2]
    score = np.asarray(score, dtype=float)
    cluster_col = adata.obs[cluster_key].astype(str)
    cond_col = adata.obs[condition_col].astype(str)

    if hasattr(adata.obs[cluster_key], "cat"):
        cats = [str(c) for c in adata.obs[cluster_key].cat.categories]
    else:
        cats = sorted(cluster_col.dropna().unique())

    centroids: dict[str, np.ndarray] = {}
    for cat in cats:
        mask = (cluster_col == cat).to_numpy()
        if mask.any():
            centroids[cat] = np.array([np.median(coords[mask, 0]), np.median(coords[mask, 1])])
    palette = _palette_from_centroids(centroids, cats)

    n_obs = int(len(score))
    size = _adaptive_point_size(n_obs) if point_size is None else point_size

    if show_state_panel:
        fig = Figure(figsize=(15.5, 5.0))
        ax_state = fig.add_subplot(131)
        ax_control = fig.add_subplot(132)
        ax_case = fig.add_subplot(133)
    else:
        fig = Figure(figsize=(10.5, 3.9))
        ax_state = None
        ax_control = fig.add_subplot(121)
        ax_case = fig.add_subplot(122)

    # --- Panel 1: cluster state map --------------------------------------
    if ax_state is not None:
        for cat in cats:
            mask = (cluster_col == cat).to_numpy()
            if not mask.any():
                continue
            ax_state.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=size,
                c=palette[cat],
                alpha=0.6,
                linewidths=0,
                rasterized=True,
            )
        texts = []
        for cat in cats:
            centroid = centroids.get(cat)
            if centroid is None:
                continue
            texts.append(
                _name_label(
                    ax_state,
                    centroid[0],
                    centroid[1],
                    cluster_labels.get(cat, cat),
                    palette.get(cat, "#888888"),
                )
            )
        ax_state.margins(0.10)
        _apply_view(ax_state, coords, view_percentile)
        _repel_labels(ax_state, texts)
        _style_axes(ax_state, axis_labels)
        ax_state.set_title(f"{n_obs:,} cells", fontsize=9, color=_TEXT)

    # --- Shared color scale, computed ONCE across all cells --------------
    finite = score[np.isfinite(score)]
    if finite.size and clip_pct > 0:
        vmin, vmax = np.percentile(finite, [clip_pct, 100 - clip_pct])
        if vmin == vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    # --- Panels 2/3: same score, split by condition, same vmin/vmax -----
    sctr = None
    for ax, label, cond_value in (
        (ax_control, control_label, control),
        (ax_case, case_label, case),
    ):
        mask = (cond_col == cond_value).to_numpy()
        order = np.argsort(score[mask], kind="mergesort")
        sctr = ax.scatter(
            coords[mask][order, 0],
            coords[mask][order, 1],
            c=score[mask][order],
            cmap=cmap,
            s=size,
            alpha=0.9,
            linewidths=0,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )
        _apply_view(ax, coords, view_percentile)
        _style_axes(ax, axis_labels)
        ax.set_title(f"{label}\n{int(mask.sum()):,} cells", fontsize=9, color=_TEXT)

    if sctr is not None:
        cbar = fig.colorbar(sctr, ax=[ax_control, ax_case], shrink=0.7, aspect=25, pad=0.02)
        cbar.set_label(score_label, fontsize=8)
        _lighten_colorbar_outline(cbar)
        cbar.ax.tick_params(labelsize=8, length=2, width=0.4)

    return fig


def plot_score_pooled(
    adata: ad.AnnData,
    *,
    score: np.ndarray,
    score_label: str,
    title: str,
    basis: str,
    axis_labels: tuple[str, str],
    cmap: str = _SEQUENTIAL_CMAP,
    clip_pct: float = 0.0,
    point_size: float | None = None,
    view_percentile: tuple[float, float] | None = None,
) -> Figure:
    """One UMAP of every cell coloured by ``score``, titled, with a labelled colourbar.

    The colour scale runs over the full range of the score unless ``clip_pct`` trims its
    tails, so the colourbar's ends are the real minimum and maximum. Cells are drawn in
    ascending score order so the highest-scoring ones are never buried.
    """
    coords = np.asarray(adata.obsm[basis])[:, :2]
    score = np.asarray(score, dtype=float)
    size = _adaptive_point_size(len(score)) if point_size is None else point_size

    finite = score[np.isfinite(score)]
    vmin = vmax = None
    if finite.size:
        vmin, vmax = np.percentile(finite, [clip_pct, 100 - clip_pct])
        if vmin == vmax:
            vmin = vmax = None

    fig = Figure(figsize=(5.6, 4.2))
    ax = fig.add_subplot(111)
    order = np.argsort(score, kind="mergesort")
    cells = ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=score[order],
        cmap=cmap,
        s=size,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.margins(0.05)
    _apply_view(ax, coords, view_percentile)
    _style_axes(ax, axis_labels)
    ax.set_title(title, fontsize=10, color=_TEXT)

    cbar = fig.colorbar(cells, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label(score_label, fontsize=8)
    _lighten_colorbar_outline(cbar)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.4)
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


# save_figure/figure_artifacts are re-exported, not redefined: the local copy was
# a bare savefig loop that left truncated files behind and abandoned the remaining
# formats when one raised mid-write. See visualization.figio.


__all__ = [
    "EMBEDDING_REGISTRY",
    "categorical_embedding",
    "continuous_overlay",
    "figure_artifacts",
    "magic_zscore_layer",
    "plot_score_pooled",
    "plot_score_by_condition",
    "save_figure",
]
