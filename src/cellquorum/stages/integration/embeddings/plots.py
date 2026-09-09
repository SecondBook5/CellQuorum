"""Biology-agnostic embedding plots: categorical (with PAGA overlay) + continuous.

Ported from the house figure library: soft rasterized points, no frame, corner
axis-name arrows, per-group median labels, PAGA nodes at per-group centroids with
connectivity-weighted edges. Works on any basis (UMAP or PHATE).

House-style figure saving integrated from save.py for the embeddings stage.
"""

from __future__ import annotations

import anndata as ad
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cellquorum.visualization.figio import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP as _SEQUENTIAL_CMAP
from cellquorum.visualization.figstyle import TEXT as _TEXT
from cellquorum.visualization.figstyle import apply_cellquorum_theme, muted_palette_colors

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
    point_size: float | None = None,
    paga_overlay: bool = True,
    min_label_frac: float = 0.001,
    legend: bool = True,
    title: str = "",
    cell_mask: np.ndarray | None = None,
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
    """
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
        cats = sorted(orig_col.astype(str).unique())
    # Compare against the stringified column so int/categorical dtypes still match.
    groups = orig_col.astype(str)
    # Per-group centroid (per-axis median: robust to trailing arcs/stragglers and
    # always sits inside the point cloud). Computed once and reused for the text
    # label, the PAGA node, and the palette sweep below, so every node sits exactly
    # under its label.
    centroids: dict[str, np.ndarray] = {}
    for _cat in cats:
        _m = (groups == _cat).to_numpy()
        if _m.any():
            centroids[_cat] = np.array([np.median(xy[_m, 0]), np.median(xy[_m, 1])])

    # Colors come from the MUTED atlas palette, not the bright CATEGORICAL_PALETTE. A
    # 200,000-cell UMAP in saturated primaries reads as clip-art; the desaturated jewel
    # tones are what make it read as data (and match the published atlas figures). Still
    # a fixed, non-repeating authority — never the golden-angle generator directly, which
    # is what once painted a 15-category atlas in raw vivids.
    #
    # Slots are then dealt in ANGULAR SWEEP order, not category order, so spatial
    # neighbours land on adjacent slots. See `_angular_sweep`.
    colors = muted_palette_colors(len(cats))
    sweep = _angular_sweep(centroids, cats)
    palette = {c: colors[i] for i, c in enumerate(sweep)}
    # Declared-but-absent categories still need a color for the PAGA node loop.
    for _i, _cat in enumerate(c for c in cats if c not in palette):
        palette[_cat] = colors[(len(sweep) + _i) % len(colors)]

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
    _style_axes(ax, axis_labels)

    # Per-group NAMED labels on top, then de-collided with leader lines.
    texts = []
    for cat in cats:
        centroid = centroids.get(cat)
        if centroid is None or int(counts.get(cat, 0)) < label_floor:
            continue
        texts.append(
            ax.text(
                centroid[0],
                centroid[1],
                cat,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10,
                clip_on=False,
                # Faint tint of the cluster's own colour behind bold black text, as in the
                # published atlas — a plain white box reads flatter and more clip-art.
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": palette.get(cat, "#FFFFFF"),
                    "ec": "none",
                    "alpha": 0.28,
                },
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
                label=f"{cat} ({int(counts[cat]):,})",
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
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor("#BDBDBD")
    cbar.ax.tick_params(labelsize=8, length=2, width=0.4)
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


# save_figure/figure_artifacts are re-exported, not redefined: the local copy was
# a bare savefig loop that left truncated files behind and abandoned the remaining
# formats when one raised mid-write. See visualization.figio.


__all__ = [
    "EMBEDDING_REGISTRY",
    "apply_theme",
    "categorical_embedding",
    "continuous_overlay",
    "figure_artifacts",
    "magic_zscore_layer",
    "save_figure",
]
