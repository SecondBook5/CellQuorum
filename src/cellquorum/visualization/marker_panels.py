"""Marker-gene figures for annotation QA: dotplot + per-lineage feature grid.

The Clarke et al. 2021 (Nat Protocols) style manual-annotation figures: does
each cell type actually express its own canonical markers, and only its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import apply_theme, cell_type_palette, save_figure

if TYPE_CHECKING:
    from collections.abc import Mapping

    import anndata as ad
    from matplotlib.figure import Figure


def plot_marker_dotplot(
    adata: ad.AnnData,
    marker_panel: Mapping[str, list[str]],
    *,
    groupby: str,
    layer: str,
    out_path: Path,
    categories_order: list[str] | None = None,
    swap_axes: bool = False,
) -> Path | None:
    """
    Plot a canonical marker-panel dotplot (dot = % detected, color = scaled mean).

    Args:
        adata: Object carrying ``layer`` and ``groupby``.
        marker_panel: Cell type -> ordered list of canonical marker genes.
            Genes absent from ``adata.var_names`` are dropped; a cell type left
            with none is dropped entirely rather than drawing an empty column.
        groupby: obs column to group cells by (rows of the dotplot, unless
            ``swap_axes``).
        layer: Expression layer to read (log-normalized).
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
        categories_order: Row order for ``groupby``'s categories. None keeps
            the column's own category order.
        swap_axes: When True, genes run down the rows (grouped by panel) and
            ``groupby`` categories run across the columns -- the layout for a
            per-cluster marker-panel profile, as opposed to the default
            per-cell-type-marker layout. For a fully custom, densely
            categorized version of this same layout (boxed category labels,
            generous spacing), see ``plot_categorized_dotplot`` instead.

    Returns:
        The PNG path written, or ``None`` when no panel gene survives the
        intersection with ``adata.var_names``.
    """
    import scanpy as sc

    present_panel = {
        group: [g for g in genes if g in adata.var_names] for group, genes in marker_panel.items()
    }
    present_panel = {group: genes for group, genes in present_panel.items() if genes}
    if not present_panel:
        return None

    apply_theme("notebook")
    out_path = Path(out_path)

    kwargs: dict[str, Any] = {}
    if categories_order is not None:
        kwargs["categories_order"] = [c for c in categories_order if c in present_panel]
    if swap_axes:
        kwargs["swap_axes"] = True

    dp = sc.pl.dotplot(
        adata,
        present_panel,
        groupby=groupby,
        layer=layer,
        standard_scale="var",
        show=False,
        return_fig=True,
        **kwargs,
    )
    dp.make_figure()

    paths = save_figure(dp.fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


def plot_marker_feature_grid(
    adata: ad.AnnData,
    markers: Mapping[str, str],
    *,
    basis: str,
    layer: str,
    out_path: Path,
    n_cols: int = 5,
) -> Path | None:
    """
    Plot one defining marker gene per cell type, small multiples on the embedding.

    Args:
        adata: Object carrying ``obsm[basis]`` and ``layer``.
        markers: Cell type -> single most-specific marker gene.
        basis: obsm key of the 2-D embedding to plot on.
        layer: Expression layer to read (log-normalized).
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
        n_cols: Panels per row.

    Returns:
        The PNG path written, or ``None`` when no marker gene survives the
        intersection with ``adata.var_names``, or ``basis`` is absent.
    """
    if basis not in adata.obsm:
        return None

    present = [(cell_type, gene) for cell_type, gene in markers.items() if gene in adata.var_names]
    if not present:
        return None

    import matplotlib.pyplot as plt
    import scipy.sparse as sp

    apply_theme("notebook")
    out_path = Path(out_path)

    genes = [gene for _, gene in present]
    sub = adata[:, genes]
    matrix = sub.layers[layer]
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)

    coords = np.asarray(adata.obsm[basis])[:, :2]
    palette = cell_type_palette([ct for ct, _ in present])

    n_rows = int(np.ceil(len(present) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for panel_index, (cell_type, gene) in enumerate(present):
        ax = axes[panel_index]
        values = matrix[:, panel_index]
        expressing = values > 0
        ax.scatter(coords[:, 0], coords[:, 1], c="#dcdee0", s=1.2, lw=0, rasterized=True)
        if expressing.any():
            order = np.argsort(values[expressing])
            vmax = float(np.quantile(values[expressing], 0.99)) or 1.0
            ax.scatter(
                coords[expressing, 0][order],
                coords[expressing, 1][order],
                c=values[expressing][order],
                cmap="magma",
                vmin=0,
                vmax=vmax,
                s=1.4,
                lw=0,
                rasterized=True,
            )
        ax.set_title(
            f"{cell_type}\n{gene}", fontsize=9, weight="bold", color=palette.get(cell_type)
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for panel_index in range(len(present), len(axes)):
        axes[panel_index].set_visible(False)

    fig.suptitle("Canonical marker expression by cell type", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    paths = save_figure(fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


def plot_categorized_dotplot(
    adata: ad.AnnData,
    gene_categories: Mapping[str, list[str]],
    *,
    groupby: str,
    layer: str,
    out_path: Path,
    categories_order: list[str] | None = None,
    vmax_percentile: float = 95.0,
    title: str | None = None,
) -> Path | None:
    """
    A fully custom, densely categorized marker dotplot: genes as rows grouped
    into named panels with a boxed label to their left, groupby categories as
    columns. Dot size = % of cells expressing; color = mean expression
    (capped at ``vmax_percentile`` so a handful of outlier cells cannot wash
    out the whole scale).

    Unlike ``plot_marker_dotplot`` (a thin scanpy.pl.dotplot wrapper), this
    draws everything directly for the specific layout a per-cluster state
    profile calls for: alternating row-band shading per category, a left-hand
    boxed category label spanning that category's own genes, and generous
    figure spacing so dense panels (dozens of genes, several categories)
    still read cleanly.

    Args:
        adata: Object carrying ``layer`` and ``groupby``.
        gene_categories: Category name -> ordered gene list (e.g.
            ``{"Pan-lymphatic": ["PROX1", "PDPN", ...], ...}``). Genes absent
            from ``adata.var_names`` are dropped; a category left with none
            is dropped entirely.
        groupby: obs column defining the dotplot's columns (e.g. a cluster or
            state label).
        layer: Expression layer to read (log-normalized).
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
        categories_order: Column order for ``groupby``'s categories. None
            keeps the column's own category order.
        vmax_percentile: Percentile of all present-gene mean-expression
            values used as the color scale's upper bound.
        title: Optional figure title.

    Returns:
        The PNG path written, or ``None`` when no gene in any category
        survives the intersection with ``adata.var_names``.
    """
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    from matplotlib.patches import FancyBboxPatch

    present_categories = {
        cat: [g for g in genes if g in adata.var_names] for cat, genes in gene_categories.items()
    }
    present_categories = {cat: genes for cat, genes in present_categories.items() if genes}
    if not present_categories:
        return None

    apply_theme("notebook")
    out_path = Path(out_path)

    groups = (
        [c for c in categories_order if c in adata.obs[groupby].astype(str).unique()]
        if categories_order is not None
        else list(pd.Categorical(adata.obs[groupby]).categories.astype(str))
    )
    if not groups:
        groups = sorted(adata.obs[groupby].astype(str).unique())

    # A gene may legitimately repeat across categories (e.g. COL1A1 in both a
    # "Homeostatic/ECM" and a "Myofibroblast/contractile" panel -- exactly the
    # case Mehrara's own table has). That means all_genes can have duplicate
    # names, so everything below is indexed by ROW POSITION, never by gene
    # name via .loc -- a duplicated label there returns a Series, not a
    # scalar, and silently breaks the color/size lookup per cell.
    all_genes = [g for genes in present_categories.values() for g in genes]
    var_positions = adata.var_names.get_indexer(all_genes)
    matrix = adata.layers[layer][:, var_positions]
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    group_labels = adata.obs[groupby].astype(str).to_numpy()

    n_rows = len(all_genes)
    mean_arr = np.full((n_rows, len(groups)), np.nan)
    pct_arr = np.full((n_rows, len(groups)), np.nan)
    for col_idx, group in enumerate(groups):
        mask = group_labels == group
        sub = matrix[mask]
        mean_arr[:, col_idx] = sub.mean(axis=0)
        pct_arr[:, col_idx] = (sub > 0).mean(axis=0) * 100.0

    vmax = float(np.nanpercentile(mean_arr, vmax_percentile)) or 1.0

    n_cols = len(groups)
    row_h = 0.34
    fig_h = max(3.0, n_rows * row_h + 1.6)
    fig_w = max(4.5, 1.9 + n_cols * 0.62 + 1.6)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.7, n_cols * 0.62, 1.1], wspace=0.06, left=0.02, right=0.98
    )
    box_ax = fig.add_subplot(gs[0, 0])
    dot_ax = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[0, 2])

    # Row position by index, not gene name: a name can repeat across
    # categories, so a name-keyed dict would silently collapse duplicates
    # onto one row.
    row_y = [n_rows - 1 - i for i in range(n_rows)]

    # Alternating row-band shading per category, and the boxed category label.
    row_cursor = 0
    for band_index, (category, genes) in enumerate(present_categories.items()):
        y_top = n_rows - 1 - row_cursor + 0.5
        y_bottom = n_rows - 1 - (row_cursor + len(genes) - 1) - 0.5
        if band_index % 2 == 0:
            dot_ax.axhspan(y_bottom, y_top, color="#F2F2F2", zorder=0)
        box_ax.add_patch(
            FancyBboxPatch(
                (0.04, y_bottom),
                0.92,
                y_top - y_bottom,
                boxstyle="round,pad=0.0,rounding_size=0.06",
                linewidth=0.9,
                edgecolor="#888888",
                facecolor="white",
                clip_on=False,
            )
        )
        box_ax.text(
            0.5,
            (y_top + y_bottom) / 2,
            category,
            ha="center",
            va="center",
            fontsize=8.5,
            wrap=True,
        )
        row_cursor += len(genes)

    box_ax.set_xlim(0, 1)
    box_ax.set_ylim(-0.5, n_rows - 0.5)
    box_ax.axis("off")

    # Dots.
    cmap = plt.get_cmap("Reds")
    for row_idx in range(n_rows):
        y = row_y[row_idx]
        for col_idx in range(n_cols):
            pct = pct_arr[row_idx, col_idx]
            val = mean_arr[row_idx, col_idx]
            size = 6.0 + (pct / 100.0) * 110.0
            color = cmap(min(val / vmax, 1.0)) if np.isfinite(val) else "#DDDDDD"
            dot_ax.scatter(
                col_idx, y, s=size, color=color, edgecolor="#555555", linewidth=0.3, zorder=3
            )

    dot_ax.set_xlim(-0.6, n_cols - 0.4)
    dot_ax.set_ylim(-0.5, n_rows - 0.5)
    dot_ax.set_xticks(range(n_cols))
    dot_ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8.5)
    dot_ax.set_yticks(row_y)
    dot_ax.set_yticklabels(all_genes, style="italic", fontsize=8.5)
    dot_ax.tick_params(axis="both", length=0)
    for spine in dot_ax.spines.values():
        spine.set_visible(False)

    # Legends: color scale + size scale, stacked in the third column.
    legend_ax.axis("off")
    cbar_ax = legend_ax.inset_axes((0.05, 0.55, 0.35, 0.4))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        f"avg. expression\n(capped at {vmax_percentile:g}th pct)", fontsize=7.5, labelpad=6
    )
    cbar.ax.tick_params(labelsize=7)

    size_ax = legend_ax.inset_axes((0.05, 0.05, 0.9, 0.4))
    size_ax.axis("off")
    for i, pct in enumerate([0, 25, 50, 75, 100]):
        yy = 4 - i
        size_ax.scatter(
            [0.15], [yy], s=6.0 + (pct / 100.0) * 110.0, color="#888888", edgecolor="#555555"
        )
        size_ax.text(0.45, yy, f"{pct}", va="center", fontsize=7.5)
    size_ax.text(0.15, 5.3, "% cells\nexpressing", fontsize=7.5, ha="left")
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(-1, 6)

    if title:
        fig.suptitle(title, fontsize=12, weight="bold", y=1.0)

    paths = save_figure(fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


@dataclass(frozen=True)
class MarkerPanelStatistics:
    """Per-group summary of a marker panel: what the lineage dotplot draws.

    ``groups`` are the rows (every observed group); ``genes`` the columns, in panel order, and a
    gene may repeat when it belongs to two panel groups; ``gene_group`` names the panel group each
    column belongs to. ``mean`` is the mean of the expression layer and ``pct`` the percent of
    cells with a nonzero value, both shaped (n_groups, n_genes).
    """

    groups: list[str]
    genes: list[str]
    gene_group: list[str]
    mean: np.ndarray
    pct: np.ndarray


def marker_panel_statistics(
    adata: ad.AnnData, marker_panel: Mapping[str, list[str]], *, groupby: str, layer: str
) -> MarkerPanelStatistics:
    """Mean expression and percent detected per group for the genes of a marker panel.

    A panel group that has no cells in ``groupby`` is dropped, and so is any gene that is absent
    from the object. Columns are addressed by position, so a gene named in two panel groups
    yields two columns.
    """
    import scipy.sparse as sp

    labels = adata.obs[groupby].astype(str).to_numpy()
    observed = pd.Series(labels).unique().tolist()
    column = adata.obs[groupby]
    groups = (
        [str(c) for c in column.cat.categories if str(c) in set(observed)]
        if hasattr(column, "cat")
        else sorted(observed)
    )
    genes: list[str] = []
    gene_group: list[str] = []
    positions: list[int] = []
    for group, panel_genes in marker_panel.items():
        if str(group) not in set(groups):
            continue
        for gene in panel_genes:
            if gene in adata.var_names:
                genes.append(gene)
                gene_group.append(str(group))
                positions.append(int(adata.var_names.get_loc(gene)))
    if not genes:
        empty = np.zeros((len(groups), 0))
        return MarkerPanelStatistics(groups, [], [], empty, empty)
    matrix = adata.layers[layer][:, positions]
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    mean = np.vstack([matrix[labels == g].mean(axis=0) for g in groups])
    pct = np.vstack([(matrix[labels == g] > 0).mean(axis=0) * 100.0 for g in groups])
    return MarkerPanelStatistics(groups, genes, gene_group, mean, pct)


def scale_columns_unit(mean: np.ndarray) -> np.ndarray:
    """Scale each column (gene) to [0, 1] across groups; a constant column becomes 0, never NaN."""
    lo = mean.min(axis=0)
    span = mean.max(axis=0) - lo
    return np.where(span > 0, (mean - lo) / np.where(span > 0, span, 1.0), 0.0)


def dendrogram_row_order(scaled: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Order the rows of ``scaled`` so similar groups sit together, and return the linkage matrix.

    Rows are clustered on Pearson correlation of their expression profiles (complete linkage,
    optimal leaf ordering), so two groups with the same shape of expression stay together
    whatever their magnitude. A constant row has no defined correlation; if any distance is not
    finite the tree falls back to Euclidean distance rather than failing.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    if scaled.shape[0] < 2:
        return list(range(scaled.shape[0])), np.zeros((0, 4))
    distance = pdist(scaled, metric="correlation")
    if not np.isfinite(distance).all():
        distance = pdist(scaled, metric="euclidean")
    z = linkage(distance, method="complete", optimal_ordering=True)
    return [int(i) for i in leaves_list(z)], z


def draw_lineage_marker_dotplot(
    stats: MarkerPanelStatistics,
    *,
    palette: Mapping[str, str] | None = None,
    panel_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    size_legend: tuple[int, ...] = (25, 50, 100),
) -> Figure:
    """Draw the lineage marker dotplot: groups as rows ordered by a dendrogram, genes as columns.

    A coloured header bar over each panel group's genes carries its label; dot area is the percent
    of cells expressing and colour the mean scaled per gene across groups. The dendrogram, the
    row-label column and the dot grid are separate axes, so labels never sit on the tree.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    apply_theme("notebook")
    palette = dict(palette) if palette is not None else cell_type_palette(stats.groups)
    panel_labels = dict(panel_labels or {})
    n_rows, n_cols = len(stats.groups), len(stats.genes)
    scaled = scale_columns_unit(stats.mean)
    order, linkage_matrix = dendrogram_row_order(scaled)
    y_of = {row: i for i, row in enumerate(order)}  # leaf 0 at the bottom, matching the dendrogram

    # Laid out in inches so each piece is the size it needs: a label column just wide enough for
    # the longest group name (the tree sits against it), a slim colorbar, and room for every gene
    # column. Rotated gene labels plus the size legend need height that scales with the longest
    # name, so the figure grows to hold it (a fixed fraction overlapped).
    gene_fontsize = 9
    tree_in, bar_in, gap_in = 0.9, 0.2, 0.08
    label_in = 0.085 * max((len(g) for g in stats.groups), default=1) + 0.05
    dots_in = 0.30 * n_cols
    axes_in = tree_in + label_in + dots_in + bar_in
    left_in, right_in = 0.15, 0.95  # the right margin holds the colorbar ticks and its label
    fig_w = max(8.0, axes_in + 3 * gap_in + left_in + right_in)
    bottom_in = 0.35 + 0.0085 * gene_fontsize * max((len(g) for g in stats.genes), default=0) + 0.6
    fig_h = 1.0 + 0.42 * n_rows + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[tree_in, label_in, dots_in, bar_in],
        height_ratios=[0.06, 1.0],
        hspace=0.04,
        wspace=gap_in / (axes_in / 4),
        left=left_in / fig_w,
        right=1 - right_in / fig_w,
        top=0.9,
        bottom=bottom_in / fig_h,
    )
    ax_tree = fig.add_subplot(gs[1, 0], label="dendrogram")
    ax_lab = fig.add_subplot(gs[1, 1], label="row_labels")
    ax_dots = fig.add_subplot(gs[1, 2], label="dots")
    ax_head = fig.add_subplot(gs[0, 2], label="header")
    ax_bar = fig.add_subplot(gs[1, 3], label="colorbar")

    if n_rows > 1:
        dendrogram(
            linkage_matrix,
            orientation="left",
            ax=ax_tree,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#7a97b8",
        )
        ax_tree.set_ylim(0, 10 * n_rows)  # leaf i sits at y = 10 * i + 5, i.e. the dot row i
    ax_tree.axis("off")
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, n_rows - 0.5)
    ax_lab.axis("off")
    for row, group in enumerate(stats.groups):
        ax_lab.text(
            1.0,
            y_of[row],
            group,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=palette.get(group, "#333333"),
        )

    rows_idx = np.repeat([y_of[r] for r in range(n_rows)], n_cols)
    cols_idx = np.tile(np.arange(n_cols), n_rows)
    ax_dots.scatter(
        cols_idx,
        rows_idx,
        s=8 + 3.0 * stats.pct.ravel(),
        c=scaled.ravel(),
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolors="none",
    )
    ax_dots.set_xticks(range(n_cols))
    ax_dots.set_xticklabels(stats.genes, rotation=90, fontsize=gene_fontsize)
    ax_dots.set_yticks([])
    ax_dots.set_xlim(-0.6, n_cols - 0.4)
    ax_dots.set_ylim(-0.5, n_rows - 0.5)
    ax_dots.set_axisbelow(True)
    ax_dots.grid(axis="y", color="#eeeeee", linewidth=0.6)
    for spine in ax_dots.spines.values():
        spine.set_visible(False)

    ax_head.set_xlim(-0.6, n_cols - 0.4)
    ax_head.set_ylim(0, 1.7)
    ax_head.axis("off")
    start = 0
    while start < n_cols:
        group = stats.gene_group[start]
        end = start
        while end + 1 < n_cols and stats.gene_group[end + 1] == group:
            end += 1
        color = palette.get(group, "#333333")
        ax_head.add_patch(
            plt.Rectangle(
                (start - 0.5, 0),
                end - start + 1,
                0.6,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
        )
        ax_head.text(
            (start + end) / 2,
            1.05,
            panel_labels.get(group, group),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color,
        )
        start = end + 1

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)), cax=ax_bar
    )
    colorbar.set_label("Scaled mean expression", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    handles = [plt.scatter([], [], s=8 + 3.0 * v, c="0.6", edgecolors="none") for v in size_legend]
    fig.legend(
        handles,
        [f"{v}%" for v in size_legend],
        title="Cells detected (%)",
        loc="lower center",
        ncol=len(size_legend),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    if title:
        fig.suptitle(title, fontsize=12, weight="bold")
    return fig


def plot_lineage_marker_dotplot(
    adata: ad.AnnData,
    marker_panel: Mapping[str, list[str]],
    *,
    groupby: str,
    layer: str,
    out_path: Path,
    palette: Mapping[str, str] | None = None,
    panel_labels: Mapping[str, str] | None = None,
    title: str | None = None,
) -> Path | None:
    """Write the lineage marker dotplot (rows ordered by a dendrogram) as PNG plus PDF.

    Args:
        adata: Object carrying ``layer`` (log-normalized) and the ``groupby`` obs column. For a
            large atlas pass a small object holding only the panel genes.
        marker_panel: Group -> ordered marker genes. Groups without cells and genes absent from
            the object are dropped.
        groupby: obs column whose categories are the rows.
        layer: Expression layer to read.
        out_path: PNG path; a ``.pdf`` is written beside it.
        palette: Optional {group: colour} for row labels and header bars; defaults to the
            project palette.
        panel_labels: Optional {group: short header label}, e.g. {"Keratinocytes": "KC"}.
        title: Optional figure title.

    Returns:
        The PNG path, or ``None`` when no panel gene survives the intersection with the object.
    """
    stats = marker_panel_statistics(adata, marker_panel, groupby=groupby, layer=layer)
    if not stats.genes:
        return None
    fig = draw_lineage_marker_dotplot(
        stats, palette=palette, panel_labels=panel_labels, title=title
    )
    out_path = Path(out_path)
    paths = save_figure(fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


__all__ = [
    "MarkerPanelStatistics",
    "dendrogram_row_order",
    "draw_lineage_marker_dotplot",
    "marker_panel_statistics",
    "plot_categorized_dotplot",
    "plot_lineage_marker_dotplot",
    "plot_marker_dotplot",
    "plot_marker_feature_grid",
    "scale_columns_unit",
]
