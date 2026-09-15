"""Marker-gene figures for annotation QA: dotplot + per-lineage feature grid.

The Clarke et al. 2021 (Nat Protocols) style manual-annotation figures: does
each cell type actually express its own canonical markers, and only its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.stages.integration.embeddings.plots import apply_theme
from cellquorum.visualization.figstyle import cell_type_palette, save_figure

if TYPE_CHECKING:
    from collections.abc import Mapping

    import anndata as ad


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

    apply_theme()
    out_path = Path(out_path)

    kwargs = {}
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

    apply_theme()
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

    apply_theme()
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
    cbar_ax = legend_ax.inset_axes([0.05, 0.55, 0.35, 0.4])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        f"avg. expression\n(capped at {vmax_percentile:g}th pct)", fontsize=7.5, labelpad=6
    )
    cbar.ax.tick_params(labelsize=7)

    size_ax = legend_ax.inset_axes([0.05, 0.05, 0.9, 0.4])
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


__all__ = ["plot_marker_dotplot", "plot_marker_feature_grid", "plot_categorized_dotplot"]
