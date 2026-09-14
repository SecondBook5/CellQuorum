"""Marker-gene figures for annotation QA: dotplot + per-lineage feature grid.

The Clarke et al. 2021 (Nat Protocols) style manual-annotation figures: does
each cell type actually express its own canonical markers, and only its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
) -> Path | None:
    """
    Plot a canonical marker-panel dotplot (dot = % detected, color = scaled mean).

    Args:
        adata: Object carrying ``layer`` and ``groupby``.
        marker_panel: Cell type -> ordered list of canonical marker genes.
            Genes absent from ``adata.var_names`` are dropped; a cell type left
            with none is dropped entirely rather than drawing an empty column.
        groupby: obs column to group cells by (rows of the dotplot).
        layer: Expression layer to read (log-normalized).
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
        categories_order: Row order for ``groupby``'s categories. None keeps
            the column's own category order.

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


__all__ = ["plot_marker_dotplot", "plot_marker_feature_grid"]
