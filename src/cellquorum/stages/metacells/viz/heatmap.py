"""Gene x metacell heatmap, columns ordered by dominant cell type then metacell id
(a stable, deterministic proxy for "within-type similarity" that does not require
a second clustering pass over metacell pseudobulk profiles). `ordered_metacell_columns`
is the single shared ordering function `composition.py` also calls, so the two
figures can never drift out of the same column order (spec section 4.4 item 3).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.visualization import figstyle


def ordered_metacell_columns(nodes: pd.DataFrame) -> list[str]:
    """Return metacell_ids ordered by dominant_group, then by id within each group.

    Groups are ordered by their first appearance in ``nodes`` (not alphabetically),
    so the row order already present in ``nodes`` -- e.g. from an upstream
    clustering or size ordering -- is preserved instead of being scrambled by a
    plain alphabetical sort of `dominant_group` values.

    Args:
        nodes: Columns metacell_id, dominant_group (n_cells optional/unused here).

    Returns:
        Ordered list of metacell_id values.
    """
    group_order = nodes["dominant_group"].drop_duplicates().tolist()
    group_rank = pd.Categorical(nodes["dominant_group"], categories=group_order, ordered=True)
    ordered = nodes.assign(_group_rank=group_rank).sort_values(["_group_rank", "metacell_id"])
    return ordered["metacell_id"].astype(str).tolist()


def top_variable_genes(expr: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    """Return the `n_genes` rows (genes) of `expr` with the highest variance.

    Selects by variance across metacells (`expr.var(axis=1)`, descending) so
    the heatmap shows the genes that actually distinguish metacells from one
    another, not every gene in the object -- with e.g. 20,000 genes, plotting
    all of them produces an illegible, effectively solid-color figure.

    Args:
        expr: Genes (rows) x metacell_id (columns) expression matrix.
        n_genes: Max number of genes to keep.

    Returns:
        A row-subset of `expr`, ordered by descending variance, at most
        `n_genes` rows (fewer if `expr` has fewer genes than that).
    """
    if len(expr) <= n_genes:
        return expr
    order = expr.var(axis=1).sort_values(ascending=False).index
    return expr.loc[order[:n_genes]]


def _build_heatmap_figure(expr: pd.DataFrame, nodes: pd.DataFrame, *, n_genes: int = 50) -> Figure:
    """Draw the heatmap and return it unsaved (for geometry tests)."""

    expr = top_variable_genes(expr, n_genes)
    order = [c for c in ordered_metacell_columns(nodes) if c in expr.columns]
    ordered_expr = expr[order]

    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE["heatmap"])
    im = ax.imshow(ordered_expr.to_numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(ordered_expr.index)))
    ax.set_yticklabels(ordered_expr.index, fontsize=6)
    ax.set_title("Metacell gene expression", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="expression", shrink=0.7)
    fig.tight_layout()
    return fig


def plot_metacell_gene_heatmap(
    expr: pd.DataFrame,
    nodes: pd.DataFrame,
    out_dir: str | Path,
    *,
    n_genes: int = 50,
    tag: str = "",
) -> list[Path]:
    """Render the gene x metacell heatmap, restricted to the top variable genes.

    Args:
        expr: Genes (rows) x metacell_id (columns) expression matrix.
        nodes: Columns metacell_id, dominant_group.
        out_dir: Output directory.
        n_genes: Max genes plotted, chosen by descending variance across
            metacells (spec section 4.4 item 2: "top variable genes").
        tag: Optional filename tag.

    Returns:
        Written PNG+PDF paths, or [] when expr is empty.
    """
    figstyle.set_style()
    if expr is None or expr.empty:
        return []
    out_dir = Path(out_dir)
    fig = _build_heatmap_figure(expr, nodes, n_genes=n_genes)
    stem = f"metacell_gene_heatmap_{tag}" if tag else "metacell_gene_heatmap"
    return figstyle.save_figure(fig, out_dir, stem)


__all__ = ["ordered_metacell_columns", "plot_metacell_gene_heatmap", "top_variable_genes"]
