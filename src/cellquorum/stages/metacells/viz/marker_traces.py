"""Per-gene expression-across-metacells trace figures. Diagnostic and on-demand:
only rendered for genes an operator explicitly names in MetacellsConfig.marker_genes
(spec section 4.4 item 6) -- unlike the other 5 figures, this one has no default
content, since "which genes matter" is a per-investigation question, not a fixed
panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cellquorum.stages.metacells.viz.heatmap import ordered_metacell_columns
from cellquorum.visualization import figstyle


def plot_metacell_marker_traces(
    expr: pd.DataFrame,
    nodes: pd.DataFrame,
    out_dir: str | Path,
    marker_genes: list[str],
    *,
    tag: str = "",
) -> list[Path]:
    """Render one expression-across-metacells trace figure per configured gene.

    Args:
        expr: Genes (rows) x metacell_id (columns) expression matrix.
        nodes: Columns metacell_id, dominant_group.
        out_dir: Output directory (files land directly here, named by gene).
        marker_genes: Gene names to render; genes absent from expr are skipped.
        tag: Optional filename suffix appended before the gene name.

    Returns:
        Written PNG+PDF paths across all genes, or [] when marker_genes is empty.
    """
    figstyle.set_style()
    if not marker_genes:
        return []
    out_dir = Path(out_dir)
    order = [c for c in ordered_metacell_columns(nodes) if c in expr.columns]

    written: list[Path] = []
    for gene in marker_genes:
        if gene not in expr.index:
            continue
        values = expr.loc[gene, order]

        fig, ax = plt.subplots(figsize=figstyle.FIGSIZE["single"])
        ax.plot(
            range(len(order)),
            values.to_numpy(),
            color=figstyle.CATEGORICAL_PALETTE[0],
            linewidth=1.2,
        )
        ax.scatter(
            range(len(order)),
            values.to_numpy(),
            color=figstyle.CATEGORICAL_PALETTE[0],
            s=12,
            zorder=3,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=5)
        ax.set_ylabel(f"{gene} expression", fontsize=9)
        ax.set_title(f"{gene} across metacells", fontsize=11, fontweight="bold")
        fig.tight_layout()
        stem = f"{gene}_{tag}" if tag else gene
        written.extend(figstyle.save_figure(fig, out_dir, stem))

    return written


__all__ = ["plot_metacell_marker_traces"]
