"""Figures for reference-mapping identity-confidence diagnostics.

One plot per file, not a combined grid -- a standing preference over this
pipeline's default multi-panel figures. Reuses the embeddings stage's own
categorical/continuous overlay primitives rather than reinventing scatter
plotting; this module only orchestrates which columns get drawn and where
the files land.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from cellquorum.stages.integration.embeddings.plots import (
    apply_theme,
    categorical_embedding,
    continuous_overlay,
)
from cellquorum.visualization.figstyle import cell_type_palette, save_figure

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd


def plot_identity_diagnostic_embeddings(
    adata: ad.AnnData,
    *,
    basis: str,
    cell_type_col: str,
    top_prob_col: str,
    doublet_col: str,
    n_genes_col: str,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Write the identity-confidence diagnostic panels the available columns support.

    Args:
        adata: Object carrying ``obsm[basis]`` and the requested obs columns.
        basis: obsm key of the 2-D embedding to plot on (e.g. ``"X_umap"``).
        cell_type_col: obs column of each cell's assigned type (categorical panel).
        top_prob_col: obs column of each cell's top reference-class probability
            (e.g. from ``cellquorum.stats.reference_identity.top_reference_probabilities``).
        doublet_col: obs column of a doublet score.
        n_genes_col: obs column of a complexity metric (genes detected).
        out_dir: Directory to write figures into (created if needed).

    Returns:
        Mapping of panel name to the PNG path written, for whichever panels
        had the data to support them. A missing column is skipped, not an
        error -- this is a diagnostic tool run against whatever the object
        happens to carry, not a pipeline stage with a fixed contract.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if basis not in adata.obsm:
        return written

    apply_theme()

    if cell_type_col in adata.obs.columns:
        fig = categorical_embedding(
            adata,
            cell_type_col,
            basis=basis,
            axis_labels=(f"{basis}1", f"{basis}2"),
            title="Cell type (assigned)",
        )
        paths = save_figure(fig, out_dir, "identity_diagnostic_cell_type")
        written["cell_type"] = next(p for p in paths if p.suffix == ".png")

    coords = np.asarray(adata.obsm[basis])
    continuous_panels = [
        ("top_refprob", top_prob_col, "Max reference probability (low = ambiguous identity)"),
        ("doublet_score", doublet_col, "Doublet score"),
        ("n_genes", n_genes_col, "n_genes (complexity)"),
    ]
    for key, col, title in continuous_panels:
        if col not in adata.obs.columns:
            continue
        fig = continuous_overlay(
            coords,
            adata.obs[col].to_numpy(),
            title=title,
            axis_labels=(f"{basis}1", f"{basis}2"),
        )
        paths = save_figure(fig, out_dir, f"identity_diagnostic_{key}")
        written[key] = next(p for p in paths if p.suffix == ".png")

    return written


def plot_lineage_confidence_bar(
    entropy: pd.Series,
    cell_type: pd.Series,
    out_path: Path,
) -> Path:
    """
    Plot mean reference-mapping label entropy per cell type, most confident first.

    Args:
        entropy: Per-cell kNN label entropy (e.g. ``cell_type_granular_knn_entropy``;
            low = confident, high = ambiguous), indexed like ``cell_type``.
        cell_type: Per-cell assigned type, same index as ``entropy``.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.

    Returns:
        The PNG path written.
    """
    apply_theme()
    out_path = Path(out_path)

    mean_entropy = entropy.groupby(cell_type.to_numpy()).mean().sort_values()
    order = [str(g) for g in mean_entropy.index]
    palette = cell_type_palette(order)
    colors = [palette.get(g, "#4A90E2") for g in order]

    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.35 * len(order) + 1.5)))
    ax.barh(range(len(order)), mean_entropy.to_numpy(), color=colors, alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Mean kNN label entropy (low = confident)", fontsize=10)
    ax.set_title("Per-cell-type annotation confidence", fontsize=11, weight="bold")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    paths = save_figure(fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


__all__ = ["plot_identity_diagnostic_embeddings", "plot_lineage_confidence_bar"]
