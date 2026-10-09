"""CellQuorum figures for unsigned gene-KO differential regulation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cellquorum.visualization import figstyle


def plot_knockout_results(
    ranking: pd.DataFrame,
    pathways: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    n_top: int,
) -> list[Path]:
    """Export native-distance rankings and an optional pathway ORA heatmap."""
    figstyle.apply_theme("publication")
    paths = []
    for group, table in ranking.groupby("group", sort=True):
        data = (
            table[~table.is_knockout_gene]
            .sort_values("distance", ascending=False)
            .head(n_top)
            .iloc[::-1]
        )
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(4.4, max(2.5, 0.2 * len(data) + 0.8)))
        y = np.arange(len(data))
        ax.hlines(y, 0, data.distance, color="#b5bdc6", lw=1)
        points = ax.scatter(
            data.distance, y, c=-np.log10(np.maximum(data.FDR, 1e-12)), cmap="viridis", s=30, vmin=0
        )
        ax.set_yticks(y, data.gene, fontsize=8)
        ax.set_xlabel("Differential network regulation (distance)", fontsize=8)
        ax.set_title(str(group), fontsize=9)
        fig.colorbar(points, ax=ax, label="−log₁₀ gene FDR", pad=0.02)
        tag = hashlib.sha256(str(group).encode()).hexdigest()[:12]
        paths.extend(
            figstyle.save_figure(fig, out_dir, f"knk_genes_{tag}", formats=tuple(formats), dpi=600)
        )
        plt.close(fig)
    if not pathways.empty:
        selected = pathways.groupby("pathway").FDR_across_cohorts.min().sort_values().head(20).index
        matrix = pathways[pathways.pathway.isin(selected)].pivot(
            index="pathway", columns="group", values="FDR_across_cohorts"
        )
        matrix = matrix.reindex(selected)
        score = -np.log10(matrix.clip(lower=1e-12))
        fig, ax = plt.subplots(
            figsize=(max(3.5, 0.65 * len(matrix.columns) + 2.5), max(2.5, 0.25 * len(matrix) + 1))
        )
        sns.heatmap(
            score,
            mask=matrix.isna(),
            cmap="viridis",
            vmin=0,
            ax=ax,
            cbar_kws={"label": "−log₁₀ pathway FDR"},
            linewidths=0.3,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8, rotation=35)
        paths.extend(
            figstyle.save_figure(
                fig, out_dir, "knk_pathway_enrichment", formats=tuple(formats), dpi=600
            )
        )
        plt.close(fig)
    return paths
