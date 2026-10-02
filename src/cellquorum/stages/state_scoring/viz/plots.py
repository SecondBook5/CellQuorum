"""Quantitative companion to state_score_umap: does a UCell/AUCell program
score actually differ between conditions, per cluster -- with a number
behind the claim, not just a color impression on a UMAP.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

from cellquorum.visualization.figstyle import empty_figure, raincloud


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def state_score_violin(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    condition_col: str,
    score_col: str,
    case: str,
    control: str,
    case_color: str,
    control_color: str,
    score_label: str,
    cluster_labels: dict[str, str] | None = None,
) -> Figure:
    """One column per cluster: left half-violin = control, right = case,
    Mann-Whitney U p-value annotated above each pair.

    Complements state_score_umap's spatial view (WHERE a score is elevated)
    with the number a paper claim actually needs (IS it elevated, and by how
    much evidence): the UMAP alone gives a color impression, not a test.
    """
    if df is None or df.empty:
        return empty_figure()
    clusters = sorted(df[cluster_col].astype(str).unique())
    if not clusters:
        return empty_figure()

    fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(clusters) + 1), 4.5))
    tops = []
    for i, cluster in enumerate(clusters):
        sub = df[df[cluster_col].astype(str) == cluster]
        control_vals = (
            sub.loc[sub[condition_col] == control, score_col].dropna().to_numpy(dtype=float)
        )
        case_vals = sub.loc[sub[condition_col] == case, score_col].dropna().to_numpy(dtype=float)
        if control_vals.size:
            raincloud(ax, control_vals, float(i), color=control_color, side="left", seed=i)
        if case_vals.size:
            raincloud(ax, case_vals, float(i), color=case_color, side="right", seed=i + 1000)

        local_top = max(
            [np.percentile(v, 99) for v in (control_vals, case_vals) if v.size], default=0.0
        )
        tops.append(local_top)
        if control_vals.size and case_vals.size:
            _, p = mannwhitneyu(control_vals, case_vals, alternative="two-sided")
            ax.text(
                i,
                local_top,
                _stars(p),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Headroom for the p-value annotations above the tallest distribution.
    if tops:
        span = max(tops) - min(min(tops), 0.0) or 1.0
        ax.set_ylim(top=max(tops) + 0.12 * span)

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(
        [(cluster_labels or {}).get(c, c) for c in clusters], rotation=45, ha="right"
    )
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    ax.set_ylabel(score_label)
    ax.legend(
        handles=[
            Line2D([0], [0], color=control_color, lw=6, label=control),
            Line2D([0], [0], color=case_color, lw=6, label=case),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.tight_layout()
    return fig


__all__ = ["state_score_violin"]
