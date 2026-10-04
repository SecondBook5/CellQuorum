"""Quantitative companion to state_score_umap: does a UCell/AUCell program
score actually differ between conditions, per cluster -- with a number
behind the claim, not just a color impression on a UMAP.

The significance marks are donor-level: cells from one donor are not independent replicates, so a
cell-level test reaches ``***`` on any cluster with a few thousand cells however little the donors
agree. The test is the project's donor-median comparison, BH-adjusted across the clusters drawn.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cellquorum.stats.donor_comparison import two_group_test_on_donor_medians
from cellquorum.stats.multiple_testing import bh_fdr
from cellquorum.visualization.figstyle import empty_figure, raincloud


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _donor_adjusted_p(
    df: pd.DataFrame,
    clusters: list[str],
    cluster_col: str,
    condition_col: str,
    score_col: str,
    case: str,
    control: str,
    donor_col: str | None,
    paired: bool,
) -> dict[str, float]:
    """BH-adjusted donor-level p per cluster; empty when there is no donor column to test on."""
    if donor_col is None or donor_col not in df.columns:
        return {}
    raw: dict[str, float] = {}
    for cluster in clusters:
        sub = df[df[cluster_col].astype(str) == cluster]
        result = two_group_test_on_donor_medians(
            sub,
            value_col=score_col,
            group_col=condition_col,
            donor_col=donor_col,
            group1=control,
            group2=case,
            paired=paired,
            incomplete_pairs="drop",
        )
        raw[cluster] = result.p_value if result is not None else float("nan")
    adjusted = bh_fdr(list(raw.values()))
    return dict(zip(raw, adjusted, strict=True))


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
    donor_col: str | None = None,
    paired: bool = True,
) -> Figure:
    """One column per cluster: left half-violin = control, right = case,
    donor-level significance marks above each pair.

    The marks come from a donor-median comparison (Wilcoxon signed-rank when ``paired``),
    BH-adjusted across the clusters drawn. Without ``donor_col`` no mark is drawn: a cell-level
    p-value is not evidence about donors and is never substituted for one.

    Complements state_score_umap's spatial view (WHERE a score is elevated)
    with the number a paper claim actually needs (IS it elevated, and by how
    much evidence): the UMAP alone gives a color impression, not a test.
    """
    if df is None or df.empty:
        return empty_figure()
    clusters = sorted(df[cluster_col].astype(str).unique())
    if not clusters:
        return empty_figure()

    adjusted = _donor_adjusted_p(
        df, clusters, cluster_col, condition_col, score_col, case, control, donor_col, paired
    )

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
        q = adjusted.get(cluster)
        if q is not None and np.isfinite(q):
            ax.text(
                i,
                local_top,
                _stars(q),
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
    if donor_col is not None:
        n_donors = df[donor_col].nunique()
        ax.set_title(
            f"Marks: donor-level {'paired ' if paired else ''}test ({n_donors} donors), "
            "BH across clusters",
            fontsize=8,
            color="0.4",
        )
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
