"""Two QC scatter figures reading the same per-cell assignment table:
1. % of a metacell's cells that MetaCell2 routed to "Outliers" vs the metacell's
   own median cell size -- the reference figure's key diagnostic (a metacell
   whose neighbors are mostly outliers is itself suspect).
2. median size of outlier cells vs median size of classified cells, per metacell
   -- distinguishes "outliers are just small/low-quality cells" (points near the
   diagonal's lower-left) from "outliers are a distinct population" (points far
   off-diagonal).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cellquorum.visualization import figstyle

_OUTLIER_LABEL = "Outliers"


def _color_by_group(nodes: pd.DataFrame) -> dict[str, str]:
    groups = sorted(nodes["dominant_group"].astype(str).unique())
    palette = (
        figstyle.MUTED_PALETTE
        if len(groups) <= len(figstyle.MUTED_PALETTE)
        else figstyle.distinct_palette(len(groups))
    )
    return {g: palette[i % len(palette)] for i, g in enumerate(groups)}


def plot_metacell_qc_unclassified_vs_size(
    assignments: pd.DataFrame, nodes: pd.DataFrame, out_dir: str | Path, *, tag: str = ""
) -> list[Path]:
    """Render: % outlier-adjacent cells vs median metacell size, colored by type.

    "% outlier" here means: of the cells whose nearest metacell (by group label,
    since Outliers carries no metacell_id of its own) is this one, what fraction
    landed in Outliers instead. Computed per dominant_group rather than per
    individual metacell_id, since Outliers cells have no metacell_id to join on.

    Args:
        assignments: Per-cell rows with metacell_id, group, n_umis.
        nodes: Columns metacell_id, dominant_group, n_cells.
        out_dir: Output directory.
        tag: Optional filename tag.

    Returns:
        Written PNG+PDF paths, or [] when there is nothing to plot.
    """
    figstyle.apply_theme("compact")
    if assignments is None or assignments.empty or nodes is None or nodes.empty:
        return []
    out_dir = Path(out_dir)

    real = assignments[assignments["metacell_id"] != _OUTLIER_LABEL]
    outliers = assignments[assignments["metacell_id"] == _OUTLIER_LABEL]

    by_mc_size = real.groupby("metacell_id")["n_umis"].median()
    outlier_pct_by_group = (
        outliers.groupby("group").size() / assignments.groupby("group").size() * 100
    ).fillna(0.0)

    color_by_group = _color_by_group(nodes)
    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE["single"])
    for _, row in nodes.iterrows():
        mc = str(row["metacell_id"])
        group = str(row["dominant_group"])
        if mc not in by_mc_size.index:
            continue
        ax.scatter(
            by_mc_size[mc],
            outlier_pct_by_group.get(group, 0.0),
            color=color_by_group.get(group, "#888888"),
            edgecolor="#222",
            linewidth=0.4,
            s=40,
        )

    ax.set_xlabel("Median metacell size (UMIs)", fontsize=9)
    ax.set_ylabel("% of same-group cells unclassified", fontsize=9)
    ax.set_title("Metacell QC: unclassified vs. size", fontsize=11, fontweight="bold")
    fig.tight_layout()
    stem = f"metacell_qc_unclassified_vs_size_{tag}" if tag else "metacell_qc_unclassified_vs_size"
    return figstyle.save_figure(fig, out_dir, stem)


def plot_metacell_qc_size_comparison(
    assignments: pd.DataFrame, nodes: pd.DataFrame, out_dir: str | Path, *, tag: str = ""
) -> list[Path]:
    """Render: median outlier-cell size vs median classified-cell size, per group.

    Args:
        assignments: Per-cell rows with metacell_id, group, n_umis.
        nodes: Columns metacell_id, dominant_group, n_cells.
        out_dir: Output directory.
        tag: Optional filename tag.

    Returns:
        Written PNG+PDF paths, or [] when there are no outlier cells to compare.
    """
    figstyle.apply_theme("compact")
    if assignments is None or assignments.empty:
        return []
    outliers = assignments[assignments["metacell_id"] == _OUTLIER_LABEL]
    if outliers.empty:
        return []
    out_dir = Path(out_dir)

    real = assignments[assignments["metacell_id"] != _OUTLIER_LABEL]
    classified_size_by_group = real.groupby("group")["n_umis"].median()
    outlier_size_by_group = outliers.groupby("group")["n_umis"].median()

    color_by_group = _color_by_group(nodes)
    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE["single"])
    common_groups = sorted(set(classified_size_by_group.index) & set(outlier_size_by_group.index))
    for group in common_groups:
        ax.scatter(
            classified_size_by_group[group],
            outlier_size_by_group[group],
            color=color_by_group.get(group, "#888888"),
            edgecolor="#222",
            linewidth=0.4,
            s=40,
            label=group,
        )
    if common_groups:
        lo = min(classified_size_by_group.min(), outlier_size_by_group.min())
        hi = max(classified_size_by_group.max(), outlier_size_by_group.max())
        ax.plot([lo, hi], [lo, hi], color="#999999", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xlabel("Median classified-cell size (UMIs)", fontsize=9)
    ax.set_ylabel("Median outlier-cell size (UMIs)", fontsize=9)
    ax.set_title("Metacell QC: outlier vs. classified size", fontsize=11, fontweight="bold")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=7.5, frameon=True
    )
    fig.tight_layout()
    stem = f"metacell_qc_size_comparison_{tag}" if tag else "metacell_qc_size_comparison"
    return figstyle.save_figure(fig, out_dir, stem)


__all__ = ["plot_metacell_qc_size_comparison", "plot_metacell_qc_unclassified_vs_size"]
