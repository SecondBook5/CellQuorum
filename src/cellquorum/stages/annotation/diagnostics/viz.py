"""Figures for scDiagnostics annotation-confidence diagnostics.

One plot per file, not a combined grid -- a standing preference over the
multi-panel figures this pipeline otherwise defaults to.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from cellquorum.visualization.figstyle import apply_theme, cell_type_palette, save_cellquorum_figure

matplotlib.use("Agg")


def _ordered_by_median(obs: pd.DataFrame, value_col: str, group_col: str) -> list[str]:
    """Return group labels ordered by ascending group median of ``value_col``."""

    medians = obs.groupby(group_col, observed=True)[value_col].median().sort_values()
    return [str(g) for g in medians.index]


def plot_anomaly_by_cell_type(obs: pd.DataFrame, cell_type_col: str, out_path: Path) -> Path | None:
    """
    Plot per-cell isolation-forest anomaly scores, grouped by cell type.

    Args:
        obs: Obs DataFrame carrying ``scdiag_anomaly`` plus ``cell_type_col``.
        cell_type_col: obs column to group by.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.

    Returns:
        ``out_path`` on success, or ``None`` when ``scdiag_anomaly`` is absent
        (the reference-based diagnostics were not run).
    """
    if "scdiag_anomaly" not in obs.columns:
        return None

    apply_theme("notebook")
    data = obs[[cell_type_col, "scdiag_anomaly"]].dropna(subset=["scdiag_anomaly"])
    order = _ordered_by_median(data, "scdiag_anomaly", cell_type_col)
    palette = cell_type_palette(order)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(order) + 2.0), 4.5))
    groups = [data.loc[data[cell_type_col] == g, "scdiag_anomaly"].to_numpy() for g in order]
    parts = ax.boxplot(groups, tick_labels=order, showfliers=False, patch_artist=True, widths=0.6)
    for patch, group in zip(parts["boxes"], order, strict=True):
        patch.set_facecolor(palette.get(group, "#4A90E2"))
        patch.set_alpha(0.75)

    ax.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.0, alpha=0.7, label="threshold")
    ax.set_ylabel("Anomaly score", fontsize=10)
    ax.set_title("Annotation anomaly by cell type (scDiagnostics)", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    save_cellquorum_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_entropy_by_cell_type(obs: pd.DataFrame, cell_type_col: str, out_path: Path) -> Path | None:
    """
    Plot per-cell annotation (categorization) entropy, grouped by cell type.

    Args:
        obs: Obs DataFrame carrying ``scdiag_entropy`` plus ``cell_type_col``.
        cell_type_col: obs column to group by.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.

    Returns:
        ``out_path`` on success, or ``None`` when ``scdiag_entropy`` is absent
        (no ``soft_scores_obsm`` was configured).
    """
    if "scdiag_entropy" not in obs.columns:
        return None

    apply_theme("notebook")
    data = obs[[cell_type_col, "scdiag_entropy"]].dropna(subset=["scdiag_entropy"])
    order = _ordered_by_median(data, "scdiag_entropy", cell_type_col)
    palette = cell_type_palette(order)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(order) + 2.0), 4.5))
    groups = [data.loc[data[cell_type_col] == g, "scdiag_entropy"].to_numpy() for g in order]
    parts = ax.boxplot(groups, tick_labels=order, showfliers=False, patch_artist=True, widths=0.6)
    for patch, group in zip(parts["boxes"], order, strict=True):
        patch.set_facecolor(palette.get(group, "#50C878"))
        patch.set_alpha(0.75)

    ax.set_ylabel("Shannon entropy (bits)", fontsize=10)
    ax.set_title("Label uncertainty by cell type (scDiagnostics)", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    save_cellquorum_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_knn_confidence(obs: pd.DataFrame, cell_type_col: str, out_path: Path) -> Path | None:
    """
    Plot per-cell-type kNN confidence against the reference atlas.

    ``scdiag_knn_prob_group`` is a population-level statistic (one value per
    cell type, broadcast to its member cells) -- this reduces it back to one
    bar per cell type, annotated with the query count it was computed from.

    Args:
        obs: Obs DataFrame carrying ``scdiag_knn_prob_group`` (and, optionally,
            ``scdiag_knn_prob_group_n``) plus ``cell_type_col``.
        cell_type_col: obs column identifying each cell type.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.

    Returns:
        ``out_path`` on success, or ``None`` when ``scdiag_knn_prob_group`` is
        absent (the reference-based diagnostics were not run).
    """
    if "scdiag_knn_prob_group" not in obs.columns:
        return None

    apply_theme("notebook")
    has_n = "scdiag_knn_prob_group_n" in obs.columns
    cols = [cell_type_col, "scdiag_knn_prob_group", *(["scdiag_knn_prob_group_n"] if has_n else [])]
    agg = {"scdiag_knn_prob_group": "first"}
    if has_n:
        agg["scdiag_knn_prob_group_n"] = "first"
    per_type = (
        obs[cols]
        .dropna(subset=["scdiag_knn_prob_group"])
        .groupby(cell_type_col, observed=True)
        .agg(agg)
        .sort_values("scdiag_knn_prob_group")
    )
    if per_type.empty:
        return None

    order = [str(g) for g in per_type.index]
    palette = cell_type_palette(order)
    colors = [palette.get(g, "#4A90E2") for g in order]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(order) + 2.0), 4.5))
    bars = ax.bar(order, per_type["scdiag_knn_prob_group"].to_numpy(), color=colors, alpha=0.85)
    if has_n:
        for bar, n in zip(bars, per_type["scdiag_knn_prob_group_n"].to_numpy(), strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={int(n):,}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("kNN probability vs. reference", fontsize=10)
    ax.set_title("Per-cell-type kNN confidence (scDiagnostics)", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    save_cellquorum_figure(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_annotation_diagnostics_figures(
    obs: pd.DataFrame,
    cell_type_col: str,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Write every annotation_diagnostics figure the available columns support.

    Args:
        obs: Obs DataFrame carrying ``scdiag_*`` columns plus ``cell_type_col``.
        cell_type_col: obs column identifying each cell's annotated type.
        out_dir: Directory to write figures into (created if needed).

    Returns:
        Mapping of figure name to the path written, for whichever figures had
        the data to support them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    anomaly_path = plot_anomaly_by_cell_type(
        obs, cell_type_col, out_dir / "scdiagnostics_anomaly.png"
    )
    if anomaly_path is not None:
        written["anomaly"] = anomaly_path

    entropy_path = plot_entropy_by_cell_type(
        obs, cell_type_col, out_dir / "scdiagnostics_entropy.png"
    )
    if entropy_path is not None:
        written["entropy"] = entropy_path

    knn_path = plot_knn_confidence(obs, cell_type_col, out_dir / "scdiagnostics_knn_confidence.png")
    if knn_path is not None:
        written["knn_confidence"] = knn_path

    return written


__all__ = [
    "plot_anomaly_by_cell_type",
    "plot_entropy_by_cell_type",
    "plot_knn_confidence",
    "write_annotation_diagnostics_figures",
]
