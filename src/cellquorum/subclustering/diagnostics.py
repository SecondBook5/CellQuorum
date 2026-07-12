"""Subclustering diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from cellquorum.visualization.style import (
    CELLQUORUM_BLUE,
    CELLQUORUM_RED,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    save_cellquorum_figure,
)

# Use Agg backend for headless plotting.
matplotlib.use("Agg")


def plot_group_recovery(
    counts: dict[str, int],
    min_cells: int,
    group_key: str,
    out_path: Path | str,
) -> Path:
    """
    Plot group-level cell counts with threshold line.

    This shows how many cells per group survived the group_filter,
    with kept/dropped groups colored by threshold.

    Args:
        counts: {group: cell_count} dict.
        min_cells: minimum cells threshold.
        group_key: obs column name for groups.
        out_path: output file path.

    Returns:
        Path to saved figure.
    """
    # Guard: skip if counts empty.
    if not counts:
        return Path(out_path)

    # Apply CellQuorum theme.
    apply_cellquorum_theme()

    # Sort groups by count (descending).
    sorted_groups = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    groups = [g for g, _ in sorted_groups]
    values = [c for _, c in sorted_groups]

    # Assign colors (kept = blue, dropped = red).
    colors = [CELLQUORUM_BLUE if v >= min_cells else CELLQUORUM_RED for v in values]

    # Create figure.
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars.
    x_pos = range(len(groups))
    ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor="none")

    # Plot threshold line.
    ax.axhline(
        min_cells,
        color=CELLQUORUM_RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {min_cells}",
    )

    # Axis labels.
    ax.set_xlabel(f"{group_key}", fontsize=11)
    ax.set_ylabel("Cell count", fontsize=11)
    ax.set_title(
        f"Group recovery: {group_key}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, rotation=45, ha="right")

    # Legend.
    ax.legend(loc="upper right", frameon=False)

    # Apply CellQuorum axis style.
    apply_cellquorum_axis_style(ax)

    # Save figure.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cellquorum_figure(fig, out_path)
    plt.close(fig)

    return out_path


def plot_stability_curve(
    per_resolution_stability: dict[str, float],
    out_path: Path | str,
) -> Path:
    """
    Plot stability metric vs. resolution for partition grid search.

    This shows how cluster stability changes across resolutions, helping
    identify the optimal resolution parameter.

    Args:
        per_resolution_stability: {resolution: stability_score} dict.
        out_path: output file path.

    Returns:
        Path to saved figure.
    """
    # Guard: skip if empty.
    if not per_resolution_stability:
        return Path(out_path)

    # Apply CellQuorum theme.
    apply_cellquorum_theme()

    # Sort resolutions numerically.
    sorted_items = sorted(
        per_resolution_stability.items(),
        key=lambda x: float(x[0]) if isinstance(x[0], int | float | str) else 0,
    )
    resolutions = [r for r, _ in sorted_items]
    stability = [s for _, s in sorted_items]

    # Create figure.
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot line + markers.
    ax.plot(
        resolutions,
        stability,
        color=CELLQUORUM_BLUE,
        marker="o",
        linewidth=2,
        markersize=6,
        alpha=0.8,
    )

    # Axis labels.
    ax.set_xlabel("Resolution", fontsize=11)
    ax.set_ylabel("Stability score", fontsize=11)
    ax.set_title(
        "Cluster stability vs. resolution",
        fontsize=12,
        fontweight="bold",
    )

    # Apply CellQuorum axis style.
    apply_cellquorum_axis_style(ax)

    # Save figure.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cellquorum_figure(fig, out_path)
    plt.close(fig)

    return out_path


def plot_subcluster_qc_panel(
    gate_result: dict,
    out_path: Path | str,
) -> Path:
    """
    Plot per-cluster QC panel (donor gate diagnostics).

    This shows per-cluster bars for:
    - n_donors (number of donors contributing)
    - max_donor_frac (with threshold line)
    - lodo_stability (leave-one-donor-out stability)
    - classifier_sep (donor-blocked one-vs-rest separability)

    Clusters that PASS are blue, FAIL are red.

    Args:
        gate_result: output of donor_reproducibility().
        out_path: output file path.

    Returns:
        Path to saved figure.
    """
    # Guard: skip if no clusters.
    if not gate_result.get("clusters"):
        return Path(out_path)

    # Apply CellQuorum theme.
    apply_cellquorum_theme()

    clusters_dict = gate_result["clusters"]

    # Extract cluster IDs and metrics.
    cluster_ids = list(clusters_dict.keys())
    n_groups = [clusters_dict[cid]["n_groups"] for cid in cluster_ids]
    max_group_frac = [clusters_dict[cid]["max_group_frac"] for cid in cluster_ids]
    lodo_stability = [clusters_dict[cid].get("lodo_stability") for cid in cluster_ids]
    classifier_sep = [clusters_dict[cid].get("classifier_sep") for cid in cluster_ids]
    qc_pass = [clusters_dict[cid]["qc_pass"] for cid in cluster_ids]

    # Assign colors (pass=blue, fail=red).
    colors = [CELLQUORUM_BLUE if p else CELLQUORUM_RED for p in qc_pass]

    # Create figure with 4 subplots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Plot 1: n_groups.
    ax = axes[0]
    x_pos = range(len(cluster_ids))
    ax.bar(x_pos, n_groups, color=colors, alpha=0.7, edgecolor="none")
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Number of donors", fontsize=10)
    ax.set_title("Donor coverage", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    apply_cellquorum_axis_style(ax)

    # Plot 2: max_group_frac (with threshold line).
    ax = axes[1]
    ax.bar(x_pos, max_group_frac, color=colors, alpha=0.7, edgecolor="none")
    # Add threshold line at 0.8 (typical max_group_frac threshold).
    ax.axhline(
        0.8,
        color=CELLQUORUM_RED,
        linestyle="--",
        linewidth=1.5,
        label="Threshold = 0.8",
    )
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Max donor fraction", fontsize=10)
    ax.set_title("One-donor-dominated detector", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    apply_cellquorum_axis_style(ax)

    # Plot 3: lodo_stability.
    ax = axes[2]
    # Replace None with 0 for plotting.
    lodo_values = [v if v is not None else 0 for v in lodo_stability]
    ax.bar(x_pos, lodo_values, color=colors, alpha=0.7, edgecolor="none")
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("LODO stability", fontsize=10)
    ax.set_title("Leave-one-donor-out stability", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    apply_cellquorum_axis_style(ax)

    # Plot 4: classifier_sep.
    ax = axes[3]
    # Replace None with 0 for plotting.
    classifier_values = [v if v is not None else 0 for v in classifier_sep]
    ax.bar(x_pos, classifier_values, color=colors, alpha=0.7, edgecolor="none")
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_title("Donor-blocked one-vs-rest", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    apply_cellquorum_axis_style(ax)

    # Adjust layout.
    plt.tight_layout()

    # Save figure.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cellquorum_figure(fig, out_path)
    plt.close(fig)

    return out_path


__all__ = ["plot_group_recovery", "plot_stability_curve", "plot_subcluster_qc_panel"]
