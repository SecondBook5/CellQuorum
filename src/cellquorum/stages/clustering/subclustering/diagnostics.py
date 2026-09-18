"""Subclustering diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.patches import Rectangle

from cellquorum.visualization.figstyle import (
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


def plot_technical_qc_consistency(
    result: dict,
    out_path: Path | str,
) -> Path:
    """
    Plot each subcluster's worst technical-QC z-score vs. its siblings.

    One bar per cluster: the largest-magnitude z-score across the metrics
    checked (mito%, counts, complexity, multiplet rate), signed so a bad
    direction is always positive height. PASS clusters are blue, FAIL
    (|z| > threshold on at least one metric) are red. See
    ``qc_consistency.subcluster_qc_consistency`` for how these are computed.

    Args:
        result: output of subcluster_qc_consistency().
        out_path: output file path.

    Returns:
        Path to saved figure (unsaved, non-existent path when there is
        nothing to plot -- the check was skipped or found no clusters).
    """
    out_path = Path(out_path)

    clusters_dict = result.get("clusters") or {}
    if not clusters_dict:
        return out_path

    apply_cellquorum_theme()

    cluster_ids = list(clusters_dict.keys())
    worst_z = []
    qc_pass = []
    for cid in cluster_ids:
        metrics = clusters_dict[cid].get("metrics", {})
        # Largest-magnitude z, signed toward the bad direction (a flagged
        # metric's z is already signed so "bad" is the sign it tripped on).
        z_values = [m["z"] for m in metrics.values()]
        worst = max(z_values, key=abs) if z_values else 0.0
        worst_z.append(abs(worst))
        qc_pass.append(clusters_dict[cid]["qc_pass"])

    colors = [CELLQUORUM_BLUE if p else CELLQUORUM_RED for p in qc_pass]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(cluster_ids))
    ax.bar(x_pos, worst_z, color=colors, alpha=0.7, edgecolor="none")
    ax.set_xlabel("Subcluster", fontsize=11)
    ax.set_ylabel("Worst |z| vs. sibling subclusters", fontsize=11)
    ax.set_title(
        "Technical-QC consistency across subclusters",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right")
    apply_cellquorum_axis_style(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cellquorum_figure(fig, out_path)
    plt.close(fig)

    return out_path


__all__ = [
    "plot_group_recovery",
    "plot_marker_identity_dotplot",
    "plot_stability_curve",
    "plot_subcluster_qc_panel",
    "plot_technical_qc_consistency",
    "select_marker_genes",
]


def select_marker_genes(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    layer: str,
    n_top: int = 6,
    method: str = "wilcoxon",
) -> dict[str, list[str]]:
    """Top-``n_top`` marker genes per cluster via ``rank_genes_groups``.

    Auto-derived, not a hand-picked gene list -- a hardcoded list goes stale
    silently the moment clusters change (different boundaries, a re-run on
    corrected input); this recomputes from whatever ``adata`` actually holds,
    every time the figure is generated.
    """
    work = adata.copy()
    work.X = work.layers[layer]
    sc.tl.rank_genes_groups(work, groupby=cluster_key, method=method, n_genes=n_top)
    result = sc.get.rank_genes_groups_df(work, group=None)
    markers: dict[str, list[str]] = {}
    for group in work.obs[cluster_key].astype(str).unique():
        ordered = result[result["group"] == group].sort_values("scores", ascending=False)
        markers[str(group)] = ordered.head(n_top)["names"].tolist()
    return markers


def _marker_highlight_boxes(
    *,
    markers: dict[str, list[str]],
    genes: list[str],
    row_order: list[str],
) -> list[tuple[float, float]]:
    """(x, y) center of each gene's dot in the cluster it was selected FOR.

    Scanpy's dotplot places column ``i`` and row ``j`` centers at
    ``i + 0.5`` / ``j + 0.5`` (confirmed against a live ``DotPlot`` -- ticks
    land at those half-integers in the order the groupby/gene list was
    given), so this stays purely a coordinate lookup rather than needing to
    introspect the rendered axis. A gene can own more than one cluster's box
    (shared top marker) or none (present only as cross-cluster context).
    """
    gene_x = {gene: i + 0.5 for i, gene in enumerate(genes)}
    row_y = {cluster: j + 0.5 for j, cluster in enumerate(row_order)}
    boxes: list[tuple[float, float]] = []
    for cluster_id, cluster_genes in markers.items():
        y = row_y.get(cluster_id)
        if y is None:
            continue
        for gene in cluster_genes:
            x = gene_x.get(gene)
            if x is not None:
                boxes.append((x, y))
    return boxes


def plot_marker_identity_dotplot(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    layer: str,
    out_path: Path | str,
    n_top: int = 6,
    cluster_labels: dict[str, str] | None = None,
) -> Path:
    """Marker-gene dotplot: which genes justify calling this cluster what it's called.

    Complements the technical-QC and donor-gate panels -- those ask "is this
    cluster trustworthy," this asks "what is it." Dot size = fraction of
    cells expressing; color = mean expression. Markers are auto-derived
    (:func:`select_marker_genes`), and a cluster with fewer than 2 siblings
    is skipped (rank_genes_groups needs something to rank against).

    ``cluster_labels`` renames clusters for display only -- ``adata.obs`` is
    never mutated in place, matching ``de_heatmap_method``'s convention, so
    the same object can be relabeled independently by each figure without
    the two colliding.
    """
    out_path = Path(out_path)
    n_clusters = adata.obs[cluster_key].astype(str).nunique()
    if n_clusters < 2:
        return out_path

    markers = select_marker_genes(adata, cluster_key=cluster_key, layer=layer, n_top=n_top)
    genes: list[str] = []
    for group_genes in markers.values():
        for g in group_genes:
            if g not in genes:
                genes.append(g)

    work = adata.copy()
    work.X = work.layers[layer]
    labels = cluster_labels or {}
    work.obs["_identity"] = work.obs[cluster_key].astype(str).map(lambda c: labels.get(c, c))
    raw_order = sorted(markers.keys(), key=lambda c: labels.get(c, c))
    labeled_order = [labels.get(c, c) for c in raw_order]
    work.obs["_identity"] = (
        work.obs["_identity"].astype("category").cat.reorder_categories(labeled_order)
    )

    apply_cellquorum_theme()
    fig = sc.pl.dotplot(work, genes, groupby="_identity", standard_scale="var", return_fig=True)
    main_ax = fig.get_axes()["mainplot_ax"]
    for x, y in _marker_highlight_boxes(markers=markers, genes=genes, row_order=raw_order):
        main_ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="gold",
                linewidth=2.2,
                zorder=10,
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close("all")
    return out_path
