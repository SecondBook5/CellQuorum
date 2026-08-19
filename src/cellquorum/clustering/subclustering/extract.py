"""Focus extraction and group filtering for subclustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

    from cellquorum.clustering.subclustering.config import FocusConfig


def extract_focus(
    adata: ad.AnnData,
    focus: FocusConfig,
    counts_layer: str,
) -> ad.AnnData:
    """
    Extract focus lineage subset and restore counts to X.

    This subsets the input AnnData to the specified focus labels, restores
    raw counts to X (for downstream CHOIR/sc-SHC), and deletes stale
    embeddings so the subset re-derives its own representations.

    Args:
        adata: Input AnnData (never mutated).
        focus: Focus configuration (label_key + labels).
        counts_layer: Layer containing raw counts.

    Returns:
        Focused AnnData subset (a copy, not a view).
    """
    # Subset to focus labels (empty labels = keep all cells).
    if focus.labels:
        mask = adata.obs[focus.label_key].isin(focus.labels)
        focused = adata[mask].copy()
    else:
        focused = adata.copy()

    # Record provenance BEFORE restoring counts (n_cells_kept needs obs).
    focused.uns["subcluster_extraction"] = {
        "label_key": focus.label_key,
        "labels": focus.labels,
        "n_cells_total": adata.n_obs,
        "n_cells_kept": focused.n_obs,
    }

    # Restore raw counts to X (for CHOIR/sc-SHC).
    if counts_layer in focused.layers:
        focused.X = focused.layers[counts_layer].copy()

    # Delete stale embeddings (subset must re-derive its own).
    stale_obsm_keys = [
        k
        for k in focused.obsm.keys()
        if k.startswith("X_")  # X_pca, X_umap, X_pca_harmony, etc.
    ]
    for key in stale_obsm_keys:
        del focused.obsm[key]

    # Delete stale varm (PCA loadings, etc.).
    stale_varm_keys = [k for k in focused.varm.keys() if k.startswith("PCs")]
    for key in stale_varm_keys:
        del focused.varm[key]

    # Delete stale uns (neighbors, etc.).
    stale_uns_keys = ["neighbors", "umap"]
    for key in stale_uns_keys:
        if key in focused.uns:
            del focused.uns[key]

    return focused


def apply_group_filter(
    adata: ad.AnnData,
    group_key: str | None,
    min_cells: int | None,
) -> tuple[ad.AnnData, dict]:
    """
    Filter groups with < min_cells (generic KC<100-per-patient rule).

    This counts cells per group and drops groups below the threshold.
    Used to remove donors/samples with insufficient focus cells before
    subclustering.

    Args:
        adata: Input AnnData (never mutated).
        group_key: obs column for grouping (e.g., patient_id, donor_id).
            None = no-op.
        min_cells: minimum cells per group to keep. None = no-op.

    Returns:
        (filtered_adata, provenance_dict).
        provenance_dict contains:
        - applied: bool (False for no-op)
        - group_key: str
        - min_cells: int
        - counts: {group: int}
        - kept: list[str]
        - dropped: list[str]
    """
    # No-op when group_key or min_cells is None.
    if group_key is None or min_cells is None:
        return adata, {"applied": False}

    # Count cells per group.
    counts = adata.obs[group_key].value_counts().to_dict()

    # Identify kept and dropped groups.
    kept = [g for g, n in counts.items() if n >= min_cells]
    dropped = [g for g, n in counts.items() if n < min_cells]

    # Filter to kept groups.
    mask = adata.obs[group_key].isin(kept)
    filtered = adata[mask].copy()

    # Build provenance.
    provenance = {
        "applied": True,
        "group_key": group_key,
        "min_cells": min_cells,
        "counts": counts,
        "kept": kept,
        "dropped": dropped,
    }

    return filtered, provenance


__all__ = [
    "extract_focus",
    "apply_group_filter",
]
