"""Focus extraction and group filtering for subclustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

    from cellquorum.stages.clustering.subclustering.config import FocusConfig, ReembedConfig


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


def ensure_focus_embedding(
    adata: ad.AnnData,
    *,
    counts_layer: str,
    embedding_key: str = "X_pca",
    reembed: ReembedConfig | None = None,
    random_state: int = 0,
) -> bool:
    """
    Ensure ``adata.obsm[embedding_key]`` exists, computing a minimal PCA if not.

    The embedding is derived from a normalized + log1p transform of the counts
    layer (or ``X`` when the counts layer is absent). This never mutates the
    caller's ``X``/layers — normalization is done on a scratch copy.

    Args:
        adata: Focus subset (counts in ``X`` or ``counts_layer``).
        counts_layer: Layer holding raw counts.
        embedding_key: obsm key to populate (default ``X_pca``).
        reembed: Optional re-embedding configuration (uses ``n_comps`` if given).
        random_state: PCA random seed.

    Returns:
        True if an embedding is present (pre-existing or freshly computed);
        False if the subset is too small/degenerate to embed.
    """

    # Respect an embedding that already exists (e.g., a real re-embedding step).
    if embedding_key in adata.obsm:
        return True

    # A PCA needs at least 2 cells and 2 genes to be meaningful.
    if adata.n_obs < 2 or adata.n_vars < 2:
        return False

    import numpy as np
    import scanpy as sc

    # Resolve the number of components: bounded by data shape, honoring config.
    default_comps = 50
    if reembed is not None:
        default_comps = int(reembed.hvg.get("n_comps", default_comps)) if reembed.hvg else 50
    n_comps = int(min(default_comps, adata.n_obs - 1, adata.n_vars - 1))
    if n_comps < 1:
        return False

    # Build a scratch object from counts so we never touch the caller's X/layers.
    if counts_layer in adata.layers:
        counts = adata.layers[counts_layer]
    else:
        counts = adata.X
    scratch = sc.AnnData(
        X=np.asarray(counts).copy() if not hasattr(counts, "toarray") else counts.copy()
    )

    # Standard normalize -> log1p -> PCA on the scratch copy.
    sc.pp.normalize_total(scratch, target_sum=1e4)
    sc.pp.log1p(scratch)
    sc.pp.pca(scratch, n_comps=n_comps, random_state=random_state)

    # Write the embedding back onto the real object.
    adata.obsm[embedding_key] = scratch.obsm["X_pca"]
    return True


__all__ = [
    "extract_focus",
    "apply_group_filter",
    "ensure_focus_embedding",
]
