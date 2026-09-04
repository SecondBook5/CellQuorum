"""Focus extraction and group filtering for subclustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import pandas as pd

    from cellquorum.backends.harmonypy_backend import HarmonyDiagnostics
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


def reembed_focus_batch_aware(
    adata: ad.AnnData,
    *,
    counts_layer: str,
    batch_key: str | None,
    embedding_key: str = "X_pca_harmony",
    n_top_genes: int = 2000,
    n_comps: int = 30,
    scale_max: float = 10.0,
    random_state: int = 0,
    max_iter_harmony: int | None = None,
    diagnostics: list[str] | None = None,
) -> str | None:
    """Re-embed the focus subset batch-aware; return the obsm key written, or None.

    Writes ``obsm[embedding_key]`` (batch-corrected when ``batch_key`` names a
    multi-valued column, plain PCA otherwise) and the boolean
    ``var['highly_variable']`` flag. Both are required by CHOIR's user-supplied
    ``reduction`` path.

    Why this exists rather than reusing :func:`ensure_focus_embedding`: that helper
    produces an UNCORRECTED PCA, and CHOIR's own documentation notes that batch
    correction is what "ensures that clusters do not originate from a single
    batch". Clustering a multi-donor subset on an uncorrected embedding yields
    donor-specific clusters that the permutation test then certifies as real — on
    the LEC subset that produced a cluster 98% composed of one donor. A
    significance-tested cluster count is only meaningful on a corrected space.

    ``diagnostics`` is an optional sink for messages the caller should surface as
    warnings — the function's return type is a single obsm key, which leaves it no
    other way to report that Harmony stopped before convergence.

    Never mutates the caller's ``X``/layers: normalization and scaling happen on
    scratch copies.
    """
    # Imported inside the function: the module only imports anndata/scanpy under
    # TYPE_CHECKING, matching `ensure_focus_embedding` above.
    import anndata as ad
    import numpy as np
    import scanpy as sc

    if adata.n_obs < 3 or adata.n_vars < 3:
        return None

    # Work from raw counts on a scratch copy, so the caller's matrices are safe.
    counts = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    dense = counts.copy() if hasattr(counts, "toarray") else np.asarray(counts).copy()
    scratch = ad.AnnData(X=dense)
    scratch.var_names = adata.var_names
    sc.pp.normalize_total(scratch, target_sum=1e4)
    sc.pp.log1p(scratch)

    n_batches = 0
    if batch_key and batch_key in adata.obs:
        scratch.obs[batch_key] = adata.obs[batch_key].astype(str).to_numpy()
        n_batches = int(scratch.obs[batch_key].nunique())

    # HVG selection needs >1 batch to use batch_key at all.
    n_genes = int(min(n_top_genes, adata.n_vars))
    sc.pp.highly_variable_genes(
        scratch,
        n_top_genes=n_genes,
        flavor="seurat",
        batch_key=batch_key if n_batches > 1 else None,
    )
    highly_variable = scratch.var["highly_variable"].to_numpy().astype(bool)
    if highly_variable.sum() < 2:
        return None
    adata.var["highly_variable"] = highly_variable

    # Scale on a further copy restricted to HVGs, then PCA.
    hvg_subset = scratch[:, highly_variable].copy()
    sc.pp.scale(hvg_subset, max_value=scale_max)
    comps = int(min(n_comps, adata.n_obs - 1, int(highly_variable.sum()) - 1))
    if comps < 2:
        return None
    sc.tl.pca(hvg_subset, n_comps=comps, svd_solver="arpack", random_state=random_state)
    embedding = hvg_subset.obsm["X_pca"]

    if n_batches > 1:
        result = _harmony_embedding(
            embedding,
            scratch.obs[batch_key],
            batch_key,
            random_state=random_state,
            max_iter_harmony=max_iter_harmony,
        )
        if result is None:
            # Harmony unavailable: fall back to the uncorrected PCA under its own
            # key, so a caller can tell corrected from uncorrected by key alone.
            adata.obsm["X_pca"] = embedding
            return "X_pca"
        corrected, harmony_diagnostics = result
        # CHOIR partitions on THIS embedding and its cluster count is a permutation
        # test against it, so a Harmony that stopped early is not a cosmetic detail:
        # it is the space in which the subclusters were declared significant.
        if diagnostics is not None and harmony_diagnostics.message:
            diagnostics.append(harmony_diagnostics.message)
        adata.obsm[embedding_key] = corrected
        return embedding_key

    adata.obsm["X_pca"] = embedding
    return "X_pca"


def _harmony_embedding(
    pcs: np.ndarray,
    batch: pd.Series,
    batch_key: str,
    *,
    random_state: int = 0,
    max_iter_harmony: int | None = None,
) -> tuple[np.ndarray, HarmonyDiagnostics] | None:
    """Harmony-correct ``pcs``, or None when harmonypy is unavailable.

    Delegates to :func:`cellquorum.backends.harmonypy_backend.harmony_correct` so
    this stage and ``integration`` call harmonypy the same way; the ``None`` return
    is this call site's own policy, because subclustering can fall back to an
    uncorrected PCA under a different obsm key whereas integration cannot.
    """
    from cellquorum.backends.harmonypy_backend import (
        DEFAULT_MAX_ITER_HARMONY,
        harmony_correct,
    )

    try:
        return harmony_correct(
            pcs,
            batch,
            batch_key,
            random_state=random_state,
            max_iter_harmony=(
                DEFAULT_MAX_ITER_HARMONY if max_iter_harmony is None else max_iter_harmony
            ),
        )
    except ImportError:
        return None


__all__ = [
    "extract_focus",
    "apply_group_filter",
    "ensure_focus_embedding",
    "reembed_focus_batch_aware",
]
