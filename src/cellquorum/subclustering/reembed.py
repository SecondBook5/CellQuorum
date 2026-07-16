"""Minimal re-embedding for the subclustering focus subset.

After ``extract_focus`` deletes the parent object's embeddings (the subset must
derive its own representation), the donor-reproducibility gate needs an
embedding to score cluster separability. This module provides a deliberately
minimal PCA re-embedding: normalize counts, log1p, and run PCA, writing the
result to ``obsm[embedding_key]``. A full HVG + integration re-embedding is a
documented follow-up (see ``ReembedConfig``); this is the smallest thing that
makes the gate meaningful without crashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

    from cellquorum.subclustering.config import ReembedConfig


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
