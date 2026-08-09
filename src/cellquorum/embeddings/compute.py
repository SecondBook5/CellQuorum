"""Biology-agnostic embedding compute helpers: UMAP, PHATE, PAGA.

Each helper mutates the AnnData in place and raises a typed error the calling
method converts into a MethodSkip. GPU (rapids_singlecell) is attempted first
with a scanpy CPU fallback. Determinism: seeds are threaded into every
stochastic call.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc


class EmbeddingsComputeError(Exception):
    """Base for typed compute skips."""


class NeighborsMissing(EmbeddingsComputeError):
    """No neighbors graph on the AnnData."""


class RepMissing(EmbeddingsComputeError):
    """A required obsm representation is absent."""


class PhateUnavailable(EmbeddingsComputeError):
    """The optional 'phate' package is not importable."""


class GroupMissing(EmbeddingsComputeError):
    """The requested grouping obs column is absent."""


def _has_neighbors(adata: ad.AnnData) -> bool:
    """True when a neighbors graph exists (connectivities present)."""
    return "neighbors" in adata.uns or "connectivities" in adata.obsp


def compute_umap(adata: ad.AnnData, *, min_dist: float, random_state: int) -> None:
    """Compute UMAP on the existing neighbors graph; write obsm['X_umap'].

    GPU via rapids_singlecell if available, else scanpy CPU. Seed threaded.
    """
    if not _has_neighbors(adata):
        raise NeighborsMissing("neighbors graph absent; run clustering first")
    try:
        import rapids_singlecell as rsc

        rsc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    except Exception:  # noqa: BLE001 — GPU absent or failed; fall back to CPU
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)


def compute_phate(
    adata: ad.AnnData, *, use_rep: str, knn: int, decay: int, random_state: int
) -> None:
    """Compute PHATE on obsm[use_rep]; write obsm['X_phate']. Seed threaded."""
    try:
        import phate
    except ImportError as exc:
        raise PhateUnavailable("phate not installed") from exc
    if use_rep not in adata.obsm:
        raise RepMissing(f"representation '{use_rep}' absent from obsm")
    operator = phate.PHATE(
        knn=knn, decay=decay, t="auto", n_jobs=-1, random_state=random_state, verbose=False
    )
    adata.obsm["X_phate"] = operator.fit_transform(np.asarray(adata.obsm[use_rep]))


def compute_paga(adata: ad.AnnData, *, groupby: str) -> None:
    """Compute PAGA connectivity over `groupby`; write uns['paga']."""
    if not _has_neighbors(adata):
        raise NeighborsMissing("neighbors graph absent; run clustering first")
    if groupby not in adata.obs.columns:
        raise GroupMissing(f"grouping column '{groupby}' absent from obs")
    sc.tl.paga(adata, groups=groupby)


def resolve_paga_groupby(
    adata: ad.AnnData,
    configured: str | None,
    *,
    cell_type_key: str,
    cluster_key: str,
) -> str | None:
    """Resolve the PAGA grouping column.

    Precedence: explicit configured column (if present) -> cell-type column
    (if present) -> cluster column (if present) -> None.
    """
    if configured is not None and configured in adata.obs.columns:
        return configured
    if cell_type_key in adata.obs.columns:
        return cell_type_key
    if cluster_key in adata.obs.columns:
        return cluster_key
    return None


__all__ = [
    "EmbeddingsComputeError",
    "NeighborsMissing",
    "RepMissing",
    "PhateUnavailable",
    "GroupMissing",
    "compute_umap",
    "compute_phate",
    "compute_paga",
    "resolve_paga_groupby",
]
