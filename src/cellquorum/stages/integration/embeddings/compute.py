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


class ComputeFailed(EmbeddingsComputeError):
    """The backend embedding routine failed (e.g. graph too small).

    Wraps a lower-level scanpy/scipy/umap error so the calling method skips
    cleanly instead of letting the raw exception abort the pipeline. Typical
    trigger: too few cells for UMAP's spectral init (scipy eigsh needs k < N).
    """


def _has_neighbors(adata: ad.AnnData) -> bool:
    """True when a neighbors graph exists (connectivities present)."""
    return "neighbors" in adata.uns or "connectivities" in adata.obsp


def _is_usable_grouping(adata: ad.AnnData, column: str) -> bool:
    """True when `column` is a grouping PAGA can run on.

    A column is usable only when it is present AND resolves to at least two
    distinct non-null groups. A degenerate column — absent, all-null, an empty
    categorical (0 categories / all codes -1), or a single group — is rejected
    so callers skip cleanly instead of crashing inside scanpy/igraph, which
    indexes a per-group counts list and raises IndexError on empty membership.
    """
    if column not in adata.obs.columns:
        return False
    values = adata.obs[column]
    # nunique(dropna=True) counts only observed, non-null groups regardless of
    # dtype (categorical unused categories and NaN codes are both excluded).
    return int(values.nunique(dropna=True)) >= 2


def compute_umap(adata: ad.AnnData, *, min_dist: float, random_state: int) -> None:
    """Compute UMAP on the existing neighbors graph; write obsm['X_umap'].

    GPU via rapids_singlecell if available, else scanpy CPU. Seed threaded.
    """
    if not _has_neighbors(adata):
        raise NeighborsMissing("neighbors graph absent; run clustering first")
    # GPU (rapids_singlecell) first when available; any GPU-path failure — the
    # common case here is that cuml is absent — falls back to the seeded scanpy
    # CPU path. A CPU-path failure (e.g. a neighbors graph too small for UMAP's
    # spectral init) is retyped as ComputeFailed so the method skips, never
    # crashing the pipeline.
    try:
        import rapids_singlecell as rsc

        rsc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
        return
    except Exception:  # noqa: BLE001 — GPU absent or failed; fall back to CPU
        pass
    try:
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    except Exception as exc:  # noqa: BLE001 — retype any backend failure as a typed skip
        raise ComputeFailed(f"umap computation failed: {exc}") from exc


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
    try:
        adata.obsm["X_phate"] = operator.fit_transform(np.asarray(adata.obsm[use_rep]))
    except Exception as exc:  # noqa: BLE001 — retype any backend failure as a typed skip
        raise ComputeFailed(f"phate computation failed: {exc}") from exc


def compute_paga(adata: ad.AnnData, *, groupby: str) -> None:
    """Compute PAGA connectivity over `groupby`; write uns['paga']."""
    if not _has_neighbors(adata):
        raise NeighborsMissing("neighbors graph absent; run clustering first")
    if groupby not in adata.obs.columns:
        raise GroupMissing(f"grouping column '{groupby}' absent from obs")
    if not _is_usable_grouping(adata, groupby):
        raise GroupMissing(f"grouping column '{groupby}' has fewer than two non-null groups")
    try:
        sc.tl.paga(adata, groups=groupby)
    except Exception as exc:  # noqa: BLE001 — retype any backend failure as a typed skip
        raise ComputeFailed(f"paga computation failed: {exc}") from exc


def resolve_paga_groupby(
    adata: ad.AnnData,
    configured: str | None,
    *,
    cell_type_key: str,
    granular_key: str | None = None,
    cluster_key: str,
) -> str | None:
    """Resolve the PAGA grouping column.

    Precedence — chosen so PAGA groups (and the figure labels) by NAMED cell
    types whenever any exist, and only falls back to numeric cluster IDs as a
    last resort::

        explicit configured column
        -> coarse cell-type column     (named; e.g. 'cell_type')
        -> granular cell-type column   (named subtypes; e.g. 'cell_type_granular')
        -> cluster column              (numeric leiden IDs — last resort)
        -> None

    The granular step is what keeps a *per-lineage* object (every cell one
    lineage, so the coarse column is single-valued and rejected as degenerate)
    labelled by its named subtypes instead of tumbling through to "0"/"1"/"2"
    leiden codes. On a multi-lineage atlas the coarse column is populated and
    wins first, giving a clean lineage-level topology; ask for granular there
    with an explicit ``paga_groupby``.

    A candidate is skipped when absent or degenerate (all-null / empty
    categorical / a single group). Returns None when no candidate is usable.
    """
    if configured is not None and _is_usable_grouping(adata, configured):
        return configured
    if _is_usable_grouping(adata, cell_type_key):
        return cell_type_key
    if granular_key and _is_usable_grouping(adata, granular_key):
        return granular_key
    if _is_usable_grouping(adata, cluster_key):
        return cluster_key
    return None


__all__ = [
    "EmbeddingsComputeError",
    "NeighborsMissing",
    "RepMissing",
    "PhateUnavailable",
    "GroupMissing",
    "ComputeFailed",
    "compute_umap",
    "compute_phate",
    "compute_paga",
    "resolve_paga_groupby",
]
