"""Pseudotime compute (DPT + Palantir) behind import guards.

Every heavy scanpy/palantir call is guarded and retyped into a recoverable
``PseudotimeComputeError`` subclass so the method layer can skip-not-crash. Root
resolution is deterministic (argmax / group centroid, ties → lowest index).
"""

from __future__ import annotations

import anndata as ad
import numpy as np


class PseudotimeComputeError(Exception):
    """Base for recoverable pseudotime failures (→ MethodSkip)."""


class PseudotimeUnavailable(PseudotimeComputeError):
    """palantir/scanpy not importable."""


class NoRepresentation(PseudotimeComputeError):
    """No usable obsm representation to build a diffusion map."""


class RootUnresolved(PseudotimeComputeError):
    """No root strategy applies (no marker score, no valid root group)."""


class DiffmapFailed(PseudotimeComputeError):
    """diffmap / run_diffusion_maps raised."""


class PseudotimeFailed(PseudotimeComputeError):
    """dpt / run_palantir raised."""


def resolve_rep(adata: ad.AnnData, configured: str | None, fallback: list[str]) -> str | None:
    """Return a rep present in obsm (configured first, then fallback), or None."""
    if configured and configured in adata.obsm:
        return configured
    for candidate in fallback:
        if candidate in adata.obsm:
            return candidate
    return None


def resolve_root(
    adata: ad.AnnData,
    *,
    rep: str,
    marker_score_key: str | None,
    root_key: str | None,
    root_group: str | None,
) -> int:
    """Resolve a root cell index deterministically.

    Priority: (1) argmax of ``marker_score_key`` if present; (2) centroid cell of
    ``root_group`` within ``root_key`` in ``rep`` space. Ties break to the lowest
    index. Raises ``RootUnresolved`` when neither strategy applies.
    """
    if marker_score_key and marker_score_key in adata.obs:
        score = np.asarray(adata.obs[marker_score_key], dtype="float64")
        return int(np.argmax(score))  # argmax returns first max → lowest index

    if root_key and root_group is not None and root_key in adata.obs:
        labels = adata.obs[root_key].astype(str).to_numpy()
        in_group = np.flatnonzero(labels == str(root_group))
        if in_group.size:
            emb = np.asarray(adata.obsm[rep], dtype="float64")
            center = emb[in_group].mean(axis=0)
            dists = ((emb[in_group] - center) ** 2).sum(axis=1)
            return int(in_group[int(np.argmin(dists))])

    raise RootUnresolved(
        "no root: set root_marker_score_key or root_key+root_group (present in obs)"
    )


def flag_outliers(adata: ad.AnnData, rep: str, mad: float) -> np.ndarray:
    """Boolean mask of top-10-component robust-z (median/MAD) outliers in ``rep``."""
    emb = np.asarray(adata.obsm[rep], dtype="float64")
    emb = emb[:, : min(10, emb.shape[1])]
    med = np.median(emb, axis=0)
    mad_vec = np.median(np.abs(emb - med), axis=0) + 1e-9
    robust_z = np.abs(emb - med) / (1.4826 * mad_vec)
    return (robust_z > float(mad)).any(axis=1)


__all__ = [
    "PseudotimeComputeError",
    "PseudotimeUnavailable",
    "NoRepresentation",
    "RootUnresolved",
    "DiffmapFailed",
    "PseudotimeFailed",
    "resolve_rep",
    "resolve_root",
    "flag_outliers",
]
