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


def compute_dpt(
    adata: ad.AnnData,
    *,
    use_rep: str | None,
    use_rep_fallback: list[str],
    n_neighbors: int,
    n_comps: int,
    n_dcs: int,
    n_branchings: int,
    iroot: int,
) -> dict:
    """Run scanpy diffmap + DPT in place; return {pseudotime, n_dcs, notes}."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise PseudotimeUnavailable("scanpy not installed") from exc

    notes: list[str] = []
    rep = resolve_rep(adata, use_rep, use_rep_fallback)
    if rep is None:
        raise NoRepresentation("no usable representation for diffmap")

    if "neighbors" not in adata.uns or "connectivities" not in adata.obsp:
        try:
            sc.pp.neighbors(adata, use_rep=rep, n_neighbors=int(n_neighbors))
            notes.append(f"built neighbor graph on '{rep}'")
        except Exception as exc:  # noqa: BLE001
            raise NoRepresentation(f"neighbor graph failed on '{rep}': {exc}") from exc

    try:
        sc.tl.diffmap(adata, n_comps=int(n_comps))
    except Exception as exc:  # noqa: BLE001
        raise DiffmapFailed(f"diffmap failed: {exc}") from exc

    adata.uns["iroot"] = int(iroot)
    try:
        sc.tl.dpt(adata, n_dcs=int(n_dcs), n_branchings=int(n_branchings))
    except Exception as exc:  # noqa: BLE001
        raise PseudotimeFailed(f"dpt failed: {exc}") from exc

    pseudotime = np.asarray(adata.obs["dpt_pseudotime"], dtype="float64")
    return {"pseudotime": pseudotime, "n_dcs": int(n_dcs), "notes": notes}


def compute_palantir(
    adata: ad.AnnData,
    *,
    use_rep: str | None,
    use_rep_fallback: list[str],
    n_components: int,
    knn: int,
    n_eigs: int,
    num_waypoints: int,
    early_cell: str,
    seed: int,
) -> dict:
    """Run the 3-step Palantir pipeline in place; return result dict."""
    try:
        import palantir
    except ImportError as exc:
        raise PseudotimeUnavailable("palantir not installed") from exc

    notes: list[str] = []
    rep = resolve_rep(adata, use_rep, use_rep_fallback)
    if rep is None:
        raise NoRepresentation("no usable representation for diffusion maps")

    try:
        palantir.utils.run_diffusion_maps(
            adata, pca_key=rep, n_components=int(n_components), knn=int(knn), seed=int(seed)
        )
        palantir.utils.determine_multiscale_space(adata, n_eigs=int(n_eigs))
    except Exception as exc:  # noqa: BLE001
        raise DiffmapFailed(f"diffusion maps failed: {exc}") from exc

    nwp = min(int(num_waypoints), int(adata.n_obs))
    try:
        res = palantir.core.run_palantir(
            adata,
            early_cell=early_cell,
            num_waypoints=nwp,
            seed=int(seed),
            use_early_cell_as_start=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise PseudotimeFailed(f"run_palantir failed: {exc}") from exc

    import pandas as pd

    pseudotime = pd.Series(np.asarray(res.pseudotime), index=list(res.pseudotime.index))
    entropy = pd.Series(np.asarray(res.entropy), index=list(res.entropy.index))
    fate_prob = None
    fate_names: list[str] = []
    branch = getattr(res, "branch_probs", None)
    if branch is not None:
        fate_prob = np.asarray(branch)
        fate_names = [str(c) for c in list(branch.columns)]
    return {
        "pseudotime": pseudotime,
        "entropy": entropy,
        "fate_prob": fate_prob,
        "fate_names": fate_names,
        "notes": notes,
    }


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
    "compute_dpt",
    "compute_palantir",
]
