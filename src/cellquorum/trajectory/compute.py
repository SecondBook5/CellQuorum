"""scVelo velocity compute, behind import guards + a typed error hierarchy.

Velocity is embedding-independent: it is computed once in ``use_rep`` space; only
``reproject_velocity`` touches 2D coordinates, so any embedding re-projects with
no recompute.
"""

from __future__ import annotations

import anndata as ad
import numpy as np


class TrajectoryComputeError(Exception):
    """Base for recoverable trajectory-compute failures (→ MethodSkip)."""


class ScveloUnavailable(TrajectoryComputeError):
    """scvelo/scanpy not importable."""


class VelocityComputeFailed(TrajectoryComputeError):
    """A scVelo backend step raised (non-convergence, degenerate input, …)."""


def resolve_use_rep(adata: ad.AnnData, configured: str | None, fallback: list[str]) -> str | None:
    """Return the representation to use for moments, or None if none present."""
    if configured and configured in adata.obsm:
        return configured
    for candidate in fallback:
        if candidate in adata.obsm:
            return candidate
    return None


def embedding_bases(adata: ad.AnnData) -> list[str]:
    """Sorted 2D-embedding basis names (``X_`` stripped), excluding ``X_pca``."""
    bases = []
    for key in adata.obsm.keys():
        if not key.startswith("X_") or key == "X_pca":
            continue
        arr = np.asarray(adata.obsm[key])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            bases.append(key[len("X_") :])
    return sorted(bases)


def compute_velocity(
    adata: ad.AnnData,
    *,
    mode: str,
    use_rep: str,
    min_shared_counts: int,
    n_top_genes: int,
    n_pcs: int,
    n_neighbors: int,
    n_jobs: int,
    seed: int,
) -> None:
    """Run the scVelo pipeline in place. Raises typed errors (skip-not-crash)."""
    try:
        import scanpy as sc
        import scvelo as scv
    except ImportError as exc:
        raise ScveloUnavailable("scvelo/scanpy not installed") from exc

    # scvelo 0.3.4's EM (recover_dynamics) exposes no ``random_state`` argument —
    # passing one raises TypeError deep in a worker and, worse, leaves scvelo's
    # progress-bar monitor thread orphaned so the interpreter deadlocks at exit.
    # The reproducibility knob that DOES exist is the process-global numpy seed;
    # set it here and disable the progress bar (which spawns that monitor thread).
    np.random.seed(seed)
    try:
        scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
        scv.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars))
        if "highly_variable" in adata.var and adata.n_vars > n_top_genes:
            adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())
        scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, use_rep=use_rep)
        if mode == "dynamical":
            scv.tl.recover_dynamics(adata, n_jobs=n_jobs, show_progress_bar=False)
        scv.tl.velocity(adata, mode=mode)
        scv.tl.velocity_graph(adata, n_jobs=n_jobs, show_progress_bar=False)
        scv.tl.velocity_confidence(adata)
        scv.tl.velocity_pseudotime(adata)
    except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
        raise VelocityComputeFailed(f"velocity computation failed: {exc}") from exc


def reproject_velocity(adata: ad.AnnData, *, bases: list[str]) -> list[str]:
    """Re-project velocity onto each basis; return those that succeeded."""
    try:
        import scvelo as scv
    except ImportError:
        return []
    projected = []
    for basis in bases:
        try:
            scv.tl.velocity_embedding(adata, basis=basis)
            projected.append(basis)
        except Exception:  # noqa: BLE001 — best-effort per basis
            continue
    return projected


__all__ = [
    "TrajectoryComputeError",
    "ScveloUnavailable",
    "VelocityComputeFailed",
    "resolve_use_rep",
    "embedding_bases",
    "compute_velocity",
    "reproject_velocity",
]
