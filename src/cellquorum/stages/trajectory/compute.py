"""scVelo velocity compute, behind import guards + a typed error hierarchy.

Velocity is embedding-independent: it is computed once in ``use_rep`` space; only
``reproject_velocity`` touches 2D coordinates, so any embedding re-projects with
no recompute.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

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


# ARPACK's iteration ceiling defaults to ``10 * n``, which makes it a function of
# CELL COUNT — so the smallest clusters are the ones most likely to run out of
# iterations, exactly backwards from what the numerics need. On the LEC arm,
# leiden cluster 5 (62 cells) died with "ARPACK error -1: No convergence (621
# iterations, 8/10 eigenvectors converged)": 621 is 10*62+1, the default ceiling
# to the digit. Neighbouring clusters of 41, 44 and 49 cells converged, so this is
# a ceiling that happens to bind, not a size threshold.
_ARPACK_RETRY_MIN_ITER = 5000
_ARPACK_RETRY_TOL = 1e-8


@contextlib.contextmanager
def _relaxed_arpack(n_obs: int) -> Iterator[None]:
    """Raise ARPACK's iteration ceiling for the duration of the block.

    scVelo reaches ARPACK twice inside ``velocity_pseudotime`` — ``eigs`` in
    ``terminal_states`` (root/end cells) and ``eigsh`` in ``VPT.compute_eigen`` —
    and passes neither ``maxiter`` nor ``tol`` to either, so both inherit the
    defaults. Neither is reachable through scVelo's own arguments (``n_dcs``
    reaches only the second, and ``terminal_states`` hard-codes ``k=10``), so the
    ceiling is raised at the scipy entry points instead. ``setdefault`` means an
    explicit caller value still wins.

    Both scvelo modules do ``from scipy.sparse import linalg`` and call
    ``linalg.eigs``/``linalg.eigsh`` as attributes, so patching the module's
    attributes is what they see. This is process-global for the duration of the
    block and therefore NOT thread-safe; it is used only around the (sequential,
    main-thread) pseudotime retry, never around scVelo's parallel steps.
    """
    import scipy.sparse.linalg as sla

    maxiter = max(_ARPACK_RETRY_MIN_ITER, 100 * int(n_obs))
    real_eigs, real_eigsh = sla.eigs, sla.eigsh

    def _relaxed(fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("maxiter", maxiter)
            kwargs.setdefault("tol", _ARPACK_RETRY_TOL)
            return fn(*args, **kwargs)

        return inner

    sla.eigs, sla.eigsh = _relaxed(real_eigs), _relaxed(real_eigsh)
    try:
        yield
    finally:
        sla.eigs, sla.eigsh = real_eigs, real_eigsh


def _velocity_pseudotime_best_effort(
    adata: ad.AnnData, scv: Any, warnings: list[str] | None
) -> str | None:
    """Velocity pseudotime + root/end cells, retried and then degraded.

    Deliberately does NOT raise, which is the fix rather than a convenience. This
    is the LAST step of the velocity chain and the only iterative eigensolve in
    it: moments, the velocity fit, the velocity graph and the confidence score
    have all already succeeded by the time it runs. Letting it raise meant an
    ARPACK non-convergence discarded a whole cluster's velocity — on the LEC arm,
    every one of leiden 5's 62 cells lost its velocity, graph and confidence
    because a downstream pseudotime eigensolve hit its iteration ceiling.

    Returns a status string for the caller's notes, or None on success.
    """
    try:
        scv.tl.velocity_pseudotime(adata)
        return None
    except Exception as exc:  # noqa: BLE001 — retried below, then degraded
        # Bound to an outer name because `except ... as` unbinds at block exit,
        # and the retry warning has to report what the first attempt said.
        first_failure = exc

    try:
        with _relaxed_arpack(adata.n_obs):
            scv.tl.velocity_pseudotime(adata)
    except Exception as exc:  # noqa: BLE001 — degrade: keep the velocity
        if warnings is not None:
            warnings.append(
                f"velocity_pseudotime failed ({exc}); velocity, velocity_graph and "
                "velocity_confidence are kept, so this group has velocity but no "
                "pseudotime or root/end cells"
            )
        return "pseudotime failed"

    if warnings is not None:
        warnings.append(
            f"velocity_pseudotime needed a relaxed-ARPACK retry; first attempt: {first_failure}"
        )
    return "pseudotime via relaxed ARPACK"


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
    warnings: list[str] | None = None,
) -> None:
    """Run the scVelo pipeline in place. Raises typed errors (skip-not-crash).

    Args:
        warnings: Appended to when a step DEGRADES rather than fails outright —
            currently only the pseudotime tail (see
            :func:`_velocity_pseudotime_best_effort`). Optional so duck-typed
            callers and tests need not thread a list through.
    """
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
        # scvelo rejects n_pcs > (width of use_rep): "does not have enough
        # Dimensions". When moments runs on a precomputed representation, cap
        # n_pcs at that rep's width so a small/auto-truncated embedding (e.g. a
        # low-cell slice, or PCA truncated below the requested n_pcs) still runs
        # instead of skipping velocity on every cluster.
        eff_n_pcs = n_pcs
        if use_rep and use_rep in adata.obsm:
            rep_dim = int(np.asarray(adata.obsm[use_rep]).shape[1])
            eff_n_pcs = min(n_pcs, rep_dim)
        scv.pp.moments(adata, n_pcs=eff_n_pcs, n_neighbors=n_neighbors, use_rep=use_rep)
        if mode == "dynamical":
            scv.tl.recover_dynamics(adata, n_jobs=n_jobs, show_progress_bar=False)
        scv.tl.velocity(adata, mode=mode)
        scv.tl.velocity_graph(adata, n_jobs=n_jobs, show_progress_bar=False)
        scv.tl.velocity_confidence(adata)
    except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
        raise VelocityComputeFailed(f"velocity computation failed: {exc}") from exc

    # Outside the raising block on purpose: everything above is the velocity
    # itself, everything here is a derived ordering along it.
    _velocity_pseudotime_best_effort(adata, scv, warnings)


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
