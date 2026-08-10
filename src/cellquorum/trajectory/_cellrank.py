"""CellRank 2.x kernel construction + GPCCA chain, behind import guards.

CellRank 2.x ONLY: kernels (``cellrank.kernels.*``) + estimators
(``cellrank.estimators.GPCCA``). The whole chain runs on ONE object with a
``cluster_key`` — never per-group — so cross-lineage fate probabilities are
valid. Every heavy call is guarded and retyped into a recoverable error so the
method layer can skip-not-crash.

Environment: ``petsc4py``/``slepc4py`` are absent, so Schur uses
``method="brandts"`` and fate probabilities use ``use_petsc=False,
solver="direct"`` — both also the deterministic choices.
"""

from __future__ import annotations

import anndata as ad
import numpy as np


class CellRankComputeError(Exception):
    """Base for recoverable CellRank failures (→ MethodSkip)."""


class CellRankUnavailable(CellRankComputeError):
    """cellrank/scanpy not importable."""


class NoKernelInput(CellRankComputeError):
    """No connectivities and no usable rep to build a transition matrix."""


class SchurFailed(CellRankComputeError):
    """compute_schur raised; the estimator chain cannot proceed."""


def _resolve_use_rep(adata: ad.AnnData, configured: str | None, fallback: list[str]) -> str | None:
    """Return a rep present in obsm, or None."""
    if configured and configured in adata.obsm:
        return configured
    for candidate in fallback:
        if candidate in adata.obsm:
            return candidate
    return None


def build_kernel(
    adata: ad.AnnData,
    *,
    pseudotime_key: str | None,
    cytotrace_key: str | None,
    use_rep: str | None,
    use_rep_fallback: list[str],
    n_neighbors: int,
    weight_connectivities: float,
    seed: int,
) -> tuple[object, dict]:
    """Build a combined CellRank kernel with graceful degradation.

    Returns ``(kernel, kernel_info)``. Raises ``CellRankUnavailable`` if cellrank
    is not importable, or ``NoKernelInput`` if neither connectivities nor a
    usable rep exist.
    """
    try:
        import scanpy as sc
        from cellrank.kernels import ConnectivityKernel, PseudotimeKernel
    except ImportError as exc:
        raise CellRankUnavailable("cellrank/scanpy not installed") from exc

    notes: list[str] = []

    # 1. Ensure a neighbor graph exists (needed for ConnectivityKernel).
    if "connectivities" not in adata.obsp:
        rep = _resolve_use_rep(adata, use_rep, use_rep_fallback)
        if rep is None:
            raise NoKernelInput("no connectivities and no usable representation")
        np.random.seed(seed)
        sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors)
        notes.append(f"built neighbor graph on '{rep}'")

    # 2. ConnectivityKernel is always constructible now.
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    kernels_used = ["connectivity"]

    # 3. Directionality kernel: pseudotime preferred, then cytotrace.
    directional = None
    if pseudotime_key and pseudotime_key in adata.obs:
        col = adata.obs[pseudotime_key]
        if not col.isna().all():
            try:
                directional = PseudotimeKernel(
                    adata, time_key=pseudotime_key
                ).compute_transition_matrix()
                kernels_used.insert(0, "pseudotime")
            except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
                notes.append(f"pseudotime kernel failed: {exc}")
        else:
            notes.append(f"pseudotime '{pseudotime_key}' all-NaN; connectivity-only")
    elif pseudotime_key:
        notes.append(f"pseudotime '{pseudotime_key}' absent; connectivity-only")

    if directional is None and cytotrace_key:
        try:
            from cellrank.kernels import CytoTRACEKernel

            if cytotrace_key in adata.obs or "Ms" in adata.layers:
                ctk = CytoTRACEKernel(adata)
                ctk = ctk.compute_cytotrace() if hasattr(ctk, "compute_cytotrace") else ctk
                directional = ctk.compute_transition_matrix()
                kernels_used.insert(0, "cytotrace")
            else:
                notes.append(f"cytotrace '{cytotrace_key}' absent; connectivity-only")
        except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
            notes.append(f"cytotrace kernel failed: {exc}")

    # 4. Combine.
    if directional is not None:
        w = float(weight_connectivities)
        kernel = (1.0 - w) * directional + w * ck
    else:
        kernel = ck
        notes.append("connectivity-only kernel (no directionality input)")

    info = {
        "kernels": kernels_used,
        "weight_connectivities": float(weight_connectivities),
        "notes": notes,
    }
    return kernel, info


__all__ = [
    "CellRankComputeError",
    "CellRankUnavailable",
    "NoKernelInput",
    "SchurFailed",
    "build_kernel",
]
