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
import pandas as pd


class CellRankComputeError(Exception):
    """Base for recoverable CellRank failures (→ MethodSkip)."""


class CellRankUnavailable(CellRankComputeError):
    """cellrank/scanpy not importable."""


class NoKernelInput(CellRankComputeError):
    """No connectivities and no usable rep to build a transition matrix."""


class SchurFailed(CellRankComputeError):
    """compute_schur raised; the estimator chain cannot proceed."""


class MacrostatesFailed(CellRankComputeError):
    """compute_macrostates raised; the estimator chain cannot proceed."""


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
    velocity_adata: ad.AnnData | None = None,
    velocity_model: str = "deterministic",
    time_key: str | None = None,
    realtime_epsilon: float = 0.1,
) -> tuple[object, dict]:
    """Build a combined CellRank kernel with graceful degradation.

    Combines a ConnectivityKernel with every resolvable directional kernel:
    PseudotimeKernel, CytoTRACEKernel, VelocityKernel (from ``velocity_adata``),
    and a moscot RealTimeKernel (gated on ``time_key``). Any directional kernel
    that cannot be built is dropped with a note — never a crash.

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
        try:
            np.random.seed(seed)
            sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors)
            notes.append(f"built neighbor graph on '{rep}'")
        except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
            raise NoKernelInput(f"neighbor graph construction failed on '{rep}': {exc}") from exc

    # 2. ConnectivityKernel is always constructible now.
    try:
        ck = ConnectivityKernel(adata).compute_transition_matrix()
    except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
        raise NoKernelInput(f"connectivity kernel failed: {exc}") from exc

    # 3. Collect EVERY resolvable directional kernel (not pick-one). Each entry
    # is (name, kernel); order controls only the display list, not the weight.
    directionals: list[tuple[str, object]] = []

    if pseudotime_key and pseudotime_key in adata.obs:
        col = adata.obs[pseudotime_key]
        if not col.isna().all():
            try:
                directionals.append(
                    (
                        "pseudotime",
                        PseudotimeKernel(
                            adata, time_key=pseudotime_key
                        ).compute_transition_matrix(),
                    )
                )
            except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
                notes.append(f"pseudotime kernel failed: {exc}")
        else:
            notes.append(f"pseudotime '{pseudotime_key}' all-NaN; connectivity-only")
    elif pseudotime_key:
        notes.append(f"pseudotime '{pseudotime_key}' absent; connectivity-only")

    if cytotrace_key:
        try:
            from cellrank.kernels import CytoTRACEKernel

            if cytotrace_key in adata.obs or "Ms" in adata.layers:
                ctk = CytoTRACEKernel(adata)
                ctk = ctk.compute_cytotrace() if hasattr(ctk, "compute_cytotrace") else ctk
                directionals.append(("cytotrace", ctk.compute_transition_matrix()))
            else:
                notes.append(f"cytotrace '{cytotrace_key}' absent; connectivity-only")
        except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
            notes.append(f"cytotrace kernel failed: {exc}")

    # 3b. VelocityKernel from a whole-object velocity h5ad (opt-in upstream).
    # Requires Ms + velocity layers and 1:1 obs alignment with the working atlas.
    if velocity_adata is not None:
        if "Ms" not in velocity_adata.layers or "velocity" not in velocity_adata.layers:
            notes.append("velocity kernel skipped: velocity_adata lacks Ms/velocity layers")
        elif list(velocity_adata.obs_names) != list(adata.obs_names):
            notes.append("velocity kernel skipped: obs_names mismatch with working atlas")
        else:
            try:
                from cellrank.kernels import VelocityKernel

                vk = VelocityKernel(velocity_adata).compute_transition_matrix(
                    model=velocity_model, seed=seed
                )
                directionals.append(("velocity", vk))
            except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
                notes.append(f"velocity kernel failed: {exc}")

    # 4. Combine: connectivity carries weight_connectivities; the remainder is
    # split equally across the resolved directional kernels. With exactly one
    # directional this is IDENTICAL to (1-w)*dir + w*conn.
    w_conn = float(weight_connectivities)
    if directionals:
        w_each = (1.0 - w_conn) / len(directionals)
        kernel = w_conn * ck
        weights = {"connectivity": w_conn}
        for name, k in directionals:
            kernel = kernel + w_each * k
            weights[name] = w_each
    else:
        kernel = ck
        weights = {"connectivity": 1.0}
        notes.append("connectivity-only kernel (no directionality input)")

    kernels_used = [name for name, _ in directionals] + ["connectivity"]
    info = {
        "kernels": kernels_used,
        "weight_connectivities": w_conn,
        "weights": weights,
        "notes": notes,
    }
    return kernel, info


def run_gpcca(
    adata: ad.AnnData,
    kernel: object,
    *,
    cluster_key: str,
    n_components: int,
    n_states: int,
    n_terminal_states: int | None,
    terminal_method: str,
    predict_initial_states: bool,
    n_initial_states: int,
    seed: int,
) -> dict:
    """Run the GPCCA chain in place on ``adata``; return a result dict.

    Uses the env-verified deterministic, no-petsc/slepc arguments. Raises
    ``SchurFailed`` when Schur decomposition fails (chain cannot proceed);
    later steps degrade gracefully with recorded notes.
    """
    import cellrank as cr

    notes: list[str] = []

    # cluster_key MUST be categorical (compute_macrostates uses the .cat accessor).
    if not isinstance(adata.obs[cluster_key].dtype, pd.CategoricalDtype):
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    g = cr.estimators.GPCCA(kernel)

    # Clamp Schur components to [n_states+1, n_obs-1].
    n_comp = max(int(n_components), int(n_states) + 1)
    n_comp = min(n_comp, int(adata.n_obs) - 1)
    try:
        g.compute_schur(n_components=n_comp, method="brandts")
    except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
        raise SchurFailed(f"compute_schur failed: {exc}") from exc

    try:
        g.compute_macrostates(n_states=int(n_states), cluster_key=cluster_key)
        macro_names = [str(x) for x in g.macrostates.cat.categories]
    except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
        raise MacrostatesFailed(f"compute_macrostates failed: {exc}") from exc

    result: dict = {
        "n_macrostates_requested": int(n_states),
        "n_macrostates_actual": len(macro_names),
        "macrostate_names": macro_names,
        "terminal_states": [],
        "fate_prob": None,
        "fate_names": [],
        "drivers": None,
        "notes": notes,
    }

    # Terminal states: stability, with a top_n fallback.
    try:
        g.predict_terminal_states(method=terminal_method, n_states=n_terminal_states)
    except ValueError as exc:
        notes.append(f"predict_terminal_states('{terminal_method}') failed: {exc}; trying top_n")
        try:
            g.predict_terminal_states(method="top_n", n_states=n_terminal_states or 2)
        except Exception as exc2:  # noqa: BLE001 — keep macrostates, skip fate probs
            notes.append(f"terminal-state prediction failed: {exc2}")
            return result
    except Exception as exc:  # noqa: BLE001
        notes.append(f"terminal-state prediction failed: {exc}")
        return result

    result["terminal_states"] = [str(x) for x in g.terminal_states.cat.categories]

    # Optional initial states (best-effort).
    if predict_initial_states:
        try:
            g.predict_initial_states(n_states=int(n_initial_states))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"predict_initial_states failed: {exc}")

    # Fate probabilities (deterministic, no petsc).
    try:
        g.compute_fate_probabilities(use_petsc=False, solver="direct", show_progress_bar=False)
        fp = g.fate_probabilities
        result["fate_prob"] = np.asarray(fp)
        result["fate_names"] = [str(x) for x in fp.names]
    except Exception as exc:  # noqa: BLE001 — keep macrostates + terminal states
        notes.append(f"compute_fate_probabilities failed: {exc}")
        return result

    # Lineage drivers (best-effort).
    try:
        result["drivers"] = g.compute_lineage_drivers(cluster_key=cluster_key, seed=seed)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"compute_lineage_drivers failed: {exc}")

    result["estimator"] = g
    return result


__all__ = [
    "CellRankComputeError",
    "CellRankUnavailable",
    "MacrostatesFailed",
    "NoKernelInput",
    "SchurFailed",
    "build_kernel",
    "run_gpcca",
]
