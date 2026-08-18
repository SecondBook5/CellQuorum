"""CellRank 2.x kernel construction + GPCCA chain, behind import guards.

CellRank 2.x ONLY: kernels (``cellrank.kernels.*``) + estimators
(``cellrank.estimators.GPCCA``). The whole chain runs on ONE object with a
``cluster_key`` — never per-group — so cross-lineage fate probabilities are
valid. Every heavy call is guarded and retyped into a recoverable error so the
method layer can skip-not-crash.

Schur decomposition auto-selects its solver: when ``slepc4py``/``petsc4py`` are
importable, Schur uses the *sparse* ``method="krylov"`` (SLEPc), which computes
the same partial real Schur decomposition without ever densifying the
transition matrix. When they are absent, it falls back to ``method="brandts"``,
which densifies the ``n_obs x n_obs`` matrix — safe only for small objects, so a
memory guard skips-not-crashes above a RAM budget rather than OOM-killing the
whole pipeline. Fate probabilities follow the SAME backend: with SLEPc present
they solve through PETSc (``use_petsc=True, solver="gmres"``) because a dense
OpenBLAS solve after a SLEPc eigensolve deadlocks in-process; without SLEPc they
use the deterministic dense ``use_petsc=False, solver="direct"``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def _slepc_available() -> bool:
    """True when the SLEPc/PETSc sparse Schur backend is importable."""
    try:
        import petsc4py  # noqa: F401
        import slepc4py  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure → sparse path off
        return False
    return True


def _available_memory_bytes() -> int | None:
    """Best-effort MemAvailable (bytes) from /proc/meminfo, or None."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 — unreadable → caller uses a fixed cap
        return None
    return None


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


class MoscotUnavailable(CellRankComputeError):
    """moscot/RealTimeKernel not importable; the RealTimeKernel is skipped."""


class RealTimeKernelFailed(CellRankComputeError):
    """moscot TemporalProblem solve / RealTimeKernel construction raised."""


def _resolve_use_rep(adata: ad.AnnData, configured: str | None, fallback: list[str]) -> str | None:
    """Return a rep present in obsm, or None."""
    if configured and configured in adata.obsm:
        return configured
    for candidate in fallback:
        if candidate in adata.obsm:
            return candidate
    return None


def _build_realtime_kernel(
    adata: ad.AnnData,
    time_key: str,
    time_numeric: pd.Series,
    use_rep: str | None,
    use_rep_fallback: list[str],
    realtime_epsilon: float,
    notes: list[str],
) -> object | None:
    """Build a moscot-backed RealTimeKernel; return it or None (with a note).

    Solves a moscot ``TemporalProblem`` over the numeric ``time_key`` axis using
    a resolved representation as the joint attribute, then wraps the solution
    with ``RealTimeKernel.from_moscot``. Never raises: import failure or a solve
    error is recorded as a note and returns None (skip-not-crash).
    """
    rep = _resolve_use_rep(adata, use_rep, use_rep_fallback)
    if rep is None:
        notes.append("realtime kernel skipped: no usable representation for moscot joint_attr")
        return None
    try:
        from cellrank.kernels import RealTimeKernel
        from moscot.problems.time import TemporalProblem
    except ImportError as exc:
        notes.append(f"realtime kernel skipped: moscot/RealTimeKernel unavailable ({exc})")
        return None

    try:
        # A dedicated numeric, ordered-categorical time column drives the problem;
        # never mutate the caller's obs in place.
        tp_adata = adata.copy()
        levels = sorted(time_numeric.dropna().unique())
        tp_adata.obs["_cq_time"] = pd.Categorical(
            time_numeric.to_numpy(), categories=levels, ordered=True
        )
        # Cells with a non-numeric/missing time cannot enter the problem.
        tp_adata = tp_adata[~time_numeric.isna().to_numpy()].copy()

        problem = TemporalProblem(tp_adata)
        problem = problem.prepare(time_key="_cq_time", joint_attr={"attr": "obsm", "key": rep})
        problem = problem.solve(epsilon=float(realtime_epsilon))
        rtk = RealTimeKernel.from_moscot(problem)
        rtk = rtk.compute_transition_matrix()
        return rtk
    except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
        notes.append(f"realtime kernel failed: {exc}")
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

    # 3c. RealTimeKernel via a moscot TemporalProblem (gated on time_key). Common
    # case/control lymphedema data has no experimental time axis, so this is
    # skipped by default. Requires EVERY cell to carry a finite numeric time and
    # ≥2 distinct levels: a partially-populated time column would yield a
    # subset-shaped kernel that neither combines with the full-atlas connectivity
    # kernel (shape mismatch) nor corresponds to it cell-for-cell.
    if time_key:
        if time_key not in adata.obs:
            notes.append(f"realtime kernel skipped: time_key '{time_key}' absent")
        else:
            time_numeric = pd.to_numeric(adata.obs[time_key], errors="coerce")
            n_levels = int(time_numeric.dropna().nunique())
            if time_numeric.isna().any():
                n_missing = int(time_numeric.isna().sum())
                notes.append(
                    f"realtime kernel skipped: time_key '{time_key}' has "
                    f"{n_missing} cell(s) with non-numeric/missing time"
                )
            elif n_levels < 2:
                notes.append(
                    f"realtime kernel skipped: time_key '{time_key}' has "
                    f"{n_levels} distinct numeric level(s)"
                )
            else:
                rtk = _build_realtime_kernel(
                    adata,
                    time_key,
                    time_numeric,
                    use_rep,
                    use_rep_fallback,
                    realtime_epsilon,
                    notes,
                )
                if rtk is not None:
                    directionals.append(("realtime", rtk))

    # 4. Combine: connectivity carries weight_connectivities; the remainder is
    # split equally across the resolved directional kernels. With exactly one
    # directional this is IDENTICAL to (1-w)*dir + w*conn. The gates above ensure
    # every directional kernel is full-atlas-shaped, but the combine is guarded
    # anyway (skip-not-crash): a CellRank shape/validation error retypes into a
    # recoverable NoKernelInput rather than escaping _run.
    w_conn = float(weight_connectivities)
    if directionals:
        w_each = (1.0 - w_conn) / len(directionals)
        try:
            kernel = w_conn * ck
            weights = {"connectivity": w_conn}
            for name, k in directionals:
                kernel = kernel + w_each * k
                weights[name] = w_each
        except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
            raise NoKernelInput(f"kernel combine failed: {exc}") from exc
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

    # Solver selection: prefer the sparse SLEPc 'krylov' Schur when available —
    # it computes the same partial real Schur decomposition as 'brandts' but on
    # the sparse transition matrix, so memory stays ~O(nnz + n_obs * n_comp)
    # instead of densifying to n_obs^2. Fall back to dense 'brandts' otherwise.
    if _slepc_available():
        schur_method = "krylov"
    else:
        schur_method = "brandts"
        # 'brandts' densifies the n_obs x n_obs matrix and scipy's real Schur
        # makes a second copy. Estimate the dense footprint and skip-not-crash
        # if it would blow past a safe RAM budget — an OOM here SIGKILLs the
        # whole pipeline (uncatchable), which is exactly what we must avoid.
        n = int(adata.n_obs)
        needed = n * n * 8 * 3  # transition matrix + Schur workspace + vectors
        avail = _available_memory_bytes()
        budget = int(avail * 0.6) if avail else (20000 * 20000 * 8 * 3)
        if needed > budget:
            raise SchurFailed(
                f"dense Schur (method='brandts') needs ~{needed / 1e9:.0f} GB for "
                f"{n} cells but only ~{budget / 1e9:.0f} GB is safely available; "
                "install petsc4py + slepc4py to enable the sparse 'krylov' solver"
            )
    notes.append(f"schur method: {schur_method}")
    try:
        g.compute_schur(n_components=n_comp, method=schur_method)
    except Exception as exc:  # noqa: BLE001 — retype as recoverable skip
        raise SchurFailed(f"compute_schur ({schur_method}) failed: {exc}") from exc

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

    # Fate probabilities. Solver choice MUST match the Schur backend: when the
    # sparse SLEPc 'krylov' Schur ran, PETSc/SLEPc is initialized in this process
    # and a subsequent *dense OpenBLAS* solve (use_petsc=False, solver="direct")
    # DEADLOCKS against the thread state SLEPc leaves behind — an unrecoverable
    # hang, not an exception. So when SLEPc is available we solve the fate-prob
    # linear system through PETSc too (use_petsc=True, iterative 'gmres', which is
    # also CellRank's default and the scalable path); otherwise we keep the
    # deterministic dense direct solve. Both are deterministic given the inputs.
    if _slepc_available():
        fate_kwargs = {"use_petsc": True, "solver": "gmres"}
    else:
        fate_kwargs = {"use_petsc": False, "solver": "direct"}
    try:
        g.compute_fate_probabilities(show_progress_bar=False, **fate_kwargs)
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
    "MoscotUnavailable",
    "NoKernelInput",
    "RealTimeKernelFailed",
    "SchurFailed",
    "build_kernel",
    "run_gpcca",
]
