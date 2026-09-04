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


def _restore_python_sigpipe() -> None:
    """Take SIGPIPE back from PETSc.

    ``PetscInitialize`` installs its own signal handlers and the SIGPIPE one calls
    ``MPI_Abort``. Python ignores SIGPIPE by default precisely so that a closed
    downstream reader surfaces as a catchable ``BrokenPipeError``; under PETSc's
    handler the same benign event kills the process from C, past every ``except``.
    Seen twice here: ``cellquorum run | head`` died as an MPI abort mid-pipeline,
    and the test suite aborted at teardown before pytest could print its summary
    or its exit code.

    ONLY SIGPIPE is restored. PETSc's SIGSEGV/SIGFPE handlers print a native
    traceback that is genuinely useful when a solver crashes, so the blunter
    ``-no_signal_handler`` (which drops all of them) is the wrong instrument.
    """
    import signal

    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, ValueError, OSError):
        # No SIGPIPE on this platform, or not the main thread — ``signal.signal``
        # is main-thread-only. Nothing to undo; the sparse path still runs.
        pass


def _slepc_available() -> bool:
    """True when the SLEPc/PETSc sparse Schur backend is importable.

    Importing the ``PETSc`` submodule (not just ``petsc4py``) is what runs
    ``PetscInitialize``, which makes it both the honest availability test — the
    package can import while its extension fails to initialize — and the moment
    the process acquires PETSc's signal handlers. Doing it HERE is deliberate: the
    alternative is letting whichever CellRank call touches the solver first
    initialize PETSc with defaults, and then there is no point at which the engine
    can undo the SIGPIPE handler (see :func:`_restore_python_sigpipe`). Both
    callers use the sparse path immediately after this returns True, so nothing is
    initialized earlier than it would have been anyway.
    """
    try:
        import petsc4py  # noqa: F401
        import slepc4py  # noqa: F401
        from petsc4py import PETSc  # noqa: F401 — triggers PetscInitialize
    except Exception:  # noqa: BLE001 — any import failure → sparse path off
        return False
    _restore_python_sigpipe()
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
    warnings: list[str],
) -> object | None:
    """Build a moscot-backed RealTimeKernel; return it or None (with a warning).

    Solves a moscot ``TemporalProblem`` over the numeric ``time_key`` axis using
    a resolved representation as the joint attribute, then wraps the solution
    with ``RealTimeKernel.from_moscot``. Never raises: import failure or a solve
    error is recorded as a warning and returns None (skip-not-crash).
    """
    rep = _resolve_use_rep(adata, use_rep, use_rep_fallback)
    if rep is None:
        warnings.append("realtime kernel skipped: no usable representation for moscot joint_attr")
        return None
    try:
        from cellrank.kernels import RealTimeKernel
        from moscot.problems.time import TemporalProblem
    except ImportError as exc:
        warnings.append(f"realtime kernel skipped: moscot/RealTimeKernel unavailable ({exc})")
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
        warnings.append(f"realtime kernel failed: {exc}")
        return None


def _build_cytotrace_kernel(
    adata: ad.AnnData,
    cytotrace_key: str,
    notes: list[str],
    warnings: list[str],
) -> object | None:
    """Build a CytoTRACE-directed kernel; return it or None (with a warning).

    Prefers a CytoTRACE score already in ``obs``. CellRank's ``CytoTRACEKernel``
    re-implements CytoTRACE 1 and, as of cellrank 2.2.0, reads a layer named
    ``imputed`` that exists only after scVelo moments — so on an object carrying a
    CytoTRACE 2 score from the cytotrace stage the kernel raised on the missing
    layer and got dropped, and the run inferred fates from pseudotime plus velocity
    with the plasticity axis silently absent.

    Reading the score instead is not a workaround: ``CytoTRACEKernel`` IS a
    ``PseudotimeKernel`` over ``1 - minmax(score)`` (its own ``compute_cytotrace``
    writes exactly that into ``ct_pseudotime``), so building that pseudotime from
    the score in obs is the same construction on a better score — CytoTRACE 2, the
    successor model fit natively over all genes, rather than a 200-gene
    re-derivation of its predecessor.

    Falls back to CellRank's own computation when no score is present but an
    imputed layer is, and returns None otherwise. Never raises.
    """
    from cellrank.kernels import PseudotimeKernel

    if cytotrace_key in adata.obs:
        score = pd.to_numeric(adata.obs[cytotrace_key], errors="coerce").to_numpy(dtype="float64")
        n_missing = int(np.isnan(score).sum())
        if n_missing:
            # Filling would fabricate a potency for cells the scorer could not
            # score, and dropping them would desynchronise obs from the kernel the
            # other directional kernels were built on.
            warnings.append(
                f"cytotrace kernel skipped: '{cytotrace_key}' is missing for "
                f"{n_missing}/{len(score)} cells"
            )
            return None
        spread = float(np.nanmax(score) - np.nanmin(score))
        if not np.isfinite(spread) or spread == 0.0:
            warnings.append(f"cytotrace kernel skipped: '{cytotrace_key}' is constant")
            return None
        # 1 - minmax(score): high plasticity = early, exactly as CytoTRACEKernel
        # defines ct_pseudotime. Written under CellRank's own key so the artifact
        # names it the way CellRank's plots expect.
        adata.obs["ct_pseudotime"] = 1.0 - (score - float(np.nanmin(score))) / spread
        try:
            kernel = PseudotimeKernel(adata, time_key="ct_pseudotime").compute_transition_matrix()
        except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
            warnings.append(f"cytotrace kernel failed on '{cytotrace_key}': {exc}")
            return None
        notes.append(f"cytotrace kernel built from obs['{cytotrace_key}']")
        return kernel

    # No score in obs: let CellRank compute one, but only from a layer that is
    # actually imputed. Its default ('imputed') is absent unless scVelo moments
    # ran; 'Ms' is that same quantity under scVelo's name.
    layer = next((k for k in ("imputed", "Ms") if k in adata.layers), None)
    if layer is None:
        warnings.append(
            f"cytotrace kernel skipped: no '{cytotrace_key}' in obs and no imputed "
            "layer ('imputed'/'Ms') to compute one from"
        )
        return None
    try:
        from cellrank.kernels import CytoTRACEKernel

        ctk = CytoTRACEKernel(adata).compute_cytotrace(layer=layer)
        kernel = ctk.compute_transition_matrix()
    except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
        warnings.append(f"cytotrace kernel failed: {exc}")
        return None
    notes.append(f"cytotrace kernel computed by cellrank from layers['{layer}']")
    return kernel


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
    that cannot be built is dropped — never a crash — and reported in
    ``kernel_info['warnings']`` rather than its notes, because a kernel the
    caller configured and did not get changes what the result MEANS. One run
    inferred fates from connectivity alone, its velocity and CytoTRACE kernels
    both silently absent, and reported success.

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
    warnings: list[str] = []

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
                warnings.append(f"pseudotime kernel failed: {exc}")
        else:
            warnings.append(f"pseudotime '{pseudotime_key}' all-NaN; connectivity-only")
    elif pseudotime_key:
        warnings.append(f"pseudotime '{pseudotime_key}' absent; connectivity-only")

    if cytotrace_key:
        ctk = _build_cytotrace_kernel(adata, cytotrace_key, notes, warnings)
        if ctk is not None:
            directionals.append(("cytotrace", ctk))

    # 3b. VelocityKernel from a whole-object velocity h5ad (opt-in upstream).
    # Requires Ms + velocity layers and 1:1 obs alignment with the working atlas.
    if velocity_adata is not None:
        if "Ms" not in velocity_adata.layers or "velocity" not in velocity_adata.layers:
            warnings.append("velocity kernel skipped: velocity_adata lacks Ms/velocity layers")
        elif list(velocity_adata.obs_names) != list(adata.obs_names):
            warnings.append("velocity kernel skipped: obs_names mismatch with working atlas")
        else:
            try:
                from cellrank.kernels import VelocityKernel

                vk = VelocityKernel(velocity_adata).compute_transition_matrix(
                    model=velocity_model, seed=seed
                )
                directionals.append(("velocity", vk))
            except Exception as exc:  # noqa: BLE001 — drop this kernel, keep going
                warnings.append(f"velocity kernel failed: {exc}")

    # 3c. RealTimeKernel via a moscot TemporalProblem (gated on time_key). Common
    # case/control lymphedema data has no experimental time axis, so this is
    # skipped by default. Requires EVERY cell to carry a finite numeric time and
    # ≥2 distinct levels: a partially-populated time column would yield a
    # subset-shaped kernel that neither combines with the full-atlas connectivity
    # kernel (shape mismatch) nor corresponds to it cell-for-cell.
    if time_key:
        if time_key not in adata.obs:
            warnings.append(f"realtime kernel skipped: time_key '{time_key}' absent")
        else:
            time_numeric = pd.to_numeric(adata.obs[time_key], errors="coerce")
            n_levels = int(time_numeric.dropna().nunique())
            if time_numeric.isna().any():
                n_missing = int(time_numeric.isna().sum())
                warnings.append(
                    f"realtime kernel skipped: time_key '{time_key}' has "
                    f"{n_missing} cell(s) with non-numeric/missing time"
                )
            elif n_levels < 2:
                warnings.append(
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
                    warnings,
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
        "warnings": warnings,
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
    # Every step below the Schur decomposition degrades gracefully, and each
    # degradation removes part of the answer (no terminal states, no fate
    # probabilities, no drivers). That belongs in warnings, not notes.
    warnings: list[str] = []

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
        "warnings": warnings,
    }

    # Terminal states: stability, with a top_n fallback.
    try:
        g.predict_terminal_states(method=terminal_method, n_states=n_terminal_states)
    except ValueError as exc:
        warnings.append(f"predict_terminal_states('{terminal_method}') failed: {exc}; trying top_n")
        try:
            g.predict_terminal_states(method="top_n", n_states=n_terminal_states or 2)
        except Exception as exc2:  # noqa: BLE001 — keep macrostates, skip fate probs
            warnings.append(f"terminal-state prediction failed: {exc2}")
            return result
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"terminal-state prediction failed: {exc}")
        return result

    result["terminal_states"] = [str(x) for x in g.terminal_states.cat.categories]

    # Optional initial states (best-effort).
    if predict_initial_states:
        try:
            g.predict_initial_states(n_states=int(n_initial_states))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"predict_initial_states failed: {exc}")

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
        warnings.append(f"compute_fate_probabilities failed: {exc}")
        return result

    # One terminal state is a RESULT, not an error, and it needs saying out loud:
    # every cell's fate probability is then 1.0, so the fate-probability figure
    # carries no information and cannot be read as "these cells are committed".
    # It happens on genuinely non-branching lineages — the LEC arm produced one
    # terminal state out of eight macrostates.
    #
    # CellRank handles it by correlating genes against the stationary
    # distribution instead of against fate probabilities, but that path needs
    # ``eigendecomposition['stationary_dist']``, which the GPCCA Schur route never
    # populates; without it, drivers fail with "No stationary distribution found
    # in .eigendecomposition['stationary_dist']" — a message that names the
    # missing intermediate rather than the actual cause. So compute it here,
    # which both enables the documented fallback and keeps the warning honest.
    single_lineage = len(result["fate_names"]) == 1
    if single_lineage:
        warnings.append(
            f"only 1 terminal state ({result['fate_names'][0]}) out of "
            f"{len(macro_names)} macrostates: fate probabilities are 1.0 for every "
            "cell and convey no lineage information; drivers fall back to "
            "correlation against the stationary distribution"
        )
        needs_stationary = (g.eigendecomposition or {}).get("stationary_dist") is None
        if needs_stationary:
            try:
                # ARPACK on the transition matrix; k below the component count the
                # Schur step already succeeded with, so this asks for strictly less
                # than what converged there.
                g.compute_eigendecomposition(k=min(20, max(2, n_comp)), only_evals=False)
            except Exception as exc:  # noqa: BLE001 — drivers stay best-effort
                warnings.append(f"compute_eigendecomposition (for driver fallback) failed: {exc}")

    # Lineage drivers (best-effort).
    try:
        result["drivers"] = g.compute_lineage_drivers(cluster_key=cluster_key, seed=seed)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"compute_lineage_drivers failed: {exc}")

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
