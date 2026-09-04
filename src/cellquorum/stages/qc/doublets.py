"""Doublet detection with optional cross-method consensus.

Runs one or more doublet detectors (Scrublet in Python, scDblFinder in R via the
Rscript adapter) and combines their per-cell calls by a consensus rule. Doublet
detection FLAGS cells (obs["predicted_doublet"]); removal is a separate QC-filter
decision. Doublet detection is distinct from ambient-RNA correction (SoupX).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from cellquorum.backends.script_paths import r_script_path
from cellquorum.stages.qc.config import QCDoubletConfig

if TYPE_CHECKING:
    from anndata import AnnData

    from cellquorum.backends.rscript import RscriptBackend

logger = logging.getLogger(__name__)

# Path to the bundled scDblFinder R script.
_SCDBLFINDER_R = r_script_path("scdblfinder.R")


def run_scrublet(
    adata: AnnData, *, expected_rate: float, random_state: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return per-cell Scrublet doublet scores AND Scrublet's own doublet calls.

    Scrublet computes a data-driven threshold from the bimodal simulated-doublet
    score histogram and returns a boolean ``predicted_doublets`` from THAT
    threshold. We return it as the native call so the caller does not have to
    re-threshold the score with an arbitrary cut (the historical
    ``score > 0.5`` never fired because observed scores ceiling near 0.5).

    Args:
        adata: AnnData with a counts layer or raw .X.
        expected_rate: Expected doublet rate.
        random_state: Seed.

    Returns:
        ``(scores, calls)`` where ``scores`` is the 1-D per-cell doublet score
        array (all-NaN if Scrublet is unavailable) and ``calls`` is Scrublet's
        boolean per-cell doublet call, or ``None`` when Scrublet could not
        auto-detect a threshold (caller falls back to the score threshold).
    """

    # Import scrublet lazily; if absent, return NaN scores + no native call.
    try:
        import scrublet as scr
    except Exception:
        return np.full(adata.n_obs, np.nan, dtype=float), None

    # Use counts if present, else .X.
    #
    # Hand Scrublet the matrix in whatever form it already has. Its own signature
    # takes "scipy sparse matrix or ndarray" and immediately converts to CSC, so
    # densifying here only allocated a full n_cells x n_genes array for Scrublet to
    # compress straight back — and on a large sample that allocation is the thing
    # that fails, not the doublet detection.
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    matrix = counts if sp.issparse(counts) else np.asarray(counts)

    # Run Scrublet and return scores + its own calls (suppress internal RuntimeWarning).
    import warnings

    scrub = scr.Scrublet(matrix, expected_doublet_rate=expected_rate, random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scrublet")
        scores, predicted = scrub.scrub_doublets(verbose=False)
    # ``predicted`` is None when Scrublet could not auto-detect a threshold.
    calls = None if predicted is None else np.asarray(predicted, dtype=bool)
    return np.asarray(scores, dtype=float), calls


def run_scdblfinder(
    adata: AnnData,
    backend: RscriptBackend | None,
    *,
    random_state: int,
    sample_key: str | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return per-cell scDblFinder scores AND scDblFinder's own doublet class calls.

    scDblFinder assigns each cell a ``scDblFinder.class`` (singlet/doublet) using
    its OWN calibrated threshold — the correct call to use, rather than
    re-thresholding ``scDblFinder.score`` at an arbitrary cut. The R adapter emits
    both columns; the class call is returned as the native boolean call.

    Args:
        adata: AnnData with counts.
        backend: RscriptBackend (or None to skip).
        random_state: Seed passed to the R script.
        sample_key: obs column naming each cell's capture. When given, it is
            handed to scDblFinder's own ``samples=`` argument, which searches for
            doublets independently within each capture — the same treatment as
            calling this function once per sample, in one R session instead of
            one per sample. See the note on wall time in ``scdblfinder.R``.
        n_jobs: Workers R may use to score the captures concurrently. Only has an
            effect alongside ``sample_key``, since the per-capture split is the
            only thing there is to parallelize, and is capped at the number of
            captures. The R side seeds the worker RNG streams, so the calls do not
            depend on the core count.

    Returns:
        ``(scores, calls)`` where ``scores`` is the 1-D per-cell score array
        (all-NaN when R is unavailable) and ``calls`` is the boolean per-cell
        ``class == "doublet"`` call, or ``None`` when the adapter emitted only a
        score column (caller falls back to the score threshold).
    """

    # No backend or no script -> skip with NaN scores + no native call.
    if backend is None or not _SCDBLFINDER_R.is_file():
        return np.full(adata.n_obs, np.nan, dtype=float), None

    # Write counts to a temp Matrix Market file, run the R script, read results.
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    mat = sp.csr_matrix(counts).T  # genes x cells for R
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mtx = tmp_path / "counts.mtx"
        out = tmp_path / "scores.csv"
        sio.mmwrite(str(mtx), mat)
        argv = [str(mtx), str(out), str(random_state)]
        if sample_key is not None:
            samples = tmp_path / "samples.csv"
            # Written as strings: sample labels are often numeric-looking codes,
            # and R would otherwise read "1" and "01" as the same capture.
            labels = adata.obs[sample_key].astype(str).to_numpy()
            pd.DataFrame({"sample": labels}).to_csv(samples, index=False)
            argv.append(str(samples))
            # Capped at the capture count: extra workers would sit idle holding a
            # fork of the session.
            threads = max(1, min(int(n_jobs), int(pd.unique(labels).size)))
            argv.append(str(threads))
        result = backend.run_script(_SCDBLFINDER_R, argv)
        if result.returncode != 0 or not out.is_file():
            # R failed — skip with NaN (caller records the note).
            return np.full(adata.n_obs, np.nan, dtype=float), None
        frame = pd.read_csv(out)
        scores = frame["score"].to_numpy(dtype=float)
        # Prefer the native class call; fall back to score-only for older outputs.
        if "class" in frame.columns:
            calls = (frame["class"].astype(str).str.lower() == "doublet").to_numpy(dtype=bool)
        else:
            calls = None
    return scores, calls


def combine_consensus(calls: pd.DataFrame, rule: str) -> pd.Series:
    """
    Combine per-method boolean doublet calls by a consensus rule.

    Args:
        calls: DataFrame of boolean columns (one per method).
        rule: "any" | "all" | "majority".

    Returns:
        Boolean Series of combined calls.
    """

    # Combine according to the rule.
    if rule == "any":
        return calls.any(axis=1)
    if rule == "all":
        return calls.all(axis=1)
    if rule == "majority":
        return calls.sum(axis=1) > (calls.shape[1] / 2)
    raise ValueError(f"Unknown consensus rule '{rule}'. Use any|all|majority.")


#: Detectors that split by capture themselves, given the sample column. For these
#: the stage hands the whole object over once instead of driving the split from
#: Python -- same per-sample statistics, one process launch instead of one per
#: sample. Anything not listed here gets the generic loop in
#: ``_score_method_per_sample``.
_NATIVE_PER_SAMPLE = frozenset({"scdblfinder"})


def _score_method(
    adata: AnnData,
    method: str,
    backend: RscriptBackend | None,
    *,
    expected_rate: float,
    sample_key: str | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run one detector and return ``(scores, native_calls)`` (None if unknown method).

    ``scores`` is the per-cell doublet score (NaN where not scored). ``native_calls``
    is a float array aligned to ``scores`` holding the detector's OWN doublet call
    (1.0 doublet / 0.0 singlet) from its calibrated threshold, with NaN where the
    detector provides no native call — the caller then falls back to the score
    threshold for those cells.

    ``sample_key`` and ``n_jobs`` are forwarded to detectors in
    ``_NATIVE_PER_SAMPLE`` so they can do the per-capture split internally, and
    parallelize it; they are ignored by the others, which the caller splits for
    them one at a time.
    """

    if method == "scrublet":
        scores, native = run_scrublet(adata, expected_rate=expected_rate, random_state=0)
    elif method == "scdblfinder":
        scores, native = run_scdblfinder(
            adata, backend, random_state=0, sample_key=sample_key, n_jobs=n_jobs
        )
    else:
        return None

    scores = np.asarray(scores, dtype=float)
    native_f = np.full(scores.shape[0], np.nan, dtype=float)
    if native is not None:
        native_f[:] = np.asarray(native, dtype=float)
    return scores, native_f


def _score_method_per_sample(
    adata: AnnData,
    method: str,
    backend: RscriptBackend | None,
    *,
    expected_rate: float,
    sample_key: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run a detector independently per sample and scatter results back to cell order.

    Doublet detectors model each capture's own doublet structure, so pooling
    libraries biases the neighborhood/kNN estimates. This runs the detector once
    per ``sample_key`` group and places each group's scores AND native calls back
    into full-length, cell-order-aligned arrays (NaN for groups a detector could
    not score / did not provide a native call).
    """

    scores = np.full(adata.n_obs, np.nan, dtype=float)
    native = np.full(adata.n_obs, np.nan, dtype=float)
    positions = np.arange(adata.n_obs)
    sample_values = adata.obs[sample_key].to_numpy()
    # Warnings, not notes: every message this loop can emit means a detector the
    # config named produced no scores for some or all captures, and the consensus
    # call is then built from fewer methods than the config asked for.
    warnings: list[str] = []

    for sample in pd.unique(sample_values):
        mask = sample_values == sample
        # AnnData subset by boolean mask, preserving order.
        sub = adata[mask]
        result = _score_method(sub, method, backend, expected_rate=expected_rate)
        if result is None:
            warnings.append(f"doublet method '{method}' is unknown (skipped)")
            continue
        sub_scores, sub_native = result
        if np.all(np.isnan(sub_scores)):
            warnings.append(f"doublet method '{method}' unavailable for sample '{sample}'")
            continue
        scores[positions[mask]] = sub_scores
        native[positions[mask]] = sub_native

    return scores, native, warnings


def detect_doublets(
    adata: AnnData,
    config: QCDoubletConfig,
    backend: RscriptBackend | None,
    *,
    sample_key: str | None = None,
    n_jobs: int = 1,
) -> dict:
    """
    Run configured doublet detectors and write flags/scores to obs.

    When ``config.per_sample`` is True and a ``sample_key`` is available, each
    detector runs independently per sample/library (the correct scDblFinder /
    Scrublet practice), then scores are combined across the full object. Falls
    back to pooled detection when no sample key is available.

    Args:
        adata: AnnData with counts.
        config: Doublet configuration (methods + consensus).
        backend: RscriptBackend for scDblFinder (or None).
        sample_key: obs column identifying the sample/library for per-sample
            detection. None disables per-sample detection.
        n_jobs: Worker count for detectors that split by capture themselves
            (``_NATIVE_PER_SAMPLE``). The Python-side loop stays serial.

    Returns:
        Metrics dict (methods run, predicted-doublet count, consensus rule), plus
        a ``notes`` list and a ``warnings`` list. The caller is expected to lift
        both onto its StageResult: left in the metrics dict they reach provenance
        JSON only, and the one message here that most needs a reader — a detector
        that scored cells and flagged zero doublets — would never appear in the
        run report at all.
    """

    # Resolve which detectors to run (prefer the list; fall back to single method).
    methods = list(config.methods) if config.methods else [config.method]
    threshold = config.score_threshold if config.score_threshold is not None else 0.5

    # Decide whether to run per-sample: opt-in flag AND a usable sample column.
    per_sample = bool(config.per_sample) and sample_key is not None and sample_key in adata.obs
    scored_scope = "per_sample" if per_sample else "pooled"

    # Run each detector, storing per-method scores and a boolean call.
    call_cols: dict[str, pd.Series] = {}
    methods_run: list[str] = []
    used_native: dict[str, bool] = {}
    # Two channels, deliberately. A note records HOW a detector ran; a warning
    # records a detector the config asked for whose scores are missing or whose
    # calls are suspect. The QC stage lifts the second into StageResult.warnings,
    # which the run report prints and counts — the first it does not.
    notes: list[str] = []
    warnings: list[str] = []
    for method in methods:
        if per_sample and method not in _NATIVE_PER_SAMPLE:
            scores, native, sample_warnings = _score_method_per_sample(
                adata,
                method,
                backend,
                expected_rate=config.expected_doublet_rate,
                sample_key=sample_key,
            )
            warnings.extend(sample_warnings)
        else:
            result = _score_method(
                adata,
                method,
                backend,
                expected_rate=config.expected_doublet_rate,
                # None when pooled, so the detector sees one capture -- which is
                # what "pooled" means -- and the same call site serves both modes.
                sample_key=sample_key if per_sample else None,
                n_jobs=n_jobs,
            )
            if per_sample and method in _NATIVE_PER_SAMPLE:
                # Provenance, not a warning: "per_sample" is true of both the
                # Python-side loop and this path, and a reader comparing two runs
                # across the change should be able to tell which one produced the
                # numbers. Agreement between the two on the LEC arm was r=0.978
                # per cell and r=0.998 on per-capture mean scores.
                notes.append(
                    f"doublet method '{method}' split by capture itself "
                    f"({sample_key}), in one process"
                )
            # Unknown method name: skip.
            if result is None:
                continue
            scores, native = result

        adata.obs[f"doublet_score_{method}"] = scores
        # A method that scored no cells (all-NaN) was unavailable; skip it.
        scored_mask = ~np.isnan(scores)
        if not scored_mask.any():
            warnings.append(f"doublet method '{method}' unavailable (skipped)")
            continue

        # Build the per-cell call: use the detector's OWN call where it gave one
        # (native, from its calibrated threshold), and fall back to the score
        # threshold only for cells with no native call. Use `>=` (not `>`) so a
        # score at the ceiling still flags — the historical `> 0.5` never fired
        # because observed scores ceiling at 0.5.
        native_mask = ~np.isnan(native)
        calls = np.zeros(adata.n_obs, dtype=bool)
        if native_mask.any():
            calls[native_mask] = native[native_mask] > 0.5
        score_only = scored_mask & ~native_mask
        if score_only.any():
            calls[score_only] = scores[score_only] >= threshold
        used_native[method] = bool(native_mask.any())

        # No-silent-decisions guard: a detector that scored cells but flagged
        # zero doublets almost always means a broken threshold — say so loudly.
        if int(calls.sum()) == 0:
            msg = (
                f"doublet method '{method}' scored {int(scored_mask.sum())} cells "
                f"but flagged 0 doublets (native calls: {used_native[method]}, "
                f"score threshold: {threshold}). Check the detector/threshold."
            )
            logger.warning(msg)
            warnings.append(msg)

        call_cols[method] = pd.Series(calls, index=adata.obs_names)
        methods_run.append(method)

    # Combine calls into predicted_doublet + a summary doublet_score (max).
    if call_cols:
        calls = pd.DataFrame(call_cols)
        adata.obs["predicted_doublet"] = combine_consensus(calls, config.consensus).to_numpy()
        score_cols = [f"doublet_score_{m}" for m in methods_run]
        # skipna=True so a per-cell NaN from one method (e.g. a cell scDblFinder
        # could not score) does not contaminate the summary when another method
        # scored that cell successfully.
        adata.obs["doublet_score"] = adata.obs[score_cols].max(axis=1, skipna=True).to_numpy()
    else:
        adata.obs["predicted_doublet"] = False
        adata.obs["doublet_score"] = np.nan

    # Return provenance metrics.
    return {
        "methods_run": methods_run,
        "consensus": config.consensus,
        "scored_scope": scored_scope,
        "sample_key": sample_key if per_sample else None,
        "used_native_calls": used_native,
        "n_predicted_doublets": int(np.nansum(adata.obs["predicted_doublet"].to_numpy())),
        "notes": notes,
        "warnings": warnings,
    }


__all__ = ["combine_consensus", "detect_doublets", "run_scdblfinder", "run_scrublet"]
