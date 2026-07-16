"""Doublet detection with optional cross-method consensus.

Runs one or more doublet detectors (Scrublet in Python, scDblFinder in R via the
Rscript adapter) and combines their per-cell calls by a consensus rule. Doublet
detection FLAGS cells (obs["predicted_doublet"]); removal is a separate QC-filter
decision. Doublet detection is distinct from ambient-RNA correction (SoupX).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from cellquorum.qc.config import QCDoubletConfig

if TYPE_CHECKING:
    from anndata import AnnData

    from cellquorum.backends.rscript import RscriptBackend

# Path to the bundled scDblFinder R script.
_SCDBLFINDER_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "scdblfinder.R"


def run_scrublet(adata: AnnData, *, expected_rate: float, random_state: int) -> np.ndarray:
    """
    Return per-cell Scrublet doublet scores (0..1), or all-NaN if unavailable.

    Args:
        adata: AnnData with a counts layer or raw .X.
        expected_rate: Expected doublet rate.
        random_state: Seed.

    Returns:
        1-D array of per-cell doublet scores.
    """

    # Import scrublet lazily; if absent, return NaN scores (skip, not crash).
    try:
        import scrublet as scr
    except Exception:
        return np.full(adata.n_obs, np.nan, dtype=float)

    # Use counts if present, else .X.
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)

    # Run Scrublet and return scores (suppress internal RuntimeWarning).
    import warnings

    scrub = scr.Scrublet(dense, expected_doublet_rate=expected_rate, random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scrublet")
        scores, _ = scrub.scrub_doublets(verbose=False)
    return np.asarray(scores, dtype=float)


def run_scdblfinder(
    adata: AnnData, backend: RscriptBackend | None, *, random_state: int
) -> np.ndarray:
    """
    Return per-cell scDblFinder scores via the R adapter, or NaN if unavailable.

    Args:
        adata: AnnData with counts.
        backend: RscriptBackend (or None to skip).
        random_state: Seed passed to the R script.

    Returns:
        1-D array of per-cell doublet scores (NaN when R unavailable).
    """

    # No backend or no script -> skip with NaN scores.
    if backend is None or not _SCDBLFINDER_R.is_file():
        return np.full(adata.n_obs, np.nan, dtype=float)

    # Write counts to a temp Matrix Market file, run the R script, read scores.
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    mat = sp.csr_matrix(counts).T  # genes x cells for R
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mtx = tmp_path / "counts.mtx"
        out = tmp_path / "scores.csv"
        sio.mmwrite(str(mtx), mat)
        result = backend.run_script(_SCDBLFINDER_R, [str(mtx), str(out), str(random_state)])
        if result.returncode != 0 or not out.is_file():
            # R failed — skip with NaN (caller records the note).
            return np.full(adata.n_obs, np.nan, dtype=float)
        scores = pd.read_csv(out)["score"].to_numpy(dtype=float)
    return scores


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


def _score_method(
    adata: AnnData,
    method: str,
    backend: RscriptBackend | None,
    *,
    expected_rate: float,
) -> np.ndarray | None:
    """Run one detector on ``adata`` and return per-cell scores (None if unknown)."""

    if method == "scrublet":
        return run_scrublet(adata, expected_rate=expected_rate, random_state=0)
    if method == "scdblfinder":
        return run_scdblfinder(adata, backend, random_state=0)
    return None


def _score_method_per_sample(
    adata: AnnData,
    method: str,
    backend: RscriptBackend | None,
    *,
    expected_rate: float,
    sample_key: str,
) -> tuple[np.ndarray, list[str]]:
    """Run a detector independently per sample and scatter scores back to cell order.

    Doublet detectors model each capture's own doublet structure, so pooling
    libraries biases the neighborhood/kNN estimates. This runs the detector once
    per ``sample_key`` group and places each group's scores back into a
    full-length, cell-order-aligned array (NaN for groups a detector could not
    score).
    """

    scores = np.full(adata.n_obs, np.nan, dtype=float)
    positions = np.arange(adata.n_obs)
    sample_values = adata.obs[sample_key].to_numpy()
    notes: list[str] = []

    for sample in pd.unique(sample_values):
        mask = sample_values == sample
        # AnnData subset by boolean mask, preserving order.
        sub = adata[mask]
        sub_scores = _score_method(sub, method, backend, expected_rate=expected_rate)
        if sub_scores is None or np.all(np.isnan(sub_scores)):
            notes.append(f"doublet method '{method}' unavailable for sample '{sample}'")
            continue
        scores[positions[mask]] = np.asarray(sub_scores, dtype=float)

    return scores, notes


def detect_doublets(
    adata: AnnData,
    config: QCDoubletConfig,
    backend: RscriptBackend | None,
    *,
    sample_key: str | None = None,
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

    Returns:
        Metrics dict (methods run, predicted-doublet count, consensus rule).
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
    notes: list[str] = []
    for method in methods:
        if per_sample:
            scores, sample_notes = _score_method_per_sample(
                adata,
                method,
                backend,
                expected_rate=config.expected_doublet_rate,
                sample_key=sample_key,
            )
            notes.extend(sample_notes)
        else:
            scores = _score_method(
                adata, method, backend, expected_rate=config.expected_doublet_rate
            )

        # Unknown method name: skip.
        if scores is None:
            continue

        adata.obs[f"doublet_score_{method}"] = scores
        # A method that returned all-NaN was unavailable; skip it from consensus.
        if np.all(np.isnan(scores)):
            notes.append(f"doublet method '{method}' unavailable (skipped)")
            continue
        call_cols[method] = pd.Series(scores > threshold, index=adata.obs_names)
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
        "n_predicted_doublets": int(np.nansum(adata.obs["predicted_doublet"].to_numpy())),
        "notes": notes,
    }


__all__ = ["combine_consensus", "detect_doublets", "run_scdblfinder", "run_scrublet"]
