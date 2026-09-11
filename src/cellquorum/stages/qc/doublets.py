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


_SCDBLFINDER_R = r_script_path("scdblfinder.R")


def run_scrublet(
    adata: AnnData, *, expected_rate: float, random_state: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return Scrublet scores and native calls; unavailable scores are NaN.

    Scrublet computes a data-driven threshold from the bimodal simulated-doublet score
    histogram and returns a boolean ``predicted_doublets`` from THAT threshold. We return
    it as the native call so the caller does not have to re-threshold the score with an
    arbitrary cut (the historical ``score > 0.5`` never fired because observed scores
    ceiling near 0.5).
    """

    try:
        import scrublet as scr
    except Exception:
        return np.full(adata.n_obs, np.nan, dtype=float), None

    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    matrix = counts if sp.issparse(counts) else np.asarray(counts)

    import warnings

    scrub = scr.Scrublet(matrix, expected_doublet_rate=expected_rate, random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scrublet")
        scores, predicted = scrub.scrub_doublets(verbose=False)

    calls = None if predicted is None else np.asarray(predicted, dtype=bool)
    return np.asarray(scores, dtype=float), calls


def run_scdblfinder(
    adata: AnnData,
    backend: RscriptBackend | None,
    *,
    random_state: int,
    expected_rate: float | None = None,
    sample_key: str | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run scDblFinder once, verify cell IDs, and return scores and native calls.

    scDblFinder assigns each cell a ``scDblFinder.class`` (singlet/doublet) using its OWN
    calibrated threshold — the correct call to use, rather than re-thresholding
    ``scDblFinder.score`` at an arbitrary cut. ``sample_key``, when given, is handed to
    scDblFinder's own ``samples=`` argument, which searches for doublets independently
    within each capture — the same treatment as calling this function once per sample, in
    one R session instead of one per sample. ``n_jobs`` caps the workers R may use to score
    captures concurrently; the R side seeds the worker RNG streams, so the calls do not
    depend on the core count.
    """

    if backend is None or not _SCDBLFINDER_R.is_file():
        return np.full(adata.n_obs, np.nan, dtype=float), None

    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    mat = sp.csr_matrix(counts).T
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mtx = tmp_path / "counts.mtx"
        out = tmp_path / "scores.csv"
        sio.mmwrite(str(mtx), mat)
        argv = [str(mtx), str(out), str(random_state)]
        if sample_key is not None:
            samples = tmp_path / "samples.csv"

            # Written as strings: sample labels are often numeric-looking codes, and R
            # would otherwise read "1" and "01" as the same capture.
            labels = adata.obs[sample_key].astype(str).to_numpy()
            pd.DataFrame({"sample": labels}).to_csv(samples, index=False)
            argv.append(str(samples))

            # Capped at the capture count: extra workers would sit idle holding a fork of
            # the session.
            threads = max(1, min(int(n_jobs), int(pd.unique(labels).size)))
            argv.append(str(threads))
        if expected_rate is not None:
            while len(argv) < 5:
                argv.append("")
            argv.append(str(expected_rate))
        result = backend.run_script(_SCDBLFINDER_R, argv)
        if result.returncode != 0 or not out.is_file():
            # R failed — skip with NaN (caller records the note).
            return np.full(adata.n_obs, np.nan, dtype=float), None
        frame = pd.read_csv(out, dtype={"cell_id": str})
        if not {"cell_id", "score"} <= set(frame.columns):
            raise ValueError("scDblFinder output requires cell_id and score columns")
        expected_ids = [str(i) for i in range(1, adata.n_obs + 1)]
        if frame["cell_id"].duplicated().any() or set(frame["cell_id"]) != set(expected_ids):
            raise ValueError("scDblFinder output cell IDs do not match the input cells")
        frame = frame.set_index("cell_id").loc[expected_ids]
        scores = frame["score"].to_numpy(dtype=float)

        if "class" in frame.columns:
            labels = frame["class"].astype("string").str.lower()
            if not labels.isin(["singlet", "doublet"]).all():
                raise ValueError("scDblFinder output contains missing or unknown class labels")
            calls = (labels == "doublet").to_numpy(dtype=bool)
        else:
            calls = None
    return scores, calls


def combine_consensus(calls: pd.DataFrame, rule: str) -> pd.Series:
    """Combine boolean detector calls using any, all, or strict majority."""

    if rule == "any":
        return calls.any(axis=1)
    if rule == "all":
        return calls.all(axis=1)
    if rule == "majority":
        return calls.sum(axis=1) > (calls.shape[1] / 2)
    raise ValueError(f"Unknown consensus rule '{rule}'. Use any|all|majority.")


#: Detectors that split by capture themselves, given the sample column. For these the
#: stage hands the whole object over once instead of driving the split from Python -- same
#: per-sample statistics, one process launch instead of one per sample. Anything not listed
#: here gets the generic loop in ``_score_method_per_sample``.
_NATIVE_PER_SAMPLE = frozenset({"scdblfinder"})


def _score_method(
    adata: AnnData,
    method: str,
    backend: RscriptBackend | None,
    *,
    expected_rate: float,
    random_state: int = 0,
    sample_key: str | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run one detector and return ``(scores, native_calls)`` (None if unknown method).

    ``native_calls`` is a float array aligned to ``scores`` holding the detector's OWN
    doublet call (1.0 doublet / 0.0 singlet) from its calibrated threshold, with NaN where
    the detector provides no native call — the caller then falls back to the score
    threshold for those cells. ``sample_key`` and ``n_jobs`` are forwarded to detectors in
    ``_NATIVE_PER_SAMPLE`` so they can do the per-capture split internally and parallelize
    it; they are ignored by the others, which the caller splits for them one at a time.
    """

    if method == "scrublet":
        scores, native = run_scrublet(adata, expected_rate=expected_rate, random_state=random_state)
    elif method == "scdblfinder":
        scores, native = run_scdblfinder(
            adata,
            backend,
            random_state=random_state,
            expected_rate=expected_rate,
            sample_key=sample_key,
            n_jobs=n_jobs,
        )
    else:
        return None

    scores = np.asarray(scores, dtype=float)
    if scores.shape != (adata.n_obs,):
        raise ValueError(
            f"{method} returned scores with shape {scores.shape}; expected {(adata.n_obs,)}"
        )
    if np.isinf(scores).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{method} returned scores outside [0, 1]")
    native_f = np.full(scores.shape[0], np.nan, dtype=float)
    if native is not None:
        native_values = np.asarray(native, dtype=float)
        if native_values.shape != scores.shape:
            raise ValueError(
                f"{method} returned calls with shape {native_values.shape}; expected {scores.shape}"
            )
        if not np.isin(native_values[~np.isnan(native_values)], [0, 1]).all():
            raise ValueError(f"{method} returned non-boolean calls")
        native_f[:] = native_values
    return scores, native_f


def _score_method_per_sample(
    adata: AnnData,
    method: str,
    backend: RscriptBackend | None,
    *,
    expected_rate: float,
    random_state: int = 0,
    sample_key: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run a detector independently per sample and scatter results back to cell order.

    Doublet detectors model each capture's own doublet structure, so pooling libraries
    biases the neighborhood/kNN estimates. This runs the detector once per ``sample_key``
    group and places each group's scores AND native calls back into full-length,
    cell-order-aligned arrays (NaN for groups a detector could not score / did not provide
    a native call).
    """

    scores = np.full(adata.n_obs, np.nan, dtype=float)
    native = np.full(adata.n_obs, np.nan, dtype=float)
    positions = np.arange(adata.n_obs)
    sample_values = adata.obs[sample_key].to_numpy()

    warnings: list[str] = []

    for sample in pd.unique(sample_values):
        mask = sample_values == sample

        sub = adata[mask]
        result = _score_method(
            sub, method, backend, expected_rate=expected_rate, random_state=random_state
        )
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
    random_state: int = 0,
) -> dict:
    """Score configured detectors, annotate calls and coverage, and return provenance."""

    methods = list(config.methods) if config.methods else [config.method]
    threshold = config.score_threshold if config.score_threshold is not None else 0.5

    per_sample = bool(config.per_sample) and sample_key is not None
    if per_sample:
        if sample_key not in adata.obs:
            raise ValueError(f"Doublet sample column {sample_key!r} is missing")
        labels = adata.obs[sample_key]
        if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError(f"Doublet sample column {sample_key!r} contains missing or blank IDs")
        if labels.nunique() != labels.astype(str).nunique():
            raise ValueError(
                f"Doublet sample column {sample_key!r} has IDs that collide as strings"
            )
    scored_scope = "per_sample" if per_sample else "pooled"

    call_cols: dict[str, pd.Series] = {}
    methods_run: list[str] = []
    used_native: dict[str, bool] = {}
    measured_cells: dict[str, int] = {}

    notes: list[str] = []
    warnings: list[str] = []
    for method in methods:
        if per_sample and method not in _NATIVE_PER_SAMPLE:
            scores, native, sample_warnings = _score_method_per_sample(
                adata,
                method,
                backend,
                expected_rate=config.expected_doublet_rate,
                random_state=random_state,
                sample_key=sample_key,
            )
            warnings.extend(sample_warnings)
        else:
            result = _score_method(
                adata,
                method,
                backend,
                expected_rate=config.expected_doublet_rate,
                random_state=random_state,
                sample_key=sample_key if per_sample else None,
                n_jobs=n_jobs,
            )
            if per_sample and method in _NATIVE_PER_SAMPLE:
                notes.append(
                    f"doublet method '{method}' split by capture itself "
                    f"({sample_key}), in one process"
                )

            if result is None:
                continue
            scores, native = result

        adata.obs[f"doublet_score_{method}"] = scores

        scored_mask = ~np.isnan(scores)
        measured_cells[method] = int(scored_mask.sum())
        adata.obs[f"doublet_measured_{method}"] = scored_mask
        if not scored_mask.any():
            warnings.append(f"doublet method '{method}' unavailable (skipped)")
            continue
        if not scored_mask.all():
            warnings.append(
                f"doublet method '{method}' left {int((~scored_mask).sum())} cells unscored; "
                "negative flags for these cells are not measured singlet calls"
            )

        native_mask = ~np.isnan(native) & scored_mask
        if config.score_threshold is not None:
            native_mask[:] = False
        calls = np.zeros(adata.n_obs, dtype=bool)
        if native_mask.any():
            calls[native_mask] = native[native_mask] > 0.5
        score_only = scored_mask & ~native_mask
        if score_only.any():
            calls[score_only] = scores[score_only] >= threshold
        used_native[method] = bool(native_mask.any())

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

    if call_cols:
        call_frame = pd.DataFrame(call_cols)
        adata.obs["predicted_doublet"] = combine_consensus(call_frame, config.consensus).to_numpy()
        score_cols = [f"doublet_score_{m}" for m in methods_run]

        adata.obs["doublet_score"] = adata.obs[score_cols].max(axis=1, skipna=True).to_numpy()
    else:
        adata.obs["predicted_doublet"] = False
        adata.obs["doublet_score"] = np.nan

    return {
        "methods_run": methods_run,
        "random_state": random_state,
        "expected_doublet_rate": config.expected_doublet_rate,
        "consensus": config.consensus,
        "scored_scope": scored_scope,
        "sample_key": sample_key if per_sample else None,
        "used_native_calls": used_native,
        "measured_cells": measured_cells,
        "score_threshold": threshold,
        "threshold_policy": "explicit"
        if config.score_threshold is not None
        else "native_or_fallback",
        "n_predicted_doublets": int(np.nansum(adata.obs["predicted_doublet"].to_numpy())),
        "notes": notes,
        "warnings": warnings,
    }


__all__ = ["combine_consensus", "detect_doublets", "run_scdblfinder", "run_scrublet"]
