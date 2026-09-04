"""One place that knows how to call harmonypy.

Two stages Harmony-correct an embedding: ``integration`` corrects the whole
object's PCA, and ``subclustering`` re-embeds the focus subset before CHOIR
partitions it. Both had their own copy of the same four decisions — silence the
INFO logger, unwrap a torch ``Z_corr``, pick the orientation, choose an iteration
cap — and the copies had already drifted: only one of them passed a cap at all,
and neither reported whether Harmony actually finished.

That last point is why this module exists rather than a shared snippet. harmonypy
reports non-convergence as ``Stopped before convergence`` at INFO level, and both
call sites raised the harmonypy logger to WARNING for the duration of the call, so
the code that most needed the signal was the code suppressing it. On the
lec_mechanotransduction arms this was not hypothetical: both LEC and BEC hit the
default 10-iteration cap and stopped short, converging only at 16 and 15
iterations respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

# harmonypy's own default. Named here so both call sites inherit one value.
DEFAULT_MAX_ITER_HARMONY = 10


@dataclass(frozen=True)
class HarmonyDiagnostics:
    """What happened inside a Harmony run, as numbers rather than log lines.

    Attributes:
        n_iter: Harmony iterations actually performed.
        converged: Whether harmonypy's own convergence test passed.
        max_iter: The cap the run was given, so a warning can name the knob.
    """

    n_iter: int
    converged: bool
    max_iter: int

    @property
    def message(self) -> str | None:
        """A ready-to-emit warning, or None when the run converged."""
        if self.converged:
            return None
        return (
            f"Harmony stopped before convergence after {self.n_iter} iteration(s) "
            f"at max_iter_harmony={self.max_iter}; the corrected embedding is only "
            f"partially batch-corrected and every stage that reads it inherits that."
        )


def harmony_correct(
    pcs: np.ndarray,
    batch: pd.Series | np.ndarray,
    batch_key: str,
    *,
    random_state: int = 0,
    max_iter_harmony: int = DEFAULT_MAX_ITER_HARMONY,
) -> tuple[np.ndarray, HarmonyDiagnostics]:
    """Harmony-correct ``pcs``, oriented ``(n_cells, n_pcs)``, plus diagnostics.

    Args:
        pcs: The ``(n_cells, n_pcs)`` embedding to correct.
        batch: Per-cell batch labels, one per row of ``pcs``.
        batch_key: Name to give the batch column in the metadata frame.
        random_state: Seed passed to harmonypy.
        max_iter_harmony: Iteration cap.

    Returns:
        ``(corrected, diagnostics)``. ``corrected`` always matches ``pcs``'s shape:
        harmonypy returns ``Z_corr`` transposed on some builds, and the scanpy
        wrapper's mishandling of that is what silently left a PCA uncorrected, so
        the orientation is resolved by matching the cell count rather than assumed.

    Raises:
        ImportError: harmonypy is not installed. Raised rather than returned so a
            caller that can proceed uncorrected has to say so explicitly.
        ValueError: The output matches neither orientation of the input, which
            would mean writing an embedding whose rows are not these cells.
    """
    import harmonypy

    pcs = np.ascontiguousarray(pcs)
    meta = pd.DataFrame({batch_key: np.asarray(batch, dtype=object)})

    # Silence harmonypy's per-iteration INFO chatter for the duration of the call
    # only, then restore the prior level: this is a library-wide logger and a stage
    # must not leave process-wide logging state changed behind it.
    harmony_logger = logging.getLogger("harmonypy")
    original_level = harmony_logger.level
    harmony_logger.setLevel(logging.WARNING)
    try:
        harmony_obj = harmonypy.run_harmony(
            pcs,
            meta,
            [batch_key],
            random_state=random_state,
            max_iter_harmony=max_iter_harmony,
        )
    finally:
        harmony_logger.setLevel(original_level)

    # ``objective_harmony`` gets one BASELINE entry from init_cluster before the
    # loop and one per iteration from cluster(), so the iteration count is the
    # length minus that baseline — not the length.
    objective = list(getattr(harmony_obj, "objective_harmony", []) or [])
    n_iter = max(len(objective) - 1, 0)
    try:
        # harmonypy's own test, re-run: it only reads the last two objective
        # values, so it is pure, and taking the verdict from the library means the
        # engine cannot disagree with the algorithm about whether it finished.
        converged = bool(harmony_obj.check_convergence(1))
    except Exception:  # noqa: BLE001
        # Other builds may not expose it; infer from the cap instead.
        converged = n_iter < max_iter_harmony

    corrected = harmony_obj.Z_corr
    if hasattr(corrected, "detach"):
        corrected = corrected.detach()
    if hasattr(corrected, "cpu"):
        corrected = corrected.cpu()
    corrected = np.asarray(corrected)

    if corrected.shape == pcs.shape:
        oriented = corrected
    elif corrected.T.shape == pcs.shape:
        oriented = corrected.T
    else:
        raise ValueError(
            f"Harmony output shape {corrected.shape} matches neither {pcs.shape} "
            f"nor its transpose; refusing to return a mis-oriented embedding."
        )

    return np.ascontiguousarray(oriented), HarmonyDiagnostics(
        n_iter=n_iter, converged=converged, max_iter=max_iter_harmony
    )


__all__ = ["DEFAULT_MAX_ITER_HARMONY", "HarmonyDiagnostics", "harmony_correct"]
