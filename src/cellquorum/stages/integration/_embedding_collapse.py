"""Shared degenerate-embedding check for the VAE-based integration methods.

Both scVI and scANVI train a latent space and can produce a degenerate one -- most
dimensions constant, carrying no information -- without the training itself erroring.
A collapsed embedding still "looks like" a normal array: right shape, finite values,
usable by every downstream stage, and silently wrong. This is checked in one place so
scVI and scANVI cannot drift on the thresholds or on how the result is surfaced.
"""

from __future__ import annotations

import numpy as np

from cellquorum.core.exceptions import CellQuorumStageError

#: Below this many variable dimensions, the embedding carries essentially no structure.
_MIN_VARIABLE_DIMENSIONS = 2


def check_embedding_collapse(latent: np.ndarray, n_latent: int, *, method_name: str) -> str | None:
    """Raise on fatal collapse, else return a partial-collapse warning or None.

    Args:
        latent: The trained model's latent representation, cells x n_latent.
        n_latent: The configured latent dimensionality (== latent.shape[1]).
        method_name: "scVI" or "scANVI", for the message.

    Returns:
        A warning message for a partial collapse, or None when the embedding looks healthy.
        Intended for the caller's ``StageResult.warnings`` -- never ``warnings.warn``, which
        nothing in this pipeline's reporting captures.

    Raises:
        CellQuorumStageError: If fewer than 2 dimensions carry any variance at all.
    """
    stds = np.std(latent, axis=0)
    n_variable = int((stds > 1e-6).sum())

    if n_variable < _MIN_VARIABLE_DIMENSIONS:
        raise CellQuorumStageError(
            "integration",
            f"{method_name} embedding collapsed: only {n_variable}/{latent.shape[1]} "
            f"dimensions have variance (std > 1e-6). Training produced a degenerate "
            f"manifold. First 10 stds: {stds[:10].tolist()}. Check: batch key has >1 "
            f"batch, counts layer is not empty, HVG selection didn't fail.",
        )

    if n_variable < n_latent // 2:
        return (
            f"{method_name} embedding has low effective dimensionality: only "
            f"{n_variable}/{n_latent} dimensions have variance. This may indicate a "
            f"problem with the data or training."
        )

    return None


__all__ = ["check_embedding_collapse"]
