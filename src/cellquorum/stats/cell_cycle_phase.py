"""Categorical G1/S/G2M call from cell-cycle signature scores.

Matches scanpy's own ``score_genes_cell_cycle`` convention exactly, so a
phase computed here reads the same way a reader already expects: both
scores non-positive means no cell-cycle signature dominates (G1); otherwise
whichever of S/G2M scores higher wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def assign_cell_cycle_phase(s_score: pd.Series, g2m_score: pd.Series) -> pd.Series:
    """
    Derive a categorical cell-cycle phase from S/G2M signature scores.

    Args:
        s_score: Per-cell S-phase signature score.
        g2m_score: Per-cell G2M-phase signature score, same index as ``s_score``.

    Returns:
        Series of ``"G1"`` / ``"S"`` / ``"G2M"``, indexed like ``s_score``,
        named ``"phase"``.
    """
    phase = np.where(
        (s_score <= 0) & (g2m_score <= 0),
        "G1",
        np.where(s_score > g2m_score, "S", "G2M"),
    )
    return s_score.__class__(phase, index=s_score.index, name="phase")


__all__ = ["assign_cell_cycle_phase"]
