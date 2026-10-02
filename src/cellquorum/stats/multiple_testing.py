"""Multiple-testing correction shared by every stage that adjusts a family of p-values."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def bh_fdr(pvalues: np.ndarray | list[float], method: str = "fdr_bh") -> np.ndarray:
    """Benjamini-Hochberg FDR that tolerates NaN p-values.

    NaNs (a test family member that could not be computed) are held out of the
    correction and returned as NaN in place, so a single un-fittable row never
    drags the whole family's q-values around. Finite p-values are corrected
    among themselves exactly as :func:`statsmodels.stats.multitest.multipletests`
    would on that subset.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = multipletests(p[finite], method=method)[1]
    return out
