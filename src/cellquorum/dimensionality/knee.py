"""Select the number of principal components from the variance-ratio curve.

`n_pcs: auto` picks the elbow of the descending per-PC variance-ratio curve via
the kneedle algorithm, rather than hardcoding a component count. The chosen value
is recorded in provenance by the stage so the choice is auditable.
"""

from __future__ import annotations

import numpy as np
from kneed import KneeLocator


def select_n_pcs(variance_ratio: np.ndarray, *, max_pcs: int) -> int:
    """
    Return the elbow component count from a descending variance-ratio curve.

    NOTE: the kneedle elbow of a scRNA-seq variance curve is known to UNDER-select
    (steep-then-flat curves put max-curvature very low). Hypothesis configs
    therefore set an explicit ``n_pcs`` (typically 50, matching field practice)
    rather than relying on ``auto``. A more principled ``auto`` (Marchenko-Pastur
    noise threshold / parallel analysis) is a possible future replacement.

    Args:
        variance_ratio: Per-PC explained-variance ratios, descending.
        max_pcs: Upper bound on the returned count.

    Returns:
        A component count in [1, min(len(variance_ratio), max_pcs)].
    """

    # Bound the search to the available components and the configured cap.
    n_available = int(len(variance_ratio))
    cap = max(1, min(n_available, int(max_pcs)))

    # A knee needs at least three points; below that, use the cap.
    if n_available < 3:
        return cap

    # x is 1-based component index; y is the (capped) variance-ratio curve.
    x = np.arange(1, cap + 1)
    y = np.asarray(variance_ratio[:cap], dtype=float)

    # Locate the elbow of the convex, decreasing curve.
    locator = KneeLocator(x, y, curve="convex", direction="decreasing")
    knee = locator.knee

    # Fall back to the cap when no knee is detected.
    if knee is None:
        return cap

    # Clamp into [1, cap] and return as an int count.
    return int(min(max(1, int(knee)), cap))


__all__ = ["select_n_pcs"]
