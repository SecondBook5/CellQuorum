"""Configuration for the QC finalization stage (order 135)."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class QCFinalizationConfig(StrictBaseModel):
    """Thresholds for per-cell rescue into qc_state_final.

    No ``enabled`` field, deliberately: whether the stage runs is declared once, in
    ``stages.qc_finalization``.

    Every threshold is stated rather than inherited so a methods section can cite it and a
    library upgrade cannot silently re-adjudicate a published cohort. They are
    intentionally uncalibrated until fitted against the lymphedema QC figures.
    """

    # A borderline cell needs at least this fraction of its core neighbours sharing one
    # label to count as having biological support.
    min_neighborhood_support: float = 0.5

    # And it must not be more out-of-distribution than this quantile of the reference's
    # own neighbour-distance distribution (1.0 = fits worse than every reference cell).
    max_ood_score: float = 0.95

    # A per-family QC severity at or above this level is a severe technical contradiction
    # that vetoes rescue regardless of how well the cell maps.
    severe_severity: float = 0.9


__all__ = ["QCFinalizationConfig"]
