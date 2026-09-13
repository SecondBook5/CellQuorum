"""Configuration for the differential-state stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DifferentialStateConfig(StrictBaseModel):
    """
    Donor-paired testing of per-cell state/program scores between conditions.

    ``state_scoring`` computes per-cell program/state scores but never tests whether
    they differ between conditions; this stage is that missing test, reusing the
    engine's own validated donor-median comparison
    (``cellquorum.stats.donor_comparison.two_group_test_on_donor_medians``) instead
    of ad hoc per-analysis statistics.

    Attributes:
        obs_columns: Explicit obs column names to test. None (the default)
            auto-detects every numeric column starting with ``score_prefix``.
        score_prefix: Prefix used for auto-detection when ``obs_columns`` is None.
        condition_key: obs column holding the condition/treatment label. None
            resolves from ``cohort.condition_key``, then ``design.condition_col``.
        donor_key: obs column identifying a donor. Same resolution as
            ``condition_key``, falling back to ``design.donor_col``.
        case: The condition level treated as the case arm. None resolves from
            ``design.case``.
        control: The condition level treated as the control arm. None resolves
            from ``design.control``.
        paired: Whether donors are matched across both conditions (donor-level
            Wilcoxon signed-rank) versus independent (Mann-Whitney U).
        min_cells_per_donor: Minimum cells a (donor, condition) group must have
            before its per-cell scores are trusted as a per-donor summary.
        min_donors: Minimum usable donors required to run a test for one program.
        fdr: FDR threshold recorded and used for the ``significant`` column.
        fdr_method: statsmodels ``multipletests`` correction method.
        output_prefix: Prefix used for output artifact filenames.
    """

    obs_columns: list[str] | None = None
    score_prefix: str = "state_"

    condition_key: str | None = None
    donor_key: str | None = None
    case: str | None = None
    control: str | None = None
    paired: bool = True

    min_cells_per_donor: int = 20
    min_donors: int = 4

    fdr: float = 0.05
    fdr_method: str = "fdr_bh"

    output_prefix: str = "differential_state"


__all__ = ["DifferentialStateConfig"]
