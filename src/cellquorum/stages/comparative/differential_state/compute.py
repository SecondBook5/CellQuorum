"""Pure pandas helpers around donor-paired state/program score testing.

No AnnData, no statistics — the actual test is
``cellquorum.stats.donor_comparison.two_group_test_on_donor_medians``, already
validated and reused as-is. This module only prepares its input: deciding which
obs columns to test, and dropping a (donor, condition) group too thin to trust as
a per-donor summary before that group's cells ever reach the median.
"""

from __future__ import annotations

import pandas as pd


def resolve_score_columns(
    obs: pd.DataFrame,
    *,
    explicit: list[str] | None,
    prefix: str,
) -> list[str]:
    """
    Return the obs columns to test: an explicit list, or numeric columns by prefix.

    Args:
        obs: The AnnData's obs frame (or any frame with the same columns).
        explicit: Column names to use verbatim, bypassing auto-detection. None
            triggers auto-detection.
        prefix: When auto-detecting, only numeric columns whose name starts with
            this prefix are returned (score_genes_method's own "state_" convention
            by default) — a non-numeric column of the same prefix (e.g. a
            categorical label) is excluded, not coerced.

    Returns:
        Column names in ``obs``'s own order.
    """

    if explicit is not None:
        return list(explicit)
    return [
        col
        for col in obs.columns
        if str(col).startswith(prefix) and pd.api.types.is_numeric_dtype(obs[col])
    ]


def drop_underpowered_donor_groups(
    frame: pd.DataFrame,
    *,
    donor_col: str,
    condition_col: str,
    min_cells: int,
) -> pd.DataFrame:
    """
    Drop rows from a (donor, condition) group with fewer than ``min_cells`` cells.

    A donor can pass in one condition and fail in the other — each (donor,
    condition) pair is evaluated independently, not the donor as a whole.

    Args:
        frame: Per-cell rows with at least ``donor_col`` and ``condition_col``.
        donor_col: Column identifying the donor.
        condition_col: Column identifying the condition.
        min_cells: Minimum cells a (donor, condition) group must have to be kept.

    Returns:
        The subset of ``frame`` belonging to sufficiently large groups.
    """

    counts = frame.groupby([donor_col, condition_col], observed=True)[donor_col].transform("size")
    return frame.loc[counts.to_numpy() >= min_cells].copy()


__all__ = ["resolve_score_columns", "drop_underpowered_donor_groups"]
