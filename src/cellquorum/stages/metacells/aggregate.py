"""Pure aggregation math over a MetaCell2 cell-to-metacell assignment table.

No AnnData, no subprocess -- every function here takes and returns plain
pandas objects so it is directly unit-testable against synthetic data. The
in-env `run_metacell2.py` script produces the assignment table this reads;
these functions compute the QC-scatter statistics (purity, dominant type,
median size, outlier share) that MetaCell2 itself does not hand back
pre-computed.

"Outliers" is a literal sentinel value in the `metacell_id` column (see the
spec's §4.3), not a real metacell -- every function here treats it as data,
excluding it from per-metacell statistics but counting it explicitly in
`outlier_summary`.
"""

from __future__ import annotations

import pandas as pd

_OUTLIER_LABEL = "Outliers"


def dominant_group(
    assignments: pd.DataFrame,
    *,
    metacell_col: str = "metacell_id",
    group_col: str = "group",
) -> pd.Series:
    """Return each metacell's most common `group_col` value.

    Args:
        assignments: One row per cell, with a metacell assignment and a group label.
        metacell_col: Column naming each cell's metacell (or "Outliers").
        group_col: Column naming each cell's group/cell-type label.

    Returns:
        Series indexed by metacell_id (excluding "Outliers"), majority group value.
    """
    real = assignments[assignments[metacell_col] != _OUTLIER_LABEL]
    return real.groupby(metacell_col)[group_col].agg(lambda s: s.value_counts().idxmax())


def purity_fraction(
    assignments: pd.DataFrame,
    *,
    metacell_col: str = "metacell_id",
    group_col: str = "group",
) -> pd.Series:
    """Return each metacell's fraction of cells belonging to its dominant group.

    Args:
        assignments: One row per cell, with a metacell assignment and a group label.
        metacell_col: Column naming each cell's metacell (or "Outliers").
        group_col: Column naming each cell's group/cell-type label.

    Returns:
        Series indexed by metacell_id (excluding "Outliers"), purity in [0, 1].
    """
    real = assignments[assignments[metacell_col] != _OUTLIER_LABEL]

    def _purity(s: pd.Series) -> float:
        counts = s.value_counts()
        return float(counts.iloc[0] / counts.sum())

    return real.groupby(metacell_col)[group_col].agg(_purity)


def median_size_per_metacell(
    assignments: pd.DataFrame,
    *,
    metacell_col: str = "metacell_id",
    size_col: str = "n_umis",
) -> pd.Series:
    """Return each metacell's median per-cell size (UMIs).

    Args:
        assignments: One row per cell, with a metacell assignment and a size column.
        metacell_col: Column naming each cell's metacell (or "Outliers").
        size_col: Column holding each cell's UMI count (or another size metric).

    Returns:
        Series indexed by metacell_id (excluding "Outliers"), median size.
    """
    real = assignments[assignments[metacell_col] != _OUTLIER_LABEL]
    return real.groupby(metacell_col)[size_col].median()


def outlier_summary(
    assignments: pd.DataFrame,
    *,
    metacell_col: str = "metacell_id",
) -> dict[str, int | float]:
    """Summarize the "Outliers" bucket against the real metacells.

    Args:
        assignments: One row per cell, with a metacell assignment.
        metacell_col: Column naming each cell's metacell (or "Outliers").

    Returns:
        Dict with n_outlier_cells, pct_outlier (0-100), and n_metacells (the
        count of distinct real metacell ids, excluding "Outliers").
    """
    is_outlier = assignments[metacell_col] == _OUTLIER_LABEL
    n_outlier = int(is_outlier.sum())
    n_total = int(len(assignments))
    pct_outlier = (n_outlier / n_total * 100) if n_total else 0.0
    n_metacells = int(assignments.loc[~is_outlier, metacell_col].nunique())
    return {
        "n_outlier_cells": n_outlier,
        "pct_outlier": pct_outlier,
        "n_metacells": n_metacells,
    }


__all__ = [
    "dominant_group",
    "median_size_per_metacell",
    "outlier_summary",
    "purity_fraction",
]
