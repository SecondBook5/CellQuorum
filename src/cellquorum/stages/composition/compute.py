"""Pure pandas composition tables: no AnnData, no plotting, no study biology.

Every downstream consumer of "what fraction of this cohort is each cell type" wants
the same two shapes: an overall table (one row per cell type) and a grouped table
(one row per group — condition, sample, donor, whatever the caller is comparing —
one column per cell type). Kept separate from the stage so they are trivially
unit-testable and reusable outside a pipeline run (e.g. from a notebook).
"""

from __future__ import annotations

import pandas as pd


def overall_composition(cell_type: pd.Series) -> pd.DataFrame:
    """
    Cell-type counts and fractions across the whole input, most abundant first.

    Args:
        cell_type: Per-cell cell-type labels.

    Returns:
        DataFrame with columns ``cell_type``, ``n_cells``, ``fraction`` (sums to 1.0),
        ordered by descending abundance.
    """

    counts = cell_type.astype(str).value_counts()
    return pd.DataFrame(
        {
            "cell_type": counts.index,
            "n_cells": counts.to_numpy(),
            "fraction": (counts / counts.sum()).to_numpy(),
        }
    )


def grouped_composition(cell_type: pd.Series, group: pd.Series) -> pd.DataFrame:
    """
    Per-group cell-type composition, in percent, rows summing to 100.

    Args:
        cell_type: Per-cell cell-type labels.
        group: Per-cell group labels (condition, sample, donor, ...), same length.

    Returns:
        DataFrame indexed by group, one column per cell type (ordered by descending
        overall abundance), values in percent. A cell type absent from a group reads
        as 0.0, never a dropped column or a NaN.
    """

    cell_type = cell_type.astype(str)
    group = group.astype(str)
    order = list(overall_composition(cell_type)["cell_type"])
    table = pd.crosstab(group, cell_type, normalize="index") * 100.0
    return table.reindex(columns=order, fill_value=0.0)


__all__ = ["overall_composition", "grouped_composition"]
