"""Pure pandas composition tables: no AnnData, no plotting, no study biology.

Every downstream consumer of "what fraction of this cohort is each cell type" wants
the same two shapes: an overall table (one row per cell type) and a grouped table
(one row per group — condition, sample, donor, whatever the caller is comparing —
one column per cell type). Kept separate from the stage so they are trivially
unit-testable and reusable outside a pipeline run (e.g. from a notebook).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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


def grouped_composition(
    cell_type: pd.Series,
    group: pd.Series,
    *,
    group_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Per-group cell-type composition, in percent, rows summing to 100.

    Args:
        cell_type: Per-cell cell-type labels.
        group: Per-cell group labels (condition, sample, donor, ...), same length.
        group_order: Row order. Groups it names come first, in that order (a name with no
            cells is left out, never invented); the rest follow in sorted order. Without it
            the rows are alphabetical, which puts "Lymphedema" before "Normal".

    Returns:
        DataFrame indexed by group, one column per cell type (ordered by descending
        overall abundance), values in percent. A cell type absent from a group reads
        as 0.0, never a dropped column or a NaN.
    """

    cell_type = cell_type.astype(str)
    group = group.astype(str)
    order = list(overall_composition(cell_type)["cell_type"])
    table = pd.crosstab(group, cell_type, normalize="index") * 100.0
    table = table.reindex(columns=order, fill_value=0.0)
    if group_order is not None:
        named = [g for g in group_order if g in table.index]
        table = table.loc[[*named, *[g for g in table.index if g not in named]]]
    return table


def donor_paired_tests(
    by_patient: Mapping[str, pd.DataFrame],
    *,
    control: str,
    case: str,
    min_pairs: int = 4,
    fdr_method: str = "fdr_bh",
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Is each cell type's share different between conditions, with the donor as the replicate?

    Cells are not independent replicates, so a test over cells (or a chi-square on pooled
    counts) would call almost any shift significant. Here each cell type's share is compared
    within donors: the case minus control difference per donor present in both conditions,
    tested by a two-sided Wilcoxon signed-rank test, BH-adjusted across cell types.

    Args:
        by_patient: Condition -> table (donors as rows, cell types as columns, in percent).
        control: The control condition's key in ``by_patient``.
        case: The case condition's key.
        min_pairs: Fewer paired donors than this gives no p-value (NaN), because a signed-rank
            test on three pairs cannot reach significance and would only look like a result.

    Returns:
        One row per cell type: ``n_pairs``, ``mean_diff_pct`` (case minus control, percentage
        points), ``p``, ``padj`` and ``significant``.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    control_table, case_table = by_patient[control], by_patient[case]
    donors = [d for d in control_table.index if d in set(case_table.index)]
    rows = []
    for cell_type in control_table.columns:
        diff = (case_table.loc[donors, cell_type] - control_table.loc[donors, cell_type]).to_numpy(
            dtype=float
        )
        p = float("nan")
        if len(donors) >= min_pairs and np.any(diff != 0):
            p = float(wilcoxon(diff, zero_method="wilcox").pvalue)
        rows.append(
            {
                "cell_type": cell_type,
                "n_pairs": len(donors),
                "mean_diff_pct": float(diff.mean()) if len(donors) else float("nan"),
                "p": p,
            }
        )
    table = pd.DataFrame(rows)
    tested = table["p"].notna()
    table["padj"] = float("nan")
    if tested.any():
        table.loc[tested, "padj"] = multipletests(table.loc[tested, "p"], method=fdr_method)[1]
    table["significant"] = table["padj"].lt(fdr).fillna(False)
    return table


__all__ = ["overall_composition", "grouped_composition", "donor_paired_tests"]
