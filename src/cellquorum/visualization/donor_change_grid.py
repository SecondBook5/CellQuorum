"""Per-donor change of scores between two conditions, with the donor-level evidence drawn in.

The usual picture of this is a heatmap of each donor's case-minus-control change with a mean
column, which shows the size of the change and nothing about whether the donors agree. Two marks
are added so the same grid answers that. A donor whose change runs against its row's mean is
outlined, so a row of nine solid cells reads as unanimous and a row with outlines does not. The
mean column carries the count of donors that move the way the mean does, out of the donors that
have both arms, and a star where that agreement survives correction across the rows shown (the
exact donor sign test, doubled because its direction is read off the same deltas, then
Benjamini-Hochberg).
The colour scale is symmetric about zero and chosen from the data, so a scale narrower than the
changes cannot saturate the cells and hide differences between donors.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.stats.paired_concordance import paired_value_concordance
from cellquorum.visualization.grids import value_grid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_STAR_BELOW = 0.05
_ROUND_NUMBERS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)


def _natural_key(label: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", label)]


def _round_up(x: float) -> float:
    """The smallest round number (1, 1.5, 2, 2.5, 3, 4, 5, 7.5 times a power of ten) >= ``x``."""
    if not np.isfinite(x) or x <= 0:
        return 0.1
    scale = 10.0 ** np.floor(np.log10(x))
    for m in _ROUND_NUMBERS:
        if m * scale >= x * (1 - 1e-9):
            return float(m * scale)
    return float(10.0 * scale)  # pragma: no cover - 10.0 covers every mantissa


def donor_change_grid(
    values: pd.DataFrame,
    donors: pd.Series,
    conditions: pd.Series,
    *,
    case: str,
    control: str,
    colorbar_label: str,
    notes: Sequence[str] = (),
    vmax: float | None = None,
    cell_in: float = 0.30,
) -> tuple[Figure, Axes]:
    """Draw each donor's case-minus-control change of every score, with donor agreement marked.

    Args:
        values: Samples (rows) by scores (columns); ``NaN`` means not measured.
        donors: Donor of each sample, indexed like ``values``.
        conditions: Condition of each sample, indexed like ``values``.
        case, control: The two condition levels; the change is case minus control.
        colorbar_label: What the colour means.
        notes: Footnotes written above the figure's own explanation of the marks.
        vmax: Half-range of the colour scale; by default the next round number above the
            largest absolute change shown.
        cell_in: Side of one cell, in inches.

    Returns:
        ``(figure, axes)`` as :func:`~cellquorum.visualization.grids.value_grid` does.
    """
    donor_of = donors.reindex(values.index).astype(str)
    arm_of = conditions.reindex(values.index).astype(str)
    order = sorted(donor_of.unique(), key=_natural_key)
    change = pd.DataFrame(np.nan, index=values.columns, columns=order, dtype=float)
    for donor in order:
        in_donor = donor_of == donor
        case_mean = values[in_donor & (arm_of == case)].mean(axis=0, skipna=True)
        control_mean = values[in_donor & (arm_of == control)].mean(axis=0, skipna=True)
        change[donor] = case_mean - control_mean
    table = paired_value_concordance(
        values, donor_of, arm_of, case=case, control=control
    ).set_index("item")
    table = table.reindex([str(c) for c in values.columns])

    grid = change.copy()
    grid.index = [str(i) for i in grid.index]
    grid["Mean"] = grid.mean(axis=1, skipna=True)
    direction = np.sign(grid["Mean"].to_numpy())
    against = pd.DataFrame(False, index=grid.index, columns=grid.columns)
    against[order] = np.sign(grid[order].to_numpy()) != direction[:, None]
    against &= grid.notna()
    label = pd.DataFrame("", index=grid.index, columns=grid.columns)
    for item, row in table.iterrows():
        text = f"{int(row.n_agree)}/{int(row.n_pairs)}"
        if (
            np.isfinite(row.sign_test_fdr_conservative)
            and row.sign_test_fdr_conservative < _STAR_BELOW
        ):
            text += "*"
        label.loc[item, "Mean"] = text

    limit = (
        float(vmax) if vmax is not None else _round_up(float(np.nanmax(np.abs(grid.to_numpy()))))
    )
    explain = (
        "Dashed outline: a donor whose change runs against the mean of its row. "
        "Mean column: donors changing the way the mean does, of the donors with both arms; "
        "* FDR < 0.05 on the exact donor sign test "
        "(doubled, Benjamini-Hochberg across the rows shown)."
    )
    return value_grid(
        grid,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        colorbar_label=colorbar_label,
        notes=[*notes, explain],
        row_label="",
        col_label="Donor (Mean = average over donors)",
        below=against,
        annotate=label,
        cell_in=cell_in,
    )


__all__ = ["donor_change_grid"]
