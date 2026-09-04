"""Whether a row of a results table could have carried a claim at all.

Two questions that every group-level results table invites and almost none answer,
both of them properties of the *design* rather than of the data:

**Could the test have cleared its own threshold?** A table of non-significant FDRs
reads as evidence of no effect, and often is not. An assumption-free paired test on
nine donors cannot return a two-sided p below ``2/2**9 = 0.0039``, so BH over
thirteen cell types cannot return an FDR below ``0.051`` — the test is arithmetically
incapable of clearing 0.05 at any effect size whatsoever. This is not hypothetical:
this project read "nothing clears FDR" off exactly that arrangement and wrote
"whole-atlas composition is a real null" into a manuscript note. Re-run with a
parametric test on the same cells, two cell types cleared at FDR 0.030. The floor
described the test, not the tissue.

So :func:`annotate_fdr_reachability` puts the design's floor and the family's
reachability *next to* every FDR, in the columns
:data:`FDR_REACHABILITY_COLUMNS`. It rewrites no p-value and vetoes nothing — a
parametric test is not bounded by the floor, which is precisely what the
distributional assumption buys, and gating on ``p_below_design_floor`` alone is the
second error committed while fixing the first (see
:func:`~cellquorum.stats.module_remodeling.randomization_floor`). The floor is a
*scale*: it says how much of a small p-value came from the cohort and how much came
from the model, and it says when a null result is uninformative by construction.

**Is the row's ratio measured on enough cells to mean anything?** A fold-change is
scale-free, which is its virtue on a compositional axis and its trap on a rare
population: a 10.6x enrichment computed from a median of 1.5 cells per sample
outranks everything real in the table, and a share of exactly zero in one arm makes
the ratio infinite. :func:`group_resolution` reports the per-sample cell counts the
ratio rests on and flags the rows below a stated floor. It does **not** drop them —
hiding data to tidy a ranking is the worse failure — it names them so a figure can
mark them and rank them separately, and so a reader can see the count rather than
infer it.

Both functions are study-agnostic: they take counts and design labels, know nothing
about lineages or biology, and are importable from any hypothesis repo.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cellquorum.stats.module_remodeling import fdr_floor_reachability, randomization_floor

#: Columns :func:`annotate_fdr_reachability` adds. ``design_floor_p`` and
#: ``p_below_design_floor`` are per row; the three ``family_*`` columns are properties of
#: the BH family and so repeat down it — which is the point, since a reader looking at one
#: row must not have to reconstruct the family to interpret its FDR.
FDR_REACHABILITY_COLUMNS: tuple[str, ...] = (
    "design_floor_p",
    "p_below_design_floor",
    "family_size",
    "family_min_concordant",
    "family_floor_reachable",
)

#: Columns :func:`group_resolution` returns, after the group-label column.
GROUP_RESOLUTION_COLUMNS: tuple[str, ...] = (
    "total_cells",
    "n_samples",
    "median_cells_per_sample",
    "min_cells_per_sample",
    "n_samples_zero",
    "median_cells_case",
    "median_cells_control",
    "ratio_rankable",
    "resolution_note",
)

#: Default per-sample cell floor below which a group's ratio is not ranked. Ten is not a
#: statistical threshold and is not claimed to be one; it is the count at which a single
#: cell stops moving the group's share by more than 10%, and it wants restating per assay.
MIN_CELLS_PER_SAMPLE = 10


def annotate_fdr_reachability(
    table: pd.DataFrame,
    *,
    donors: Sequence,
    is_case: Sequence[bool],
    p_col: str = "pvalue",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add the design floor and the BH family's reachability beside an existing FDR.

    Computes the floor once from the design — condition is assigned to a *sample*, so
    the floor follows the donor/arm structure and not the cell count — and the family
    reachability from the floor together with the number of rows that were actually
    corrected. Rows whose p-value is not finite were held out of the correction, so
    they are not counted in the family either.

    Nothing is re-tested and no p-value or FDR is modified.

    Args:
        table: One row per tested group, already carrying ``p_col`` and its FDR.
        donors: Donor identifier per *sample* used by the test (not per cell). For a
            paired test each paired donor appears once per arm.
        is_case: Whether each of those samples is in the case arm.
        p_col: Name of the p-value column whose family is being described.
        alpha: FDR level the family is corrected at.

    Returns:
        A copy of ``table`` with :data:`FDR_REACHABILITY_COLUMNS` appended. Row order is
        preserved. An empty input is returned with the columns present but no rows, so a
        downstream reader never has to test for their existence.

    Notes:
        ``family_min_concordant`` is the smallest number of rows that must simultaneously
        reach the floor before BH can call *any* of them. When it exceeds the family size
        (``family_floor_reachable`` is False), a lone significant result was unavailable to
        this cohort at this alpha whatever the effect, and a table of null FDRs is
        uninformative rather than negative.
    """
    out = table.copy()
    if p_col not in out.columns:
        raise KeyError(f"annotate_fdr_reachability: no p-value column {p_col!r} in table")

    floor_p, _ = randomization_floor(donors, is_case)

    if out.empty:
        for column in FDR_REACHABILITY_COLUMNS:
            out[column] = pd.Series(dtype="float64" if "floor_p" in column else "object")
        return out

    # The family is what BH actually corrected over. A row whose test could not be fitted
    # was held out of the correction (see ``bh_fdr``), so counting it here would inflate
    # the family and overstate how many rows must move together.
    finite = np.isfinite(pd.to_numeric(out[p_col], errors="coerce").to_numpy(dtype=float))
    family_size = int(finite.sum())
    min_concordant, reachable = fdr_floor_reachability(floor_p, family_size, alpha=alpha)

    out["design_floor_p"] = floor_p
    out["p_below_design_floor"] = (
        pd.to_numeric(out[p_col], errors="coerce") < floor_p if np.isfinite(floor_p) else False
    )
    out["family_size"] = family_size
    out["family_min_concordant"] = min_concordant
    out["family_floor_reachable"] = bool(reachable)
    return out


def group_resolution(
    counts: pd.DataFrame,
    conditions: pd.Series | None = None,
    *,
    case: str | None = None,
    control: str | None = None,
    group_label: str = "cell_type",
    min_cells_per_sample: int = MIN_CELLS_PER_SAMPLE,
) -> pd.DataFrame:
    """Per-group per-sample cell counts, and whether the group's ratio is rankable.

    A results table gives every group one row of equal visual weight, and a fold-change
    axis then hands the top of the ranking to whichever group is rarest. This says how
    many cells each row's ratio was computed from, on the unit that matters — cells *per
    sample*, since that is what the per-sample proportion divides — rather than the pooled
    total, which is large enough to look reassuring for a group present in only half the
    samples.

    Args:
        counts: Samples (rows) x groups (columns) count matrix, as produced by
            ``aggregate_celltype_counts``.
        conditions: Per-sample condition labels indexed by ``counts.index``. Optional; the
            per-arm medians are NaN without it.
        case: Condition label for the case arm. Required for ``median_cells_case``.
        control: Condition label for the control arm.
        group_label: Name given to the group column in the output.
        min_cells_per_sample: Median per-sample count at or above which a group's ratio is
            marked rankable.

    Returns:
        One row per group with ``group_label`` followed by
        :data:`GROUP_RESOLUTION_COLUMNS`, ordered by descending
        ``median_cells_per_sample``. ``resolution_note`` is empty for a rankable group and
        otherwise states the measured reason in words, so a figure caption or a table can
        quote it without recomputing anything.

    Notes:
        ``n_samples_zero`` is reported separately from the minimum because they fail
        differently: a low minimum makes a ratio noisy, whereas a sample with none of a
        group makes that sample's proportion exactly zero, and a zero in the control arm
        makes the group's fold-change infinite rather than merely uncertain.
    """
    empty = pd.DataFrame(columns=[group_label, *GROUP_RESOLUTION_COLUMNS])
    if counts is None or counts.empty or counts.shape[1] == 0:
        return empty

    matrix = counts.to_numpy(dtype=float)
    n_samples = int(matrix.shape[0])

    arm_medians: dict[str, np.ndarray] = {}
    for name, label in (("median_cells_case", case), ("median_cells_control", control)):
        if conditions is None or label is None:
            arm_medians[name] = np.full(matrix.shape[1], np.nan)
            continue
        aligned = conditions.reindex(counts.index).astype(str)
        rows = np.flatnonzero((aligned == str(label)).to_numpy())
        arm_medians[name] = (
            np.median(matrix[rows], axis=0) if rows.size else np.full(matrix.shape[1], np.nan)
        )

    medians = np.median(matrix, axis=0)
    minima = matrix.min(axis=0)
    zeros = (matrix == 0).sum(axis=0)

    rows = []
    for column, group in enumerate(counts.columns):
        median = float(medians[column])
        rankable = median >= float(min_cells_per_sample)
        notes = []
        if not rankable:
            notes.append(
                f"median {median:.1f} cells/sample is below the {min_cells_per_sample}-cell "
                "floor, so the ratio is not ranked"
            )
        if zeros[column]:
            notes.append(f"{int(zeros[column])} of {n_samples} samples contain none of this group")
        rows.append(
            {
                group_label: str(group),
                "total_cells": int(matrix[:, column].sum()),
                "n_samples": n_samples,
                "median_cells_per_sample": median,
                "min_cells_per_sample": float(minima[column]),
                "n_samples_zero": int(zeros[column]),
                "median_cells_case": float(arm_medians["median_cells_case"][column]),
                "median_cells_control": float(arm_medians["median_cells_control"][column]),
                "ratio_rankable": bool(rankable),
                "resolution_note": "; ".join(notes),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        "median_cells_per_sample", ascending=False, kind="stable"
    )
    return table.reset_index(drop=True)[[group_label, *GROUP_RESOLUTION_COLUMNS]]


__all__ = [
    "FDR_REACHABILITY_COLUMNS",
    "GROUP_RESOLUTION_COLUMNS",
    "MIN_CELLS_PER_SAMPLE",
    "annotate_fdr_reachability",
    "group_resolution",
]
