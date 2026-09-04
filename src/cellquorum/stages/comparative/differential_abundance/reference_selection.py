"""Reference cell-type selection for compositional differential abundance.

Compositional models cannot measure all cell types at once: proportions sum to
one, so "everything went up" is not expressible. scCODA resolves this by holding
one cell type fixed and expressing every other effect relative to it. That choice
is not cosmetic -- it is the denominator of every reported effect, so a noisy
reference adds its own variance to all of them.

Why the engine chooses the reference instead of delegating to scCODA
-------------------------------------------------------------------
scCODA's ``reference_cell_type="automatic"`` documents itself as picking "the
cell type with the lowest dispersion in relative abundance", but the quantity it
minimises is ``var(p) / mean(p)``. For a proportion with mean ``p`` and
coefficient of variation ``c``, that equals ``c**2 * p`` -- it scales with the
mean, so it is not a measure of stability at all. It ranks cell types
substantially by how RARE they are, and the rarest cell type usually wins however
noisy it is.

Measured on a 9-donor, 13-lineage skin cohort, that criterion ranked cell types
at rank-correlation 0.77 with mean abundance and selected a population making up
0.28% of cells with a coefficient of variation of 0.90 -- the second least stable
of the eligible types. The genuinely stable lineages, at CV 0.31 and 0.39, ranked
near-last. Its only guard is a presence filter, which a rare-but-never-absent
population passes trivially.

So the engine picks the reference itself, on a scale-free criterion, and records
what it picked and why. Two rules, both of which the scCODA default is missing:

* Judge stability by the variance of the centred log-ratio, not by var/mean. The
  CLR is the standard transform for compositional data and is invariant to the
  overall scale of a cell type, so a rare population gets no automatic advantage.
* Require a minimum mean abundance, not merely presence. Ratios against a
  population of a few dozen cells are dominated by counting noise no matter how
  stable that population looks.

Why not simply pick the cell type that changes least between conditions
-----------------------------------------------------------------------
Because that is selecting on the outcome. The reference's third requirement, after
being abundant and ubiquitous, is genuinely the important one: a reference that
itself responds to the condition contaminates every other effect with its own
change, carried in with the opposite sign. It is tempting to measure that directly
and take the cell type with the smallest condition effect.

That would bias the whole table. Choosing the minimum of a set of estimated
condition effects and then reporting all the others relative to it inflates the
remaining effects by construction, and the inflation is largest exactly when the
cohort is small and the estimates noisy.

The centred-log-ratio variance avoids this because it never looks at the condition
labels. It is the total spread of a cell type's share across all samples, which a
strong condition effect does inflate -- so a responding cell type is penalised --
but it is not a function of the labels and cannot be gamed by them. It also
penalises donor-to-donor noise, which for a denominator is a feature. The cost is
that it cannot distinguish a cell type that is stable because it does not respond
from one that is stable because the cohort is homogeneous, which is why the
criterion table is emitted rather than just the winner.

This does not always change the answer -- on the cohort above the fit was a
genuine null under either reference -- but on a cohort with real compositional
structure the reference is the denominator of every effect size in the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Require a cell type to be observed in essentially every sample to be eligible.
#
# A reference that is absent from a sample makes that sample's ratios undefined,
# which the pseudocount then papers over. One tolerated absence in twenty is
# enough slack for a sampling accident without admitting a population that is
# genuinely missing from part of the cohort.
DEFAULT_MIN_PRESENCE = 0.95

# Require a cell type to hold at least this share of cells, on average.
#
# Five percent. The reference is the denominator of every reported effect, so its
# own counting noise propagates into all of them; below a few percent that noise
# dominates. This is the guard scCODA's automatic selection lacks entirely, and it
# is what excludes the 0.28% population that criterion preferred.
DEFAULT_MIN_MEAN_ABUNDANCE = 0.05

# Pseudocount added before the log-ratio, matching what scCODA does internally.
#
# The CLR is undefined at zero. scCODA adds 0.5 to the whole matrix when it sees
# any zero, so using the same value keeps the selection criterion consistent with
# the model that consumes the choice.
LOG_RATIO_PSEUDOCOUNT = 0.5

# Name the columns of the criterion table, so an empty selection still has a schema.
REFERENCE_CRITERION_COLUMNS: tuple[str, ...] = (
    "cell_type",
    "presence",
    "mean_abundance",
    "abundance_cv",
    "clr_variance",
    "sccoda_dispersion",
    "eligible",
    "selected",
)


@dataclass(frozen=True)
class ReferenceChoice:
    """Record which cell type was chosen as the compositional reference, and why.

    Attributes:
        cell_type: Chosen reference, or None when no cell type qualified and the
            decision is being handed back to the fitting backend.
        reason: Plain-language account of how the choice was reached, including
            any relaxation that was applied. Always populated.
        relaxed: True when no cell type met the abundance floor and the floor had
            to be dropped to reach a choice, which is worth reporting alongside
            the result rather than silently absorbing.
        criterion: Per-cell-type table behind the decision, ordered by
            ``clr_variance``. Carries both the engine's criterion and scCODA's,
            so the two can be compared on any dataset.
    """

    cell_type: str | None
    reason: str
    relaxed: bool
    criterion: pd.DataFrame


def select_compositional_reference(
    counts: pd.DataFrame,
    *,
    min_presence: float = DEFAULT_MIN_PRESENCE,
    min_mean_abundance: float = DEFAULT_MIN_MEAN_ABUNDANCE,
) -> ReferenceChoice:
    """
    Choose the most stable abundant cell type to use as the compositional reference.

    Ranks cell types by the variance of their centred log-ratio across samples --
    a scale-free measure of stability -- among those present in nearly every
    sample and holding at least ``min_mean_abundance`` of cells on average.

    Args:
        counts: Samples (rows) × cell types (columns) integer count matrix.
        min_presence: Fraction of samples in which a cell type must be observed.
        min_mean_abundance: Minimum mean relative abundance for eligibility.

    Returns:
        ReferenceChoice carrying the selected cell type, the reasoning, and the
        full criterion table. ``cell_type`` is None only when the matrix has no
        usable cell types at all.

    Notes:
        Both ``clr_variance`` and scCODA's own ``sccoda_dispersion`` are reported
        so the divergence between the two is visible per dataset rather than
        taken on trust. The abundance floor is relaxed rather than failing, since
        a cohort of rare populations still needs some reference; the relaxation
        is flagged instead of hidden.
    """

    empty = pd.DataFrame(columns=list(REFERENCE_CRITERION_COLUMNS))

    if counts is None or counts.empty or counts.shape[1] == 0:
        return ReferenceChoice(None, "no cell types in the count matrix", False, empty)

    matrix = counts.to_numpy(dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)

    # A sample with no cells carries no compositional information and would make
    # every proportion undefined, so it is dropped before anything is measured.
    usable = (totals[:, 0] > 0) & np.isfinite(totals[:, 0])
    if not usable.any():
        return ReferenceChoice(None, "every sample has a zero cell total", False, empty)
    matrix = matrix[usable]
    totals = totals[usable]

    relative = matrix / totals

    presence = (matrix > 0).mean(axis=0)
    mean_abundance = relative.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        abundance_cv = np.where(mean_abundance > 0, relative.std(axis=0) / mean_abundance, np.inf)
        # scCODA's own criterion, reported for comparison rather than used.
        sccoda_dispersion = np.where(
            mean_abundance > 0, relative.var(axis=0) / mean_abundance, np.inf
        )

    # The centred log-ratio: log abundance minus the per-sample mean log abundance.
    # Subtracting that mean is what makes the result independent of the arbitrary
    # total, and taking the variance across samples is then a scale-free measure
    # of how steady a cell type's share is.
    padded = matrix + LOG_RATIO_PSEUDOCOUNT
    log_share = np.log(padded / padded.sum(axis=1, keepdims=True))
    clr = log_share - log_share.mean(axis=1, keepdims=True)
    clr_variance = clr.var(axis=0)

    table = pd.DataFrame(
        {
            "cell_type": list(counts.columns),
            "presence": presence,
            "mean_abundance": mean_abundance,
            "abundance_cv": abundance_cv,
            "clr_variance": clr_variance,
            "sccoda_dispersion": sccoda_dispersion,
        }
    )

    present_enough = table["presence"] >= min_presence
    abundant_enough = table["mean_abundance"] >= min_mean_abundance

    relaxed = False
    eligible = present_enough & abundant_enough
    if not eligible.any():
        # No population is both ubiquitous and abundant. Refusing to choose would
        # leave the fit with scCODA's rarity-seeking default, which is worse than
        # a knowingly-relaxed choice, so drop the abundance floor and say so.
        relaxed = True
        eligible = present_enough

    if not eligible.any():
        table["eligible"] = False
        table["selected"] = False
        return ReferenceChoice(
            None,
            (
                f"no cell type is present in at least {min_presence:.0%} of samples, so the "
                f"reference choice is left to the fitting backend"
            ),
            relaxed,
            table.sort_values("clr_variance").reset_index(drop=True),
        )

    table["eligible"] = eligible
    candidates = table[eligible]
    chosen = candidates.loc[candidates["clr_variance"].idxmin(), "cell_type"]
    table["selected"] = table["cell_type"] == chosen

    row = table[table["cell_type"] == chosen].iloc[0]
    reason = (
        f"{chosen} has the steadiest share of the {int(eligible.sum())} eligible cell "
        f"types (centred-log-ratio variance {row['clr_variance']:.4g}, "
        f"{row['mean_abundance']:.1%} of cells on average, present in "
        f"{row['presence']:.0%} of samples)"
    )
    if relaxed:
        reason += (
            f"; no cell type reached the {min_mean_abundance:.0%} abundance floor, so the "
            f"floor was dropped and effects measured against this reference carry its "
            f"counting noise"
        )

    # Compare against what scCODA would ACTUALLY have chosen, which means applying
    # its own guard -- a presence filter and nothing else -- rather than restricting
    # it to the abundance-eligible set. Comparing within the eligible set would hide
    # precisely the failure this module exists to prevent, since the rare population
    # scCODA prefers is the one the abundance floor already removed.
    contenders = table[present_enough]
    sccoda_pick = contenders.loc[contenders["sccoda_dispersion"].idxmin(), "cell_type"]
    if sccoda_pick != chosen:
        # Worth stating explicitly: this is the case the module exists to prevent,
        # and seeing it named per-dataset is how a reader knows it mattered here.
        other = table[table["cell_type"] == sccoda_pick].iloc[0]
        reason += (
            f"; scCODA's own criterion would have picked {sccoda_pick} instead "
            f"({other['mean_abundance']:.1%} of cells, abundance CV {other['abundance_cv']:.2f} "
            f"against {row['abundance_cv']:.2f})"
        )

    return ReferenceChoice(
        str(chosen), reason, relaxed, table.sort_values("clr_variance").reset_index(drop=True)
    )


def split_reference_fits(
    da: pd.DataFrame, reference: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate the reported scCODA fit from the reference-sensitivity fit.

    The scCODA helper returns two fits stacked in one table whenever a reference was
    resolved -- one at that reference, one at scCODA's own automatic pick -- told
    apart only by the ``reference`` column. Both are wanted: the second is a free
    sensitivity check on the choice of denominator. But the table then holds every
    cell type twice, and a reader (or a metric) that takes it whole counts each cell
    type twice. Splitting is therefore not a plotting convenience; it is what makes
    the table mean one thing.

    Args:
        da: scCODA result table, with or without a ``reference`` column.
        reference: The reference the engine reported on. ``None`` or a label absent
            from the table falls back to the first block present, so a caller always
            gets a usable primary frame.

    Returns:
        ``(primary, sensitivity)``, each with a fresh integer index and the columns
        the input had. ``sensitivity`` is empty when only one fit ran.
    """

    if da is None or da.empty:
        empty = pd.DataFrame(columns=list(da.columns) if da is not None else [])
        return empty, empty

    # No reference column means one fit, which is the reported one.
    if "reference" not in da.columns:
        return da.reset_index(drop=True), da.iloc[0:0]

    labels = da["reference"].astype(str)
    available = list(dict.fromkeys(labels))
    wanted = str(reference) if reference is not None else None
    chosen = wanted if wanted in available else available[0]

    is_primary = labels == chosen
    return (
        da[is_primary].reset_index(drop=True),
        da[~is_primary].reset_index(drop=True),
    )


__all__ = [
    "DEFAULT_MIN_MEAN_ABUNDANCE",
    "DEFAULT_MIN_PRESENCE",
    "REFERENCE_CRITERION_COLUMNS",
    "ReferenceChoice",
    "select_compositional_reference",
    "split_reference_fits",
]
