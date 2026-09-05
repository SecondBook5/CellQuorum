# Pipeline step (order=20): qc — post-run self-check that fails when QC contradicts itself.
"""Self-check: does the QC verdict agree with the evidence it claims to be based on?

Every defect found during the graded rollout was found by a human asking a question, not by a
test. Tests verify what the author intended; they cannot notice that the intention was wrong.
Four examples, all real, all shipped green:

**A calibrated probability was re-scaled.** The miQC posterior — median 0.035, MAD 0.015 — was
passed through a robust z, so a cell with a 10% chance of being compromised scored severity 0.59.
22,541 cells changed state. Every test passed, including thirty written for that area.

**A fallback null partitioned instead of nesting.** Cells that fell back to the library level
were compared only against each other, and cells fall back precisely when they are damaged. Real
damage detection fell from 100% to 10% while every rare-population test still passed.

**An audit flagged a doublet cluster as a lost population**, because it measured exclusion
without asking what caused it.

**An optional audit hung a 202,000-cell run** at 0% CPU for 27 minutes.

The common shape: the run produced a plausible number, and nothing compared that number against
an independent account of the same thing. That is what this module does. It is not more tests —
it runs inside the pipeline, on the actual output, and it **fails the stage** rather than logging.

## What a check must be

Each check compares the verdict against evidence derived a *different way*. A check that recomputes
the verdict and compares it with itself always passes and is worse than nothing, because it looks
like coverage. So: the posterior check compares the metabolic axis against the mixture's own
output; the nesting check compares group membership against level assignment; the coherence check
compares exclusion against transcriptional coherence.

``fail`` stops the run. ``warn`` reaches the report. Nothing is silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from cellquorum.stages.qc.lineage import UNASSIGNED

logger = logging.getLogger(__name__)

type Verdict = Literal["fail", "warn"]


@dataclass(frozen=True)
class Check:
    """One self-check outcome.

    Args:
        name: Stable identifier, used in provenance and in the failure message.
        passed: Whether the check held.
        detail: What was compared and what was found — written to be read by someone who did
            not write the check.
        verdict: ``fail`` stops the run; ``warn`` reaches the report.
    """

    name: str
    passed: bool
    detail: str
    verdict: Verdict = "fail"


@dataclass(frozen=True)
class SelfCheckReport:
    """The full self-check outcome for one QC run."""

    checks: list[Check] = field(default_factory=list)

    def failures(self) -> list[Check]:
        """Checks that failed with ``fail`` — these stop the run."""
        return [check for check in self.checks if not check.passed and check.verdict == "fail"]

    def warnings(self) -> list[str]:
        """Human-readable text for every check that did not hold."""
        return [
            f"QC self-check '{check.name}' did not hold: {check.detail}"
            for check in self.checks
            if not check.passed
        ]

    def summary(self) -> dict[str, bool]:
        """Per-check outcome, for provenance."""
        return {check.name: check.passed for check in self.checks}


def check_posterior_not_rescaled(
    metabolic_severity: pd.Series | None,
    mito_posterior: pd.Series | None,
    *,
    tolerance: float = 0.05,
) -> Check:
    """The metabolic axis must still *be* the mixture posterior, not a transform of it.

    The posterior is a calibrated probability. Any monotone rescaling preserves the ordering, so
    a rank-based check would pass; what breaks is the *scale*, and with it the meaning of every
    bar in the policy. So the comparison is on the values.

    This is the check that would have caught the 22,541-cell regression on the run that produced
    it, instead of three sessions later.
    """
    name = "posterior_not_rescaled"
    if metabolic_severity is None or mito_posterior is None:
        return Check(name, True, "no mixture posterior on this run; nothing to compare.", "warn")

    aligned = mito_posterior.reindex(metabolic_severity.index)
    both = pd.concat([metabolic_severity, aligned], axis=1).dropna()
    if both.empty:
        return Check(name, True, "no overlapping cells between axis and posterior.", "warn")

    largest = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    if largest <= tolerance:
        return Check(name, True, f"axis matches the posterior (max deviation {largest:.4f}).")
    return Check(
        name,
        False,
        f"the metabolic axis deviates from the mixture posterior by up to {largest:.3f} "
        f"(tolerance {tolerance}). The axis is a transform of the posterior rather than the "
        f"posterior itself, so every severity bar means something other than what it says. "
        f"Calibrate the mixture per lineage instead of rescaling its output.",
    )


def check_fallback_nulls_are_nested(
    level: pd.Series | None,
    keys: dict[str, pd.Series] | None,
) -> Check:
    """A coarser reference class must be *wider* than the cells that fell back to it.

    If a level's group contains only the cells assigned to that level, the fallback partitioned
    rather than nested — and since cells fall back precisely when their own group is unusable,
    which selects for damaged barcodes, their null then gets estimated from damage.

    Checked by comparing group membership against level assignment, which is independent of the
    severity those nulls produced.
    """
    name = "fallback_nulls_are_nested"
    if level is None or not keys:
        return Check(name, True, "no lineage-conditional grouping on this run.", "warn")

    levels = [value for value in keys if bool((level == value).any())]
    if len(levels) <= 1:
        return Check(name, True, "one level in use; nesting is not applicable.")

    finest = levels[0]
    for value in levels[1:]:
        assigned = (level == value).to_numpy()
        if not assigned.any():
            continue
        key = keys[value]
        # Every cell assigned to this level must sit in a group that also contains cells
        # assigned to a finer level; otherwise the group is exactly the fallback set.
        members = key.groupby(key).transform("size").to_numpy()
        own = pd.Series(assigned, index=key.index).groupby(key).transform("sum").to_numpy()
        partitioned = members[assigned] <= own[assigned]
        if partitioned.all():
            return Check(
                name,
                False,
                f"every cell that fell back to '{value}' sits in a group containing only other "
                f"fallback cells, so that null was estimated from the fallback set rather than "
                f"from all cells at that level. Cells fall back when their own group is "
                f"unusable, which selects for damaged barcodes, so this inverts the result: "
                f"damage becomes the reference. (Finest level in use: '{finest}'.)",
            )
    return Check(name, True, f"each fallback level ({', '.join(levels[1:])}) nests correctly.")


def check_core_fraction_plausible(
    state: pd.Series,
    *,
    minimum_core: float = 0.50,
) -> Check:
    """Most barcodes above the floor should survive as core, or something is miscalibrated.

    A blunt check on purpose. Graded QC assigns permissions rather than deleting, so a cohort
    where most cells cannot fit anything has produced a manifold defined by a minority — which
    may be correct for a badly degraded experiment and is never correct silently.
    """
    name = "core_fraction_plausible"
    fraction = float((state.astype(str) == "core").mean())
    if fraction >= minimum_core:
        return Check(name, True, f"{fraction:.1%} of cells are core.")
    return Check(
        name,
        False,
        f"only {fraction:.1%} of cells are core (floor {minimum_core:.0%}). The biological "
        f"reference would be defined by a minority of the data. Check the severity bars against "
        f"the distribution figures before trusting anything downstream.",
        "warn",
    )


def check_no_coherent_population_removed(
    lineage_audit: pd.DataFrame | None,
    *,
    minimum_cells: int = 50,
    minimum_cohort_share: float = 0.02,
) -> Check:
    """No transcriptionally coherent group may be excluded wholesale for damage reasons.

    The rare-population failure, as a gate. Multiplet-driven exclusion is already factored out
    of ``vulnerable`` by the audit, so a flag here means damage severity removed a group that
    looks like a real population.
    """
    name = "no_coherent_population_removed"
    if lineage_audit is None or lineage_audit.empty or "vulnerable" not in lineage_audit.columns:
        return Check(name, True, "no lineage audit on this run.", "warn")

    # The `unassigned` bucket is not a lineage and cannot be a lost population: it holds exactly
    # the barcodes that failed the lineage gene floor, so they have no group to be coherent with
    # and the clustering never placed them. It is the one label where "most of this group was
    # excluded" is the expected result rather than a warning — the floor is 50 genes, below which
    # a barcode carries too little to be anything, and the lowest-complexity real population
    # measured in this tissue sits at 744.
    #
    # Left in, it fires on any run with real debris and at any size, which is how a check that
    # gates the run becomes a check people switch off.
    flagged = lineage_audit[lineage_audit["vulnerable"]]
    flagged = flagged[flagged.index.astype(str) != UNASSIGNED]
    real = flagged
    if "n_cells" in flagged.columns:
        # Two floors, because two different things make a flag untrustworthy. A handful of
        # sub-floor barcodes is not a population — the floor already judged those. And a lineage
        # that is a negligible share of the cohort has a noisy per-lineage null, which is how a
        # two-library smoke subset flagged a lineage at 60% that sits at 28% on the full cohort.
        # A gate that stops a run on noise is a gate people disable.
        total = float(lineage_audit["n_cells"].sum()) or 1.0
        share = flagged["n_cells"] / total
        real = flagged[(flagged["n_cells"] >= minimum_cells) & (share >= minimum_cohort_share)]
    if real.empty:
        return Check(name, True, "no lineage is losing most of its cells to damage evidence.")
    worst = real.index.tolist()[:3]
    return Check(
        name,
        False,
        f"lineage(s) {worst} lose most of their non-multiplet cells to damage evidence. If any "
        f"of them is real biology this is the rare-population loss the lineage-conditional "
        f"design exists to prevent — check their markers before proceeding.",
    )


def check_masks_agree_with_state(
    state: pd.Series,
    fit_mask: pd.Series | None,
) -> Check:
    """The eligibility mask must match the verdict it was derived from.

    Cheap, and it catches the class of bug where a mask is written from a stale verdict — which
    is how a careful decision ends up controlling nothing.
    """
    name = "masks_agree_with_state"
    if fit_mask is None:
        return Check(name, True, "no eligibility masks on this run.", "warn")

    core = (state.astype(str) == "core").reindex(fit_mask.index)
    permitted = fit_mask.astype(bool)
    # Core cells may lose FIT to a multiplet call, but a non-core cell may never hold it.
    violations = int((permitted & ~core.fillna(False)).sum())
    if violations == 0:
        return Check(name, True, "every cell permitted to fit is core.")
    return Check(
        name,
        False,
        f"{violations:,} cells are permitted to fit while not being core. Non-core cells would "
        f"shape the biological reference, which the eligibility model exists to prevent.",
    )


def run_self_check(
    state: pd.Series,
    *,
    metabolic_severity: pd.Series | None = None,
    mito_posterior: pd.Series | None = None,
    null_level: pd.Series | None = None,
    null_keys: dict[str, pd.Series] | None = None,
    lineage_audit: pd.DataFrame | None = None,
    fit_mask: pd.Series | None = None,
    minimum_core: float = 0.50,
) -> SelfCheckReport:
    """Run every self-check that this run's inputs support.

    Args:
        state: Per-cell graded state.
        metabolic_severity: The metabolic family severity, for the posterior check.
        mito_posterior: The mixture's own posterior, compared against the axis.
        null_level: Per-cell null hierarchy level.
        null_keys: Group key per level, for the nesting check.
        lineage_audit: Per-lineage audit table.
        fit_mask: The manifold FIT mask.
        minimum_core: Core fraction below which the run is questioned.

    Returns:
        The report. Checks whose inputs are absent pass with a ``warn`` saying so, because
        "could not check" and "checked and fine" must not look identical.
    """
    report = SelfCheckReport(
        checks=[
            check_posterior_not_rescaled(metabolic_severity, mito_posterior),
            check_fallback_nulls_are_nested(null_level, null_keys),
            check_no_coherent_population_removed(lineage_audit),
            check_masks_agree_with_state(state, fit_mask),
            check_core_fraction_plausible(state, minimum_core=minimum_core),
        ]
    )
    for check in report.checks:
        if check.passed:
            logger.debug("QC self-check %s: ok — %s", check.name, check.detail)
        else:
            logger.warning("QC self-check %s FAILED: %s", check.name, check.detail)
    return report


__all__ = [
    "Check",
    "SelfCheckReport",
    "check_core_fraction_plausible",
    "check_fallback_nulls_are_nested",
    "check_masks_agree_with_state",
    "check_no_coherent_population_removed",
    "check_posterior_not_rescaled",
    "run_self_check",
]
