"""Design-interaction checks for CellQuorum QC: differential attrition and leaks.

This module asks one question twice -- does QC interact with the study design? --
once before the fact from the configuration (``audit_qc_design_leaks``) and once
after the fact from the decisions (``audit_differential_attrition``). The pre-hoc
check catches the configurations that guarantee the problem; the post-hoc check
measures whatever happened regardless of how the rules were written.

Differential-attrition auditing.

Why this module exists: every QC rule in this pipeline is chosen to be defensible
on its own terms -- a mixture model rather than a guessed ceiling, a projection
that makes the rule monotone in the metric it names. None of that guarantees the
one property the downstream statistics actually depend on, which is that QC
removed cells at the SAME rate in every arm of the study.

When it does not, QC has stopped being a filter and become a covariate. A
differentially-filtered dataset carries a signal that is indistinguishable from
biology by any test run afterwards: the diseased arm looks different partly
because a different slice of it survived. Nothing about the individual rule being
principled prevents this. Adaptive thresholds MAKE it more likely, because they
are estimated from the data and the data differ between arms.

This is checkable, cheaply, on any dataset, from the decision table the QC stage
already produces -- so the engine checks it rather than leaving it to whoever
happens to look. Three tests, in increasing honesty about the unit of analysis:

* Unstratified, cells as the unit (Fisher exact, or chi-square above two levels).
  The most sensitive and the least trustworthy; reported for completeness.
* Stratified on a blocking factor, usually donor (Cochran-Mantel-Haenszel). Donor
  quality varies enormously and correlates with arm composition in most cohorts,
  so the pooled table can show a large difference that no donor exhibits.
* Paired on the blocking factor, donors as the unit (Wilcoxon signed-rank on the
  per-donor removal rates). Cells within a donor are not independent
  observations, so this is the number a reviewer will ask for.

Each of the three then runs a second time WITHIN each subset of the object,
normally each cell type, because a cohort-level rate is an average and the
downstream analyses are not. A whole-object removal rate that differs by half a
point between arms can be four points inside one lineage and zero everywhere
else, and the per-lineage contrast is the one that goes in the paper. Pooling
hides exactly the imbalance that matters, and the fix costs nothing: the same
tests, on the same decision table, restricted by one label column.

Subsets carry a multiplicity problem the two cohort rows do not. The cohort test
is pre-specified -- one factor, one question -- so its p-value stands as
computed. The subset pass asks the same question of every cell type at once, so
its p-values are Benjamini-Hochberg adjusted within each (factor, unit) family
and it is the ADJUSTED value that decides whether a warning fires. Reporting
thirteen raw p-values as if each were pre-specified would manufacture roughly one
alarming lineage per run out of nothing.

Warnings are gated on effect size as well as significance. With tens of
thousands of cells, a half-point difference in removal rate is significant at any
alpha and means nothing; an engine that warns about it teaches its users to
ignore the warning.
"""

from __future__ import annotations

# Import Mapping for the factor collection.
from collections.abc import Mapping

# Import dataclass helpers for structured audit records. ``replace`` is what lets
# the multiplicity correction stamp an adjusted p-value onto a frozen record.
from dataclasses import dataclass, field, replace

# Import NumPy for the contingency arithmetic.
import numpy as np

# Import pandas for label handling and grouping.
import pandas as pd

# Set the significance level at which a difference in attrition is called real.
ATTRITION_ALPHA = 0.05

# Set the smallest difference in removal rate worth warning about, as a fraction.
#
# Two percentage points. Below that the difference is not something a methods
# section can act on, and on any object above a few thousand cells it is reached
# by noise alone. This gates the WARNING only; the measured difference is always
# recorded, so a reader who cares about a smaller gap can find it in the table.
ATTRITION_MIN_RATE_DIFFERENCE = 0.02

# Set the smallest number of paired blocks that can produce a usable p-value.
#
# The exact Wilcoxon signed-rank test on n pairs cannot go below 2^-n one-sided,
# so with five pairs the smallest attainable two-sided p is 0.0625 -- the test
# cannot reject at 0.05 however consistent the effect. Running it anyway would
# report a non-significant result that says nothing about the data.
MIN_PAIRED_BLOCKS = 6

# Set the multiple-testing procedure applied across subsets.
#
# Benjamini-Hochberg, matching every other multiplicity correction in the engine.
# The subsets of one object are not independent tests of independent hypotheses,
# but they are positively dependent, which is the regime BH controls under.
ATTRITION_FDR_METHOD = "fdr_bh"

# Name the columns of the audit table, so an empty audit still has a schema.
#
# ``subset`` is None on the two cohort-level rows and carries the subset label on
# the stratified ones, which is what lets a reader filter the table down to the
# pre-specified test. ``p_value_adjusted`` is populated only where a correction
# was actually applied, so a populated cell always means "this was one of many".
ATTRITION_COLUMNS: tuple[str, ...] = (
    "factor",
    "subset",
    "unit",
    "test",
    "levels",
    "n_cells",
    "n_removed",
    "removal_rate",
    "rate_difference",
    "odds_ratio",
    "p_value",
    "p_value_adjusted",
    "n_strata",
    "skipped",
)


@dataclass(frozen=True)
class AttritionTest:
    """
    Store one test of whether QC removal is associated with a design factor.

    Args:
        factor: Design factor tested, named as its metadata column.
        subset: Subset of the object the test was restricted to, or None for the
            pre-specified whole-cohort test.
        unit: Unit of analysis -- ``"cell"``, or the blocking column when the
            record is the paired block-level test.
        test: Procedure used, or ``"none"`` when the record was skipped.
        levels: Factor levels, in the order the count tuples follow.
        n_cells: Cells per level. Per-block means for a block-level record.
        n_removed: Cells removed per level.
        removal_rate: Removal rate per level. For a block-level record this is
            the MEAN of the per-block rates, which is not ``n_removed/n_cells``.
        rate_difference: Largest removal-rate gap between any two levels.
        odds_ratio: Odds of removal in ``levels[0]`` relative to ``levels[1]``,
            Mantel-Haenszel pooled when stratified. None above two levels.
        p_value: Test p-value, or None when the record was skipped.
        p_value_adjusted: Benjamini-Hochberg adjusted p-value, set only on subset
            records, where the same question was asked of every subset at once.
            None on the pre-specified cohort records, whose p-value stands as
            computed.
        n_strata: Blocks contributing to a stratified or paired test.
        skipped: Why no test was run, or None when one was.
    """

    # Store the factor and the unit of analysis.
    factor: str
    unit: str

    # Store the procedure used.
    test: str

    # Store the per-level counts.
    levels: tuple[str, ...]
    n_cells: tuple[float, ...]
    n_removed: tuple[float, ...]
    removal_rate: tuple[float, ...]

    # Store the effect size.
    rate_difference: float | None = None
    odds_ratio: float | None = None

    # Store the inference.
    p_value: float | None = None
    n_strata: int | None = None

    # Store the reason no test was run.
    skipped: str | None = None

    # Store which subset the record covers, None meaning the whole cohort.
    subset: str | None = None

    # Store the multiplicity-adjusted p-value, set only on subset records.
    p_value_adjusted: float | None = None

    def to_dict(self) -> dict[str, object]:
        """
        Convert the record into a JSON-friendly dictionary.

        Returns:
            Flat payload keyed by :data:`ATTRITION_COLUMNS`.
        """

        # Return a flat payload suitable for a table row or JSON summary.
        return {
            "factor": self.factor,
            "subset": self.subset,
            "unit": self.unit,
            "test": self.test,
            "levels": list(self.levels),
            "n_cells": list(self.n_cells),
            "n_removed": list(self.n_removed),
            "removal_rate": list(self.removal_rate),
            "rate_difference": self.rate_difference,
            "odds_ratio": self.odds_ratio,
            "p_value": self.p_value,
            "p_value_adjusted": self.p_value_adjusted,
            "n_strata": self.n_strata,
            "skipped": self.skipped,
        }

    def decisive_p_value(self) -> float | None:
        """
        Return the p-value this record should be judged on.

        Returns:
            The adjusted p-value when one was computed, otherwise the raw one, or
            None when the record was skipped. A subset record only ever has an
            adjusted value, so this resolves to "corrected where correction
            applies, raw where the test was pre-specified" without the caller
            having to know which kind of record it holds.
        """

        # Prefer the adjusted value, which exists only where it is required.
        return self.p_value if self.p_value_adjusted is None else self.p_value_adjusted

    def is_significant(self, alpha: float = ATTRITION_ALPHA) -> bool:
        """
        Report whether this record found an association at ``alpha``.

        Args:
            alpha: Significance level.

        Returns:
            True when a p-value was produced and falls below ``alpha``. Subset
            records are judged on the adjusted value.
        """

        # Treat a skipped record as finding nothing.
        p_value = self.decisive_p_value()
        return p_value is not None and p_value < alpha


@dataclass(frozen=True)
class AttritionAudit:
    """
    Store every attrition test run for one QC stage, plus what to warn about.

    Args:
        tests: One record per (factor, unit) pair, including skipped ones.
        warnings: Messages for factors whose attrition is both significantly and
            materially unbalanced.
    """

    # Store the test records.
    tests: list[AttritionTest] = field(default_factory=list)

    # Store the warnings.
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the audit into a table.

        Returns:
            One row per record, with an explicit schema when there are none.
        """

        # Return a schema-aware empty table when nothing was tested.
        if not self.tests:
            return pd.DataFrame(columns=list(ATTRITION_COLUMNS))

        # Return one row per record, in the declared column order.
        return pd.DataFrame([record.to_dict() for record in self.tests])[list(ATTRITION_COLUMNS)]

    def to_summary_dict(self) -> dict[str, object]:
        """
        Convert the audit into a JSON-friendly summary.

        Returns:
            Records, warnings, the factors flagged at cohort level, and the
            subsets flagged within them. The two are kept apart because they mean
            different things: a flagged factor says the cohort is differentially
            filtered, while a flagged subset says one lineage is even when the
            cohort is not -- which is the more common and more easily missed case,
            and would be indistinguishable if both fed one list.
        """

        # Select the records worth flagging: significant on whichever p-value
        # applies to them, and large enough to act on.
        def flagged(record: AttritionTest) -> bool:
            return (
                record.is_significant()
                and record.rate_difference is not None
                and record.rate_difference >= ATTRITION_MIN_RATE_DIFFERENCE
            )

        # Return the payload used by stage metrics and provenance.
        return {
            "tests": [record.to_dict() for record in self.tests],
            "warnings": list(self.warnings),
            "flagged_factors": sorted(
                {
                    record.factor
                    for record in self.tests
                    if record.subset is None and flagged(record)
                }
            ),
            "flagged_subsets": sorted(
                {
                    f"{record.factor}:{record.subset}"
                    for record in self.tests
                    if record.subset is not None and flagged(record)
                }
            ),
        }


def audit_differential_attrition(
    *,
    keep: pd.Series,
    factors: Mapping[str, pd.Series],
    block: pd.Series | None = None,
    subset: pd.Series | None = None,
    contrast: str | None = None,
    alpha: float = ATTRITION_ALPHA,
    min_rate_difference: float = ATTRITION_MIN_RATE_DIFFERENCE,
) -> AttritionAudit:
    """
    Test whether QC removal is associated with any design factor.

    Args:
        keep: Boolean keep decision for every cell that ENTERED QC. Must be
            indexed by the full input cell index, not the surviving subset --
            counted over survivors alone, every removal rate is zero.
        factors: Design factors to test, keyed by name, each a per-cell label
            series aligned to ``keep``.
        block: Optional per-cell blocking label, normally donor. When supplied,
            the cell-level test is stratified on it and a paired block-level test
            is added.
        subset: Optional per-cell subset label, normally cell type. When supplied,
            every test is repeated within each subset and the subset p-values are
            Benjamini-Hochberg adjusted within each (factor, unit) family. A
            cohort rate is an average; the analyses that follow QC are per subset,
            so an imbalance confined to one lineage is both the likeliest case and
            the one pooling hides.
        contrast: Optional name of the factor the downstream analysis actually
            contrasts, normally the condition. Supplying it changes how the OTHER
            factors' warnings are worded, and that distinction matters more than it
            sounds: a cohort with a dozen captures of differing quality will always
            show a significant attrition gap between its best and worst capture, so
            a batch factor raises the same alarm on every real dataset. When every
            batch spans both conditions the gap cannot bias the contrast, and
            saying otherwise trains a reader to ignore the warning that counts.
        alpha: Significance level for the warning.
        min_rate_difference: Smallest removal-rate gap that may raise a warning.

    Returns:
        AttritionAudit holding one record per (factor, subset, unit) and any
        warnings. Cohort records come first, so a reader taking the first row of a
        unit still gets the pre-specified test.
    """

    # Normalize the decision to a boolean removal mask over the input cells.
    removed = ~keep.astype(bool)

    # Resolve the blocking labels once, as strings, keeping missing as missing.
    block_name = str(block.name) if block is not None and block.name is not None else "block"
    block_labels = _as_labels(block, keep.index) if block is not None else None

    # Resolve the subset labels the same way, so an unlabelled cell is excluded
    # from the subset pass rather than forming a subset of its own.
    subset_labels = _as_labels(subset, keep.index) if subset is not None else None

    # Resolve the contrast's labels once, when the caller named a factor that is
    # actually present. A contrast named but absent is treated as unnamed rather
    # than as an error: it only changes wording.
    contrast_labels = (
        _as_labels(factors[contrast], keep.index)
        if contrast is not None and contrast in factors
        else None
    )

    # Collect records and warnings.
    records: list[AttritionTest] = []
    warnings: list[str] = []

    # Audit each factor independently, in a stable order.
    for factor in sorted(factors):
        # Align the factor's labels to the decision.
        labels = _as_labels(factors[factor], keep.index)

        # Report and exclude cells with no label, rather than inventing a level
        # for them or dropping them silently.
        unlabelled = int(labels.isna().sum())
        if unlabelled:
            warnings.append(
                f"{unlabelled} of {len(labels)} cell(s) carry no '{factor}' label, so "
                "they were excluded from its attrition audit. Their QC removal is "
                "unaudited: fill the column or drop those cells before analysis."
            )

        # Restrict to labelled cells.
        usable = labels.notna()
        level_labels = labels[usable]
        level_removed = removed[usable]
        level_block = block_labels[usable] if block_labels is not None else None

        # Run the pre-specified cohort tests.
        cohort_records = _test_factor(
            factor=factor,
            labels=level_labels,
            removed=level_removed,
            block=level_block,
            block_name=block_name,
            subset=None,
        )
        records.extend(cohort_records)

        # Establish what this factor is to the contrast, which decides whether an
        # imbalance is a confounder or is capture quality. This is a property of the
        # DESIGN, so it is measured once over the factor's labelled cells and reused
        # for its subset warnings rather than re-derived per lineage.
        relation = _classify_against_contrast(
            factor=factor,
            labels=level_labels,
            contrast=contrast,
            contrast_labels=contrast_labels[usable] if contrast_labels is not None else None,
        )

        # Warn on the cohort result using the p-value as computed: one factor, one
        # question, nothing to correct for.
        warning = _describe_imbalance(
            cell_record=cohort_records[0],
            paired_record=cohort_records[1] if len(cohort_records) > 1 else None,
            alpha=alpha,
            min_rate_difference=min_rate_difference,
            relation=relation,
        )
        if warning is not None:
            warnings.append(warning)

        # Repeat within each subset, then correct across them.
        if subset_labels is not None:
            records.extend(
                _audit_subsets(
                    factor=factor,
                    labels=level_labels,
                    removed=level_removed,
                    block=level_block,
                    block_name=block_name,
                    subsets=subset_labels[usable],
                    alpha=alpha,
                    min_rate_difference=min_rate_difference,
                    relation=relation,
                    warnings=warnings,
                )
            )

    # Return the assembled audit.
    return AttritionAudit(tests=records, warnings=warnings)


def _test_factor(
    *,
    factor: str,
    labels: pd.Series,
    removed: pd.Series,
    block: pd.Series | None,
    block_name: str,
    subset: str | None,
) -> list[AttritionTest]:
    """
    Run every test one factor supports over one population of cells.

    Args:
        factor: Design factor.
        labels: Per-cell factor labels, already restricted to labelled cells.
        removed: Per-cell removal mask over the same cells.
        block: Optional per-cell blocking labels over the same cells.
        block_name: Name of the blocking column, used as the paired record's unit.
        subset: Subset label to stamp on the records, or None for the cohort.

    Returns:
        The cell-level record, followed by the paired record when the design
        supports one. The order is relied on by the callers, which read the first
        element for the effect size and the second for the reviewer-facing test.
    """

    # Run the cell-level test, stratified when a block was supplied.
    produced = [
        _test_cells(factor=factor, labels=labels, removed=removed, block=block, subset=subset)
    ]

    # Add the paired block-level test when the design supports one.
    if block is not None:
        produced.append(
            _test_blocks(
                factor=factor,
                unit=block_name,
                labels=labels,
                removed=removed,
                block=block,
                subset=subset,
            )
        )
    return produced


def _audit_subsets(
    *,
    factor: str,
    labels: pd.Series,
    removed: pd.Series,
    block: pd.Series | None,
    block_name: str,
    subsets: pd.Series,
    alpha: float,
    min_rate_difference: float,
    relation: _ContrastRelation,
    warnings: list[str],
) -> list[AttritionTest]:
    """
    Repeat one factor's tests within every subset, then correct across subsets.

    Args:
        factor: Design factor.
        labels: Per-cell factor labels.
        removed: Per-cell removal mask.
        block: Optional per-cell blocking labels.
        block_name: Name of the blocking column.
        subsets: Per-cell subset labels, missing values already excluded.
        alpha: Significance level for the warning.
        min_rate_difference: Smallest removal-rate gap that may raise a warning.
        relation: How this factor stands to the contrast, measured on the whole
            cohort. Reused unchanged for every subset because it describes the
            design rather than the lineage.
        warnings: Warning list to append to, in place.

    Returns:
        The subset records, adjusted, grouped by subset in label order.
    """

    # Run the tests within each subset, in a stable order.
    produced: list[AttritionTest] = []
    for level in sorted(subsets.dropna().unique()):
        within = subsets == level
        produced.extend(
            _test_factor(
                factor=factor,
                labels=labels[within],
                removed=removed[within],
                block=block[within] if block is not None else None,
                block_name=block_name,
                subset=str(level),
            )
        )

    # Correct within each unit of analysis. The cell-level and block-level rows
    # are separate families: they answer the same question with different n, and
    # pooling them into one correction would let a long tail of cell-level rows
    # inflate the threshold the reviewer-facing rows are judged against.
    adjusted = _adjust_within_units(produced)

    # Warn per subset, on the adjusted p-value.
    by_subset: dict[str, list[AttritionTest]] = {}
    for record in adjusted:
        by_subset.setdefault(str(record.subset), []).append(record)
    for records in by_subset.values():
        warning = _describe_imbalance(
            cell_record=records[0],
            paired_record=records[1] if len(records) > 1 else None,
            alpha=alpha,
            min_rate_difference=min_rate_difference,
            relation=relation,
        )
        if warning is not None:
            warnings.append(warning)

    return adjusted


def _adjust_within_units(records: list[AttritionTest]) -> list[AttritionTest]:
    """
    Benjamini-Hochberg adjust subset p-values, one family per unit of analysis.

    Args:
        records: Subset records, tested and skipped alike.

    Returns:
        The same records in the same order, with ``p_value_adjusted`` set on those
        that produced a p-value. A skipped record contributes nothing to the
        family: it was never a test, so counting it would only dilute the ones
        that were.
    """

    from statsmodels.stats.multitest import multipletests

    # Group the positions of the tested records by unit.
    families: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        if record.p_value is not None:
            families.setdefault(record.unit, []).append(position)

    # Adjust each family in place, on a copy of the record list.
    adjusted = list(records)
    for positions in families.values():
        # A record with no p_value is a test that did not run; feeding None to float() would
        # crash the whole audit over one absent test, so those positions are dropped from the
        # family rather than corrected.
        positions = [position for position in positions if records[position].p_value is not None]
        if not positions:
            continue
        raw = [float(records[position].p_value or 0.0) for position in positions]
        corrected = multipletests(raw, method=ATTRITION_FDR_METHOD)[1]
        for position, value in zip(positions, corrected, strict=True):
            adjusted[position] = replace(adjusted[position], p_value_adjusted=float(value))
    return adjusted


def audit_qc_stage_attrition(
    *,
    obs: pd.DataFrame,
    keep: pd.Series,
    config: object,
    cohort: object = None,
    design: object = None,
) -> AttritionAudit:
    """
    Run the attrition audit for a QC stage, resolving factors from the config.

    The factors are resolved from the cohort and design blocks rather than named
    per dataset, which is the whole point: the check has to work on the next
    cohort without anyone editing it. A dataset that declares no condition key
    gets an empty audit and no warnings, not an error.

    Args:
        obs: Observation metadata for every cell that ENTERED QC. Must be the
            unfiltered object's obs -- under ``mode="filter"`` the stage's output
            has already lost the removed cells, and every removal rate measured
            on it is zero.
        keep: Boolean keep decision indexed by the same cells.
        config: QC config block exposing ``attrition_audit``.
        cohort: Optional cohort block exposing ``condition_key``, ``batch_key``
            and ``donor_key``.
        design: Optional design block exposing ``condition_col``, ``batch_col``
            and ``donor_col``.

    Returns:
        AttritionAudit for the resolved factors, empty when none resolve. Each
        factor is tested over the whole cohort and, when the object carries a
        cell-type annotation, within each cell type as well.
    """

    # Read the audit settings, tolerating a config that predates the block.
    settings = getattr(config, "attrition_audit", None)
    if settings is not None and not getattr(settings, "enabled", True):
        return AttritionAudit()

    # Resolve the factors to audit, in a deliberate order: the condition first
    # because it is the factor every downstream contrast uses. It is also kept by
    # name, because whether another factor's attrition can bias the study depends on
    # how that factor sits relative to THIS one.
    condition_key = _first_present(
        obs, getattr(cohort, "condition_key", None), getattr(design, "condition_col", None)
    )
    factors: dict[str, pd.Series] = {}
    candidates: list[str | None] = [condition_key]

    # Add the batch key unless it was turned off. Attrition tracking batch is the
    # same defect as attrition tracking condition, and integration cannot fix it:
    # integration aligns the cells that are present.
    if settings is None or getattr(settings, "audit_batch", True):
        candidates.append(
            _first_present(
                obs,
                getattr(cohort, "batch_key", None),
                getattr(design, "batch_col", None),
            )
        )

    # Add whatever else the config named.
    candidates.extend(getattr(settings, "factors", None) or [])

    # Keep the resolvable ones, de-duplicated, without losing the order.
    for candidate in candidates:
        if candidate and candidate in obs.columns and candidate not in factors:
            factors[candidate] = obs[candidate]

    # Nothing to audit is a normal configuration -- a per-lineage single-arm run
    # has no factor -- so return an empty audit rather than raising.
    if not factors:
        return AttritionAudit()

    # Resolve the blocking column, normally the donor.
    block_key = _first_present(
        obs,
        getattr(settings, "block", None),
        getattr(cohort, "donor_key", None),
        getattr(design, "donor_col", None),
    )

    # Never block on a column that is also being tested: stratifying a factor on
    # itself leaves every stratum with one level and no information.
    block = obs[block_key] if block_key and block_key not in factors else None

    # Resolve the subset column, normally the cell-type annotation.
    subset_key = (
        _resolve_subset_key(obs, settings=settings) if _audit_subsets_on(settings) else None
    )

    # Never subset on a column that is being tested or blocked on, for the same
    # reason: a subset of one level, or one donor, carries no comparison.
    subset = (
        obs[subset_key]
        if subset_key and subset_key not in factors and subset_key != block_key
        else None
    )

    # Run the audit with the configured thresholds.
    return audit_differential_attrition(
        keep=keep,
        factors=factors,
        block=block,
        subset=subset,
        contrast=condition_key,
        alpha=getattr(settings, "alpha", ATTRITION_ALPHA),
        min_rate_difference=getattr(settings, "min_rate_difference", ATTRITION_MIN_RATE_DIFFERENCE),
    )


def _audit_subsets_on(settings: object) -> bool:
    """
    Report whether the per-subset pass should run.

    Args:
        settings: Attrition-audit config block, or None on a config predating it.

    Returns:
        True unless the config turned the subset pass off. On by default: it is
        the same arithmetic on a table already in memory, and the imbalance it
        finds is the one that reaches a figure.
    """

    return settings is None or bool(getattr(settings, "audit_subsets", True))


def _resolve_subset_key(obs: pd.DataFrame, *, settings: object) -> str | None:
    """
    Resolve the obs column to stratify the audit by.

    Args:
        obs: Observation metadata to search.
        settings: Attrition-audit config block, possibly naming a column.

    Returns:
        The column name, or None when the object carries no usable annotation --
        which is normal, because QC also runs before annotation.
    """

    # Honour an explicitly named column, and say nothing when it is absent: a
    # config that names a column the object does not have is a config error, and
    # the audit is not the place to raise it.
    named = getattr(settings, "subset", None)
    if named:
        return named if named in obs.columns else None

    # Otherwise fall back to the engine's annotation-column convention. Imported
    # from the figure module rather than restated here so that the audit, the
    # panels and the publication tables all group by the same column -- a table
    # whose rows do not match the figure's rows is worse than no table.
    from cellquorum.visualization.qc.panels import resolve_cell_type_keys

    coarse, _granular = resolve_cell_type_keys(obs)
    return coarse


def audit_qc_design_leaks(
    *,
    config: object,
    cohort: object = None,
    design: object = None,
) -> list[str]:
    """
    Warn when an adaptive QC threshold is estimated per level of a design factor.

    This is the pre-hoc half of this module. The attrition audit measures what a
    rule did; this reads what the rule was asked to do, and refuses two
    configurations that produce differential attrition by construction rather than
    by accident:

    A threshold fit separately per CONDITION cannot be a filter. Whatever the
    metric, estimating the boundary within each arm makes the arms more alike than
    the data are, and the part of the difference that gets absorbed is
    unrecoverable afterwards -- it is not distinguishable from the biology in any
    downstream test. This is wrong for every adaptive rule, unconditionally.

    A threshold fit separately per SAMPLE or DONOR inverts with quality. Both
    adaptive mitochondrial policies in this stage were measured doing it on the
    skin atlas: per-sample MAD set a 2.0% ceiling on the cleanest sample and 11.2%
    on the dirtiest, and adding ``sample_id`` to the mixture grouping made the
    cleanest sample's fibroblasts lose 21.1% of cells at a median of 0.67%
    mitochondrial content. Damage is an absolute state, so what varies between
    samples is the PROPORTION of damaged cells, not the fraction at which damage
    begins.

    Batch is deliberately NOT checked. Grouping thresholds within batch is a
    defensible response to technical variation, unlike grouping within arm, so the
    question of whether it produced differential attrition is left to the
    measurement rather than settled by a rule here.

    Args:
        config: QC config block exposing ``mad`` and ``mito_mixture``.
        cohort: Optional cohort block exposing ``condition_key``, ``sample_key``
            and ``donor_key``.
        design: Optional design block exposing ``condition_col``, ``sample_col``
            and ``donor_col``.

    Returns:
        Warning strings, empty when no adaptive grouping names a design factor.
    """

    # Resolve the design keys by name. Unlike the attrition audit this cannot
    # check obs, because the point is to catch the configuration before the run
    # spends anything -- and a grouping column that is missing from obs is a
    # separate error the grouping code itself raises.
    condition = _first_named(
        getattr(cohort, "condition_key", None), getattr(design, "condition_col", None)
    )
    replicates = {
        name
        for name in (
            getattr(cohort, "sample_key", None),
            getattr(design, "sample_col", None),
            getattr(cohort, "donor_key", None),
            getattr(design, "donor_col", None),
        )
        if name
    }

    # Collect every grouping an enabled adaptive rule will actually use, labelled
    # by the config path a user would have to edit to change it.
    groupings: list[tuple[str, tuple[str, ...]]] = []

    # Read the MAD block. Grouping matters here only for the mitochondrial metric:
    # per-sample grouping of depth-like metrics is standard and defensible, while
    # the mitochondrial pathology above is specific to mitochondrial content.
    mad = getattr(config, "mad", None)
    if mad is not None and getattr(mad, "enabled", True):
        mad_groupby = tuple(getattr(mad, "groupby", None) or ())
        if mad_groupby:
            if getattr(mad, "mito_metric", None):
                groupings.append(("mad.groupby", mad_groupby))

            # With mitochondrial MAD off, only the condition rule applies, so
            # check that alone rather than dropping the grouping entirely.
            elif condition and condition in mad_groupby:
                groupings.append(("mad.groupby", (condition,)))

    # Read the mixture block, including its fallbacks: a fallback level is a
    # grouping that gets used, so naming a design factor there is the same defect
    # arriving later.
    mixture = getattr(config, "mito_mixture", None)
    if mixture is not None and getattr(mixture, "enabled", False):
        groupings.append(("mito_mixture.groupby", tuple(getattr(mixture, "groupby", None) or ())))
        for position, level in enumerate(getattr(mixture, "fallback_groupby", None) or ()):
            groupings.append((f"mito_mixture.fallback_groupby[{position}]", tuple(level or ())))

    # Report each offending grouping once, naming the key and the reason.
    warnings: list[str] = []
    for path, columns in groupings:
        if condition and condition in columns:
            warnings.append(
                f"Design leak in QC: '{path}' groups on '{condition}', which is the "
                "condition every downstream contrast tests. A threshold estimated "
                "within each arm makes the arms more similar than the data are, and "
                "no later test can separate the absorbed difference from biology. "
                f"Remove '{condition}' from '{path}'."
            )
        leaked_replicates = [column for column in columns if column in replicates]
        if leaked_replicates:
            named = ", ".join(f"'{column}'" for column in sorted(leaked_replicates))
            warnings.append(
                f"Design leak in QC: '{path}' groups on {named}, so the threshold is "
                "estimated per replicate and therefore tightens on the CLEANEST "
                "ones -- an adaptive boundary shrinks as the distribution it is "
                "estimated from tightens. Damage is an absolute state: what varies "
                "between replicates is the proportion of damaged cells, not the "
                f"level at which damage begins. Remove {named} from '{path}' and "
                "group on cell identity instead."
            )
    return warnings


def _first_named(*candidates: str | None) -> str | None:
    """
    Pick the first candidate name that is a non-empty string.

    Args:
        candidates: Column names in preference order, possibly None or empty.

    Returns:
        The first usable name, or None.
    """

    # Walk the candidates in the order the caller ranked them.
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _first_present(obs: pd.DataFrame, *candidates: str | None) -> str | None:
    """
    Pick the first candidate column name that exists in ``obs``.

    Args:
        obs: Observation metadata.
        candidates: Column names in preference order, possibly None.

    Returns:
        The first present name, or None.
    """

    # Walk the candidates in the order the caller ranked them.
    for candidate in candidates:
        if candidate and candidate in obs.columns:
            return candidate
    return None


def _as_labels(values: pd.Series, index: pd.Index) -> pd.Series:
    """
    Align a label series to the decision index and normalize it to strings.

    Args:
        values: Per-cell labels.
        index: Index of the QC decision.

    Returns:
        String labels aligned to ``index``, with missing values preserved as NaN.

    Raises:
        AttritionError: If the labels do not cover the decision index.
    """

    # Reindex onto the decision's own index so a partial or reordered series
    # cannot silently mismatch the decisions it is being crossed with.
    aligned = values.reindex(index)

    # Drop the dtype before stringifying, not just the values. obs columns here
    # are routinely categorical, and a categorical carries its full category list
    # regardless of what is present -- a per-lineage object sliced out of an atlas
    # keeps every condition and every donor the atlas had. Grouped on that dtype,
    # pandas emits a row per ABSENT level, and its default for doing so is
    # deprecated and changing. The counts below survive it only because each site
    # happens to reach for `.unique()` or `dropna()`; normalizing once here means
    # the audit's level and stratum counts follow the data rather than the dtype,
    # and no future site has to remember.
    plain = aligned.astype(object)

    # Stringify present values and keep absent ones absent.
    return plain.where(plain.isna(), plain.astype(str))


def _level_counts(
    labels: pd.Series, removed: pd.Series
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """
    Count cells and removals per factor level.

    Args:
        labels: Per-cell factor labels, already restricted to labelled cells.
        removed: Per-cell removal mask.

    Returns:
        Sorted levels, cells per level, and removals per level.
    """

    # Take the levels in sorted order so a record's tuples are comparable
    # between runs.
    levels = tuple(sorted(labels.unique()))

    # Count per level.
    n_cells = np.array([int((labels == level).sum()) for level in levels], dtype=float)
    n_removed = np.array([int(removed[labels == level].sum()) for level in levels], dtype=float)
    return levels, n_cells, n_removed


def _skipped(
    *,
    factor: str,
    unit: str,
    levels: tuple[str, ...],
    n_cells: np.ndarray,
    n_removed: np.ndarray,
    reason: str,
    n_strata: int | None = None,
    subset: str | None = None,
) -> AttritionTest:
    """
    Build a record for a comparison that could not be tested.

    Args:
        factor: Design factor.
        unit: Unit of analysis.
        levels: Factor levels.
        n_cells: Cells per level.
        n_removed: Removals per level.
        reason: Why no test was run.
        n_strata: Blocks available, when relevant to the reason.
        subset: Subset the record covers, or None for the whole cohort.

    Returns:
        Record carrying the counts, the reason, and no p-value.
    """

    # Keep the counts: "nothing was tested" is far more useful next to "and here
    # is what the cohort looked like".
    return AttritionTest(
        factor=factor,
        unit=unit,
        test="none",
        levels=levels,
        n_cells=tuple(n_cells.tolist()),
        n_removed=tuple(n_removed.tolist()),
        removal_rate=tuple(_rates(n_removed, n_cells).tolist()),
        skipped=reason,
        n_strata=n_strata,
        subset=subset,
    )


def _rates(n_removed: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    """
    Divide removals by cells, treating an empty level as a zero rate.

    Args:
        n_removed: Removals per level.
        n_cells: Cells per level.

    Returns:
        Removal rate per level.
    """

    # Guard the division so an empty level reports 0.0 rather than a NaN that
    # would propagate into the effect size.
    return np.divide(
        n_removed,
        n_cells,
        out=np.zeros_like(n_removed, dtype=float),
        where=n_cells > 0,
    )


def _test_cells(
    *,
    factor: str,
    labels: pd.Series,
    removed: pd.Series,
    block: pd.Series | None,
    subset: str | None = None,
) -> AttritionTest:
    """
    Test the association between removal and a factor, with cells as the unit.

    Args:
        factor: Design factor.
        labels: Per-cell factor labels.
        removed: Per-cell removal mask.
        block: Optional per-cell blocking labels to stratify on.
        subset: Subset the record covers, or None for the whole cohort.

    Returns:
        One cell-level record, tested or skipped.
    """

    # Count the cohort first, so even a skipped record carries the numbers.
    levels, n_cells, n_removed = _level_counts(labels, removed)

    # Skip a factor that does not vary.
    if len(levels) < 2:
        return _skipped(
            factor=factor,
            unit="cell",
            levels=levels,
            n_cells=n_cells,
            n_removed=n_removed,
            reason=(
                f"the factor has one level ({levels[0] if levels else 'none'}), so "
                "there is nothing to compare"
            ),
            subset=subset,
        )

    # Skip a decision that does not vary either.
    total_removed = float(n_removed.sum())
    total_cells = float(n_cells.sum())
    if total_removed == 0 or total_removed == total_cells:
        outcome = "no cell was removed" if total_removed == 0 else "every cell was removed"
        return _skipped(
            factor=factor,
            unit="cell",
            levels=levels,
            n_cells=n_cells,
            n_removed=n_removed,
            reason=f"{outcome} by QC, so removal cannot be associated with anything",
            subset=subset,
        )

    # Measure the effect size the same way regardless of which test runs: the
    # widest gap between any two levels' removal rates.
    rates = _rates(n_removed, n_cells)
    rate_difference = float(rates.max() - rates.min())

    # Above two levels, stratification and an odds ratio stop being well defined,
    # so fall back to the omnibus chi-square on the full contingency table.
    if len(levels) > 2:
        from scipy.stats import chi2_contingency

        statistic = chi2_contingency(np.column_stack([n_removed, n_cells - n_removed]))
        return AttritionTest(
            factor=factor,
            unit="cell",
            test="chi_square",
            levels=levels,
            n_cells=tuple(n_cells.tolist()),
            n_removed=tuple(n_removed.tolist()),
            removal_rate=tuple(rates.tolist()),
            rate_difference=rate_difference,
            p_value=float(statistic.pvalue),
            subset=subset,
        )

    # With two levels and a blocking factor, stratify. Donor quality varies far
    # more than QC thresholds do, and donors are rarely balanced across arms, so
    # the pooled table can show an association no donor exhibits.
    if block is not None:
        stratified = _mantel_haenszel(levels=levels, labels=labels, removed=removed, block=block)
        if stratified is not None:
            odds_ratio, p_value, n_strata = stratified
            return AttritionTest(
                factor=factor,
                unit="cell",
                test="cochran_mantel_haenszel",
                levels=levels,
                n_cells=tuple(n_cells.tolist()),
                n_removed=tuple(n_removed.tolist()),
                removal_rate=tuple(rates.tolist()),
                rate_difference=rate_difference,
                odds_ratio=odds_ratio,
                p_value=p_value,
                n_strata=n_strata,
                subset=subset,
            )

    # Otherwise test the pooled two-by-two table exactly.
    from scipy.stats import fisher_exact

    odds_ratio, p_value = fisher_exact(
        [
            [int(n_removed[0]), int(n_cells[0] - n_removed[0])],
            [int(n_removed[1]), int(n_cells[1] - n_removed[1])],
        ]
    )
    return AttritionTest(
        factor=factor,
        unit="cell",
        test="fisher_exact",
        levels=levels,
        n_cells=tuple(n_cells.tolist()),
        n_removed=tuple(n_removed.tolist()),
        removal_rate=tuple(rates.tolist()),
        rate_difference=rate_difference,
        odds_ratio=float(odds_ratio),
        p_value=float(p_value),
        subset=subset,
    )


def _mantel_haenszel(
    *,
    levels: tuple[str, ...],
    labels: pd.Series,
    removed: pd.Series,
    block: pd.Series,
) -> tuple[float, float, int] | None:
    """
    Run the Cochran-Mantel-Haenszel test of removal against a two-level factor.

    Implemented directly rather than through a table object, because the whole
    procedure is two sums and the formulae are worth having in front of the
    reader. Per stratum ``i`` with removals ``a_i`` in the first level, the
    statistic compares the observed total to its null expectation, continuity
    corrected, and is chi-square on one degree of freedom.

    Args:
        levels: The two factor levels, in record order.
        labels: Per-cell factor labels.
        removed: Per-cell removal mask.
        block: Per-cell blocking labels.

    Returns:
        Pooled odds ratio, p-value, and usable stratum count; or None when fewer
        than two strata are informative, in which case there is nothing for
        stratification to buy and the caller should test the pooled table.
    """

    # Accumulate the four MH sums across strata.
    observed = 0.0
    expected = 0.0
    variance = 0.0
    numerator = 0.0
    denominator = 0.0
    n_strata = 0

    # Walk the strata in sorted order for reproducibility.
    for stratum in sorted(block.dropna().unique()):
        # Restrict to this stratum.
        in_stratum = block == stratum
        stratum_labels = labels[in_stratum]
        stratum_removed = removed[in_stratum]

        # Build the 2x2 table: rows are the levels, columns removed/kept.
        first = stratum_labels == levels[0]
        second = stratum_labels == levels[1]
        a = float(stratum_removed[first].sum())
        b = float(first.sum() - a)
        c = float(stratum_removed[second].sum())
        d = float(second.sum() - c)
        total = a + b + c + d

        # Skip strata that carry no information: a stratum missing either level,
        # or in which nothing (or everything) was removed, contributes zero to
        # both the numerator and the variance.
        if total < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        # Accumulate the observed count, its null expectation, and its
        # hypergeometric variance.
        observed += a
        expected += (a + b) * (a + c) / total
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (total**2 * (total - 1.0))

        # Accumulate the Mantel-Haenszel odds-ratio sums.
        numerator += a * d / total
        denominator += b * c / total
        n_strata += 1

    # Decline when stratification has nothing to work with.
    if n_strata < 2 or variance <= 0:
        return None

    # Apply the continuity correction, which keeps the test conservative on the
    # small strata a per-donor design produces.
    from scipy.stats import chi2

    statistic = max(abs(observed - expected) - 0.5, 0.0) ** 2 / variance
    p_value = float(chi2.sf(statistic, df=1))

    # Report the pooled odds ratio, or infinity when no stratum has a discordant
    # pair in the denominator.
    odds_ratio = float(numerator / denominator) if denominator > 0 else float("inf")
    return odds_ratio, p_value, n_strata


def _test_blocks(
    *,
    factor: str,
    unit: str,
    labels: pd.Series,
    removed: pd.Series,
    block: pd.Series,
    subset: str | None = None,
) -> AttritionTest:
    """
    Test the same association with the blocking unit, usually the donor, as the
    unit of analysis.

    One removal rate per (block, level), paired within block. This is the test
    whose n is the number of donors rather than the number of cells, and so the
    only one of the three whose p-value is not inflated by within-donor
    correlation.

    Args:
        factor: Design factor.
        unit: Name of the blocking column, used as the record's unit.
        labels: Per-cell factor labels.
        removed: Per-cell removal mask.
        block: Per-cell blocking labels.
        subset: Subset the record covers, or None for the whole cohort.

    Returns:
        One block-level record, tested or skipped.
    """

    # Count the cohort so a skipped record still carries numbers.
    levels, n_cells, n_removed = _level_counts(labels, removed)

    # A paired test needs exactly two levels to pair.
    if len(levels) != 2:
        return _skipped(
            factor=factor,
            unit=unit,
            levels=levels,
            n_cells=n_cells,
            n_removed=n_removed,
            reason=(
                f"a paired test needs exactly two levels to pair, and the factor has "
                f"{len(levels)}"
            ),
            subset=subset,
        )

    # Build the per-block, per-level removal rate table.
    frame = pd.DataFrame({"block": block, "level": labels, "removed": removed.astype(float)})
    rates = frame.pivot_table(index="block", columns="level", values="removed", aggfunc="mean")

    # Keep only blocks that contributed both levels, which are the only ones a
    # paired test can use.
    paired = rates.dropna()

    # Refuse a paired test that could not reach significance regardless of the
    # data, rather than reporting its foregone non-significance.
    if len(paired) < MIN_PAIRED_BLOCKS:
        return _skipped(
            factor=factor,
            unit=unit,
            levels=levels,
            n_cells=n_cells,
            n_removed=n_removed,
            reason=(
                f"only {len(paired)} {unit}(s) contributed both levels, below the "
                f"{MIN_PAIRED_BLOCKS} pairs an exact signed-rank test needs to be able "
                "to reach p<0.05 at all"
            ),
            n_strata=int(len(paired)),
            subset=subset,
        )

    # Take the per-block mean rate for each level, which is what the paired test
    # actually compares -- deliberately not the pooled cell rate, which weights
    # by block size.
    mean_rates = np.array([float(paired[level].mean()) for level in levels], dtype=float)

    # Run the exact paired test, unless every block moved by exactly zero.
    differences = paired[levels[0]].to_numpy() - paired[levels[1]].to_numpy()
    if not np.any(differences != 0):
        return _skipped(
            factor=factor,
            unit=unit,
            levels=levels,
            n_cells=n_cells,
            n_removed=n_removed,
            reason=f"every {unit} had an identical removal rate in both levels",
            n_strata=int(len(paired)),
            subset=subset,
        )

    from scipy.stats import wilcoxon

    result = wilcoxon(differences)

    # Report the block-level record. n_cells and n_removed stay in cells so the
    # row remains readable next to the cell-level one; removal_rate is the
    # per-block mean, which is the quantity actually tested.
    return AttritionTest(
        factor=factor,
        unit=unit,
        test="wilcoxon_signed_rank",
        levels=levels,
        n_cells=tuple(n_cells.tolist()),
        n_removed=tuple(n_removed.tolist()),
        removal_rate=tuple(mean_rates.tolist()),
        rate_difference=float(abs(mean_rates[0] - mean_rates[1])),
        p_value=float(result.pvalue),
        n_strata=int(len(paired)),
        subset=subset,
    )


@dataclass(frozen=True)
class _ContrastRelation:
    """
    How one design factor stands to the factor the analysis contrasts.

    This exists to answer a single question about a significant attrition gap: can
    it move the comparison the study is actually making? For the contrast itself
    the answer is yes by definition. For anything else it depends on whether the
    factor's levels are crossed with the contrast or nested inside it, and the two
    cases deserve opposite wording.
    """

    # Store the factor this describes and what it was compared against.
    factor: str
    contrast: str | None

    # Store how many of the factor's levels lie entirely within one contrast
    # level, out of how many levels it has. Both zero when there is no contrast.
    n_pure: int
    n_levels: int

    @property
    def is_contrast(self) -> bool:
        """Whether this factor IS the contrast, making any gap a direct confounder."""
        return self.contrast is not None and self.factor == self.contrast

    @property
    def is_crossed(self) -> bool:
        """
        Whether every level spans more than one contrast level.

        A fully crossed factor cannot carry the contrast: each of its levels
        contributes cells to both arms, so losing cells unevenly between levels is
        uneven capture quality and not a shift between the arms being compared.
        """
        return (
            self.contrast is not None
            and not self.is_contrast
            and self.n_levels > 0
            and self.n_pure == 0
        )


def _classify_against_contrast(
    *,
    factor: str,
    labels: pd.Series,
    contrast: str | None,
    contrast_labels: pd.Series | None,
) -> _ContrastRelation:
    """
    Measure whether a factor's levels are crossed with the contrast or nested in it.

    Args:
        factor: The factor being audited.
        labels: Its per-cell labels, restricted to labelled cells.
        contrast: Name of the contrast factor, or None when none was declared.
        contrast_labels: The contrast's per-cell labels over the same cells.

    Returns:
        The relation. With no contrast declared, or when this factor IS the
        contrast, the counts are zero and only the flags carry meaning -- there is
        nothing to cross a factor against but another factor.
    """

    # Return early when there is nothing to compare against.
    if contrast is None or contrast_labels is None or factor == contrast:
        return _ContrastRelation(factor=factor, contrast=contrast, n_pure=0, n_levels=0)

    # Count, per level of the factor, how many contrast levels it contains. A level
    # holding exactly one is "pure": every cell in it belongs to one arm, so the
    # level is nested inside the contrast and its attrition is that arm's attrition.
    spans = contrast_labels.groupby(labels, observed=True).nunique()
    return _ContrastRelation(
        factor=factor,
        contrast=contrast,
        n_pure=int((spans <= 1).sum()),
        n_levels=int(len(spans)),
    )


def _describe_imbalance(
    *,
    cell_record: AttritionTest,
    paired_record: AttritionTest | None,
    alpha: float,
    min_rate_difference: float,
    relation: _ContrastRelation | None = None,
) -> str | None:
    """
    Build the warning for a factor whose attrition is materially unbalanced.

    Args:
        cell_record: The cell-level record for this factor.
        paired_record: The block-level record, when one was produced.
        alpha: Significance level.
        min_rate_difference: Smallest gap that may raise a warning.
        relation: How this factor stands to the contrast, which decides whether the
            message calls the gap a confounder or capture quality. None keeps the
            unconditional wording, which is right when no contrast was declared:
            with nothing named as the comparison, every factor might be it.

    Returns:
        One warning message, or None when there is nothing to warn about.
    """

    # Gather the records that produced a p-value.
    tested = [record for record in (cell_record, paired_record) if record is not None]

    # Require a significant result somewhere.
    if not any(record.is_significant(alpha) for record in tested):
        return None

    # Require the gap to be large enough to act on. Significance alone is reached
    # by noise once there are tens of thousands of cells.
    difference = cell_record.rate_difference
    if difference is None or difference < min_rate_difference:
        return None

    # Name the levels that bound the gap, worst first.
    rates = dict(zip(cell_record.levels, cell_record.removal_rate, strict=True))
    ordered = sorted(rates, key=lambda level: rates[level], reverse=True)
    worst, best = ordered[0], ordered[-1]

    # Summarise each test that ran, naming its unit of analysis and saying plainly
    # which p-value is being quoted. A reader who cannot tell an adjusted value
    # from a raw one cannot tell how much the subset pass corrected for.
    def quote(record: AttritionTest) -> str:
        statistic = (
            f"p={record.p_value:.3g}"
            if record.p_value_adjusted is None
            else f"p_adj={record.p_value_adjusted:.3g} (BH; {record.p_value:.3g} raw)"
        )
        strata = f", n={record.n_strata}" if record.n_strata else ""
        return f"{record.test} {statistic}, {record.unit} as the unit{strata}"

    evidence = "; ".join(quote(record) for record in tested if record.p_value is not None)

    # Name the population, so a subset warning cannot be read as a cohort one.
    where = "" if cell_record.subset is None else f" within '{cell_record.subset}'"

    # Say what the gap MEANS for the comparison the study is making, which is not
    # the same sentence for every factor. A cohort of a dozen captures of differing
    # quality always has a significant best-to-worst attrition gap, so wording that
    # calls every such gap a confounder fires on every real dataset and teaches a
    # reader to skip the one warning that matters.
    if relation is not None and relation.is_crossed:
        consequence = (
            f"All {relation.n_levels} levels of '{cell_record.factor}' contain cells from "
            f"more than one '{relation.contrast}' level, so this is uneven capture "
            f"quality and not a shift between the '{relation.contrast}' arms: each level "
            "loses cells from both. Report the per-level attrition in the methods, and "
            f"read the '{relation.contrast}' rows -- not this one -- for whether the "
            "filter tracked the comparison."
        )
    elif relation is not None and not relation.is_contrast and relation.n_pure:
        consequence = (
            f"{relation.n_pure} of {relation.n_levels} levels of '{cell_record.factor}' "
            f"lie entirely within a single '{relation.contrast}' level, so a gap here is "
            f"partly a '{relation.contrast}' gap and the two cannot be separated. QC that "
            "removes cells at different rates across the design is a covariate, not a "
            "filter: every downstream comparison then partly reflects which cells "
            "survived. Check whether an adaptive threshold was estimated at a level that "
            "varies with this factor, and report the per-arm attrition either way."
        )
    else:
        consequence = (
            "QC that removes cells at different rates in different arms of the design is "
            "a covariate, not a filter: every downstream comparison then partly reflects "
            "which cells survived. Check whether an adaptive threshold was estimated at "
            "a level that varies with this factor, and report the per-arm attrition in "
            "the methods either way."
        )

    # Return the assembled message.
    return (
        f"Differential attrition by '{cell_record.factor}'{where}: QC removed "
        f"{100 * rates[worst]:.1f}% of '{worst}' cells but {100 * rates[best]:.1f}% of "
        f"'{best}' cells, a {100 * difference:.1f}-point gap [{evidence}]. {consequence}"
    )


__all__ = [
    "ATTRITION_ALPHA",
    "ATTRITION_COLUMNS",
    "ATTRITION_FDR_METHOD",
    "ATTRITION_MIN_RATE_DIFFERENCE",
    "MIN_PAIRED_BLOCKS",
    "AttritionAudit",
    "AttritionTest",
    "audit_differential_attrition",
    "audit_qc_design_leaks",
    "audit_qc_stage_attrition",
]
