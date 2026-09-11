"""Design-interaction checks for CellQuorum QC: differential attrition and leaks.

This module asks one question twice -- does QC interact with the study design? --
once before the fact from the configuration (``audit_qc_design_leaks``) and once
after the fact from the decisions (``audit_differential_attrition``). The pre-hoc
check catches the configurations that guarantee the problem; the post-hoc check
measures whatever happened regardless of how the rules were written.

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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

ATTRITION_ALPHA = 0.05


ATTRITION_MIN_RATE_DIFFERENCE = 0.02


MIN_PAIRED_BLOCKS = 6


ATTRITION_FDR_METHOD = "fdr_bh"


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
    """Store one test of whether QC removal is associated with a design factor.

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

    factor: str
    unit: str
    test: str
    levels: tuple[str, ...]
    n_cells: tuple[float, ...]
    n_removed: tuple[float, ...]
    removal_rate: tuple[float, ...]
    rate_difference: float | None = None
    odds_ratio: float | None = None
    p_value: float | None = None
    n_strata: int | None = None
    skipped: str | None = None
    subset: str | None = None
    p_value_adjusted: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the record into a JSON-friendly dictionary.

        Returns:
            Flat payload keyed by :data:`ATTRITION_COLUMNS`.
        """

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
        """Return the p-value this record should be judged on.

        Returns:
            The adjusted p-value when one was computed, otherwise the raw one, or
            None when the record was skipped. A subset record only ever has an
            adjusted value, so this resolves to "corrected where correction
            applies, raw where the test was pre-specified" without the caller
            having to know which kind of record it holds.
        """

        return self.p_value if self.p_value_adjusted is None else self.p_value_adjusted

    def is_significant(self, alpha: float = ATTRITION_ALPHA) -> bool:
        """Report whether this record found an association at ``alpha``.

        Args:
            alpha: Significance level.

        Returns:
            True when a p-value was produced and falls below ``alpha``. Subset
            records are judged on the adjusted value.
        """

        p_value = self.decisive_p_value()
        return p_value is not None and p_value < alpha


@dataclass(frozen=True)
class AttritionAudit:
    """Store every attrition test run for one QC stage, plus what to warn about.

    Args:
        tests: One record per (factor, unit) pair, including skipped ones.
        warnings: Messages for factors whose attrition is both significantly and
            materially unbalanced.
    """

    tests: list[AttritionTest] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the audit into a table.

        Returns:
            One row per record, with an explicit schema when there are none.
        """

        if not self.tests:
            return pd.DataFrame(columns=list(ATTRITION_COLUMNS))

        return pd.DataFrame([record.to_dict() for record in self.tests])[list(ATTRITION_COLUMNS)]

    def to_summary_dict(self) -> dict[str, object]:
        """Convert the audit into a JSON-friendly summary.

        Returns:
            Records, warnings, the factors flagged at cohort level, and the
            subsets flagged within them. The two are kept apart because they mean
            different things: a flagged factor says the cohort is differentially
            filtered, while a flagged subset says one lineage is even when the
            cohort is not -- which is the more common and more easily missed case,
            and would be indistinguishable if both fed one list.
        """

        def flagged(record: AttritionTest) -> bool:
            return (
                record.is_significant()
                and record.rate_difference is not None
                and record.rate_difference >= ATTRITION_MIN_RATE_DIFFERENCE
            )

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
    """Test whether QC removal is associated with any design factor.

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

    removed = ~keep.astype(bool)

    block_name = str(block.name) if block is not None and block.name is not None else "block"
    block_labels = _as_labels(block, keep.index) if block is not None else None

    subset_labels = _as_labels(subset, keep.index) if subset is not None else None

    contrast_labels = (
        _as_labels(factors[contrast], keep.index)
        if contrast is not None and contrast in factors
        else None
    )

    records: list[AttritionTest] = []
    warnings: list[str] = []

    for factor in sorted(factors):
        labels = _as_labels(factors[factor], keep.index)

        unlabelled = int(labels.isna().sum())
        if unlabelled:
            warnings.append(
                f"{unlabelled} of {len(labels)} cell(s) carry no '{factor}' label, so "
                "they were excluded from its attrition audit. Their QC removal is "
                "unaudited: fill the column or drop those cells before analysis."
            )

        usable = labels.notna()
        level_labels = labels[usable]
        level_removed = removed[usable]
        level_block = block_labels[usable] if block_labels is not None else None

        cohort_records = _test_factor(
            factor=factor,
            labels=level_labels,
            removed=level_removed,
            block=level_block,
            block_name=block_name,
            subset=None,
        )
        records.extend(cohort_records)

        relation = _classify_against_contrast(
            factor=factor,
            labels=level_labels,
            contrast=contrast,
            contrast_labels=contrast_labels[usable] if contrast_labels is not None else None,
        )

        warning = _describe_imbalance(
            cell_record=cohort_records[0],
            paired_record=cohort_records[1] if len(cohort_records) > 1 else None,
            alpha=alpha,
            min_rate_difference=min_rate_difference,
            relation=relation,
        )
        if warning is not None:
            warnings.append(warning)

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
    """Run every test one factor supports over one population of cells.

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

    produced = [
        _test_cells(factor=factor, labels=labels, removed=removed, block=block, subset=subset)
    ]

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
    """Repeat one factor's tests within every subset, then correct across subsets.

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

    adjusted = _adjust_within_units(produced)

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
    """Benjamini-Hochberg adjust subset p-values, one family per unit of analysis.

    Args:
        records: Subset records, tested and skipped alike.

    Returns:
        The same records in the same order, with ``p_value_adjusted`` set on those
        that produced a p-value. A skipped record contributes nothing to the
        family: it was never a test, so counting it would only dilute the ones
        that were.
    """

    from statsmodels.stats.multitest import multipletests

    families: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        if record.p_value is not None:
            families.setdefault(record.unit, []).append(position)

    adjusted = list(records)
    for positions in families.values():
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
    """Run the attrition audit for a QC stage, resolving factors from the config.

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

    settings = getattr(config, "attrition_audit", None)
    if settings is not None and not getattr(settings, "enabled", True):
        return AttritionAudit()

    condition_key = _first_present(
        obs, getattr(cohort, "condition_key", None), getattr(design, "condition_col", None)
    )
    factors: dict[str, pd.Series] = {}
    candidates: list[str | None] = [condition_key]

    if settings is None or getattr(settings, "audit_batch", True):
        candidates.append(
            _first_present(
                obs,
                getattr(cohort, "batch_key", None),
                getattr(design, "batch_col", None),
            )
        )

    candidates.extend(getattr(settings, "factors", None) or [])

    for candidate in candidates:
        if candidate and candidate in obs.columns and candidate not in factors:
            factors[candidate] = obs[candidate]

    if not factors:
        return AttritionAudit()

    block_key = _first_present(
        obs,
        getattr(settings, "block", None),
        getattr(cohort, "donor_key", None),
        getattr(design, "donor_col", None),
    )

    block = obs[block_key] if block_key and block_key not in factors else None

    subset_key = (
        _resolve_subset_key(obs, settings=settings) if _audit_subsets_on(settings) else None
    )

    subset = (
        obs[subset_key]
        if subset_key and subset_key not in factors and subset_key != block_key
        else None
    )

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
    """Report whether the per-subset pass should run.

    Args:
        settings: Attrition-audit config block, or None on a config predating it.

    Returns:
        True unless the config turned the subset pass off. On by default: it is
        the same arithmetic on a table already in memory, and the imbalance it
        finds is the one that reaches a figure.
    """

    return settings is None or bool(getattr(settings, "audit_subsets", True))


def _resolve_subset_key(obs: pd.DataFrame, *, settings: object) -> str | None:
    """Resolve the obs column to stratify the audit by.

    Args:
        obs: Observation metadata to search.
        settings: Attrition-audit config block, possibly naming a column.

    Returns:
        The column name, or None when the object carries no usable annotation --
        which is normal, because QC also runs before annotation.
    """

    named = getattr(settings, "subset", None)
    if named:
        return named if named in obs.columns else None

    from cellquorum.visualization.qc.panels import resolve_cell_type_keys

    coarse, _granular = resolve_cell_type_keys(obs)
    return coarse


def audit_qc_design_leaks(
    *,
    config: object,
    cohort: object = None,
    design: object = None,
) -> list[str]:
    """Warn when an adaptive QC threshold is estimated per level of a design factor.

    Args:
        config: QC config block exposing ``mad`` and ``mito_mixture``.
        cohort: Optional cohort block exposing ``condition_key``, ``sample_key``
            and ``donor_key``.
        design: Optional design block exposing ``condition_col``, ``sample_col``
            and ``donor_col``.

    Returns:
        Warning strings, empty when no adaptive grouping names a design factor.
    """

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

    groupings: list[tuple[str, tuple[str, ...]]] = []

    mixture = getattr(config, "mito_mixture", None)
    if mixture is not None and getattr(mixture, "enabled", False):
        groupings.append(("mito_mixture.groupby", tuple(getattr(mixture, "groupby", None) or ())))
        for position, level in enumerate(getattr(mixture, "fallback_groupby", None) or ()):
            groupings.append((f"mito_mixture.fallback_groupby[{position}]", tuple(level or ())))

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
    """Pick the first candidate name that is a non-empty string.

    Args:
        candidates: Column names in preference order, possibly None or empty.

    Returns:
        The first usable name, or None.
    """

    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _first_present(obs: pd.DataFrame, *candidates: str | None) -> str | None:
    """Pick the first candidate column name that exists in ``obs``.

    Args:
        obs: Observation metadata.
        candidates: Column names in preference order, possibly None.

    Returns:
        The first present name, or None.
    """

    for candidate in candidates:
        if candidate and candidate in obs.columns:
            return candidate
    return None


def _as_labels(values: pd.Series, index: pd.Index) -> pd.Series:
    """Align a label series to the decision index and normalize it to strings.

    Args:
        values: Per-cell labels.
        index: Index of the QC decision.

    Returns:
        String labels aligned to ``index``, with missing values preserved as NaN.

    Raises:
        AttritionError: If the labels do not cover the decision index.
    """

    aligned = values.reindex(index)

    plain = aligned.astype(object)

    return plain.where(plain.isna(), plain.astype(str))


def _level_counts(
    labels: pd.Series, removed: pd.Series
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Count cells and removals per factor level.

    Args:
        labels: Per-cell factor labels, already restricted to labelled cells.
        removed: Per-cell removal mask.

    Returns:
        Sorted levels, cells per level, and removals per level.
    """

    levels = tuple(sorted(labels.unique()))

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
    """Build a record for a comparison that could not be tested.

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
    """Divide removals by cells, treating an empty level as a zero rate.

    Args:
        n_removed: Removals per level.
        n_cells: Cells per level.

    Returns:
        Removal rate per level.
    """

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
    """Test the association between removal and a factor, with cells as the unit.

    Args:
        factor: Design factor.
        labels: Per-cell factor labels.
        removed: Per-cell removal mask.
        block: Optional per-cell blocking labels to stratify on.
        subset: Subset the record covers, or None for the whole cohort.

    Returns:
        One cell-level record, tested or skipped.
    """

    levels, n_cells, n_removed = _level_counts(labels, removed)

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

    rates = _rates(n_removed, n_cells)
    rate_difference = float(rates.max() - rates.min())

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
    """Run the Cochran-Mantel-Haenszel test of removal against a two-level factor.

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

    observed = 0.0
    expected = 0.0
    variance = 0.0
    numerator = 0.0
    denominator = 0.0
    n_strata = 0

    for stratum in sorted(block.dropna().unique()):
        in_stratum = block == stratum
        stratum_labels = labels[in_stratum]
        stratum_removed = removed[in_stratum]

        first = stratum_labels == levels[0]
        second = stratum_labels == levels[1]
        a = float(stratum_removed[first].sum())
        b = float(first.sum() - a)
        c = float(stratum_removed[second].sum())
        d = float(second.sum() - c)
        total = a + b + c + d

        if total < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        observed += a
        expected += (a + b) * (a + c) / total
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (total**2 * (total - 1.0))

        numerator += a * d / total
        denominator += b * c / total
        n_strata += 1

    if n_strata < 2 or variance <= 0:
        return None

    from scipy.stats import chi2

    statistic = max(abs(observed - expected) - 0.5, 0.0) ** 2 / variance
    p_value = float(chi2.sf(statistic, df=1))

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
    """Test the same association with the blocking unit, usually the donor, as the
    unit of analysis.

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

    levels, n_cells, n_removed = _level_counts(labels, removed)

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

    frame = pd.DataFrame({"block": block, "level": labels, "removed": removed.astype(float)})
    rates = frame.pivot_table(index="block", columns="level", values="removed", aggfunc="mean")

    paired = rates.dropna()

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

    mean_rates = np.array([float(paired[level].mean()) for level in levels], dtype=float)

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
    """How one design factor stands to the factor the analysis contrasts."""

    factor: str
    contrast: str | None

    n_pure: int
    n_levels: int

    @property
    def is_contrast(self) -> bool:
        """Whether this factor IS the contrast, making any gap a direct confounder."""
        return self.contrast is not None and self.factor == self.contrast

    @property
    def is_crossed(self) -> bool:
        """Whether every level spans more than one contrast level."""
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
    """Measure whether a factor's levels are crossed with the contrast or nested in it.

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

    if contrast is None or contrast_labels is None or factor == contrast:
        return _ContrastRelation(factor=factor, contrast=contrast, n_pure=0, n_levels=0)

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
    """Build the warning for a factor whose attrition is materially unbalanced.

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

    tested = [record for record in (cell_record, paired_record) if record is not None]

    if not any(record.is_significant(alpha) for record in tested):
        return None

    difference = cell_record.rate_difference
    if difference is None or difference < min_rate_difference:
        return None

    rates = dict(zip(cell_record.levels, cell_record.removal_rate, strict=True))
    ordered = sorted(rates, key=lambda level: rates[level], reverse=True)
    worst, best = ordered[0], ordered[-1]

    def quote(record: AttritionTest) -> str:
        statistic = (
            f"p={record.p_value:.3g}"
            if record.p_value_adjusted is None
            else f"p_adj={record.p_value_adjusted:.3g} (BH; {record.p_value:.3g} raw)"
        )
        strata = f", n={record.n_strata}" if record.n_strata else ""
        return f"{record.test} {statistic}, {record.unit} as the unit{strata}"

    evidence = "; ".join(quote(record) for record in tested if record.p_value is not None)

    where = "" if cell_record.subset is None else f" within '{cell_record.subset}'"

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
