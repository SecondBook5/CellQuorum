"""Analysis and validation of a specimen-level factorial / repeated-measures design.

``FactorialDesignConfig`` (in :mod:`cellquorum.config.design`) declares the design:
which columns are the specimen and the subject, which factors are modelled, how each
factor relates to the subject structure, and which interactions and random-intercept
grouping variables are requested. This module checks that declaration against the
observed experiment and reports what the experiment actually supports.

Everything here operates on a **specimen table: one row per specimen**, indexed by
specimen id (the shape :func:`build_specimen_universe` produces). Counting cells would
let 10,000 cells pose as 10,000 observations; a duplicated specimen row is therefore
rejected outright rather than counted twice.

Three severities are kept deliberately distinct:

* **structural impossibility** -- the declared design contradicts the data (a
  between-subject factor that changes within a subject, a missing reference level, an
  empty interaction cell, a rank-deficient fixed-effect matrix, a random effect with
  no repeated groups). These are recorded in :attr:`FactorialDesignReport.errors` and
  raised by :func:`validate_factorial_design`.
* **weak replication or incomplete coverage** -- a level backed by a single independent
  unit, a subject missing a within-subject level. These are descriptive
  :attr:`FactorialDesignReport.warnings`; whether they are acceptable is a method or
  support-gate question for later, not a property of the design.
* **uninterpretable input** -- absent columns, missing identifiers, duplicated or
  contradictory specimen rows. No report can be built from these, so
  :func:`analyze_factorial_design` raises immediately.

The fixed-effect matrix covers factors and interactions only. Random-intercept grouping
variables are never added to it, which is what keeps a between-subject factor estimable
alongside a per-subject random effect.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from cellquorum.config.design import (
    FactorialDesignConfig,
    analyze_design,
)
from cellquorum.core.exceptions import CellQuorumConfigError, CellQuorumDataError

# How many offending entities an error message spells out before truncating.
_MAX_REPORTED = 5


@dataclass(frozen=True)
class ResolvedFactorialColumns:
    """The specimen and subject columns after precedence resolution.

    Args:
        sample_col: Column (or index name) identifying the specimen.
        subject_col: Column identifying the subject, or ``None`` when the design has no
            subject structure.
    """

    sample_col: str
    subject_col: str | None


def resolve_factorial_columns(
    config: object, factorial: FactorialDesignConfig
) -> ResolvedFactorialColumns:
    """
    Resolve the specimen and subject columns of a factorial design.

    Precedence, per column: the value declared on ``design.factorial`` wins; otherwise
    the central cohort key (``cohort.sample_key`` for the specimen, ``cohort.donor_key``
    for the subject). A dataset that declares its structural keys once in ``cohort`` need
    not spell them again. This is the single place the precedence lives.

    Note the direction differs from ``resolve_cohort_key``, where the cohort value beats a
    stage value: here the more specific, explicitly declared design column wins, so a
    design can address a different specimen column than the cohort default.

    Args:
        config: A CellQuorumConfig (or anything exposing ``.cohort``), or ``None``.
        factorial: The factorial design declaration.

    Returns:
        The resolved columns.

    Raises:
        CellQuorumConfigError: If no specimen column can be resolved, or the resolved
            specimen and subject columns coincide.
    """

    cohort = getattr(config, "cohort", None)

    sample_col = factorial.sample_col or getattr(cohort, "sample_key", None)
    subject_col = factorial.subject_col or getattr(cohort, "donor_key", None)

    if not sample_col:
        raise CellQuorumConfigError(
            "Factorial design has no specimen column: set design.factorial.sample_col or "
            "cohort.sample_key."
        )
    if subject_col and subject_col == sample_col:
        raise CellQuorumConfigError(
            f"Specimen and subject resolve to the same column '{sample_col}'; a repeated-"
            "measures design needs distinct specimen and subject identifiers."
        )
    return ResolvedFactorialColumns(sample_col=sample_col, subject_col=subject_col or None)


def required_specimen_columns(
    factorial: FactorialDesignConfig, *, sample_col: str, subject_col: str | None
) -> list[str]:
    """
    List the specimen-metadata columns a factorial design needs, for the universe builder.

    Pass the result as ``metadata_cols`` to ``build_specimen_universe`` so the specimen
    table carries the subject, every factor and every random-effect grouping variable
    (the specimen column itself is the table's index, so it is excluded).

    Args:
        factorial: The factorial design declaration.
        sample_col: Resolved specimen column.
        subject_col: Resolved subject column, if any.

    Returns:
        Unique column names, subject first, then factors, then random effects.
    """

    columns = [
        *([subject_col] if subject_col else []),
        *factorial.factors,
        *factorial.random_effects,
    ]
    return [c for c in dict.fromkeys(columns) if c != sample_col]


@dataclass(frozen=True)
class FactorReport:
    """Observed support for one declared factor.

    Args:
        name: Factor (column) name.
        scope: Declared relationship to the subject structure.
        reference: Declared reference level.
        observed_levels: Levels present among the specimens (sorted).
        reference_observed: Whether the declared reference is an observed level.
        n_samples_by_level: Specimens per level.
        n_subjects_by_level: Distinct subjects per level; ``None`` without a subject column.
        subject_levels: Levels each subject shows; ``None`` without a subject column.
        n_subjects_varying: Subjects showing more than one level; ``None`` without subjects.
        constant_within_subject: Whether every subject shows exactly one level.
        complete_subjects: Subjects showing every observed level.
        incomplete_subjects: Subject -> observed levels it lacks.
    """

    name: str
    scope: str
    reference: str
    observed_levels: list[str]
    reference_observed: bool
    n_samples_by_level: dict[str, int]
    n_subjects_by_level: dict[str, int] | None = None
    subject_levels: dict[str, list[str]] | None = None
    n_subjects_varying: int | None = None
    constant_within_subject: bool | None = None
    complete_subjects: list[str] = field(default_factory=list)
    incomplete_subjects: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "scope": self.scope,
            "reference": self.reference,
            "observed_levels": list(self.observed_levels),
            "reference_observed": self.reference_observed,
            "n_samples_by_level": dict(self.n_samples_by_level),
            "n_subjects_by_level": (
                None if self.n_subjects_by_level is None else dict(self.n_subjects_by_level)
            ),
            "subject_levels": (
                None
                if self.subject_levels is None
                else {k: list(v) for k, v in self.subject_levels.items()}
            ),
            "n_subjects_varying": self.n_subjects_varying,
            "constant_within_subject": self.constant_within_subject,
            "complete_subjects": list(self.complete_subjects),
            "incomplete_subjects": {k: list(v) for k, v in self.incomplete_subjects.items()},
        }


@dataclass(frozen=True)
class InteractionReport:
    """Factorial-cell support for one requested two-way interaction.

    Args:
        factors: The two crossed factors, in declaration order.
        cells: Every cell of the observed-level grid, each as
            ``{"levels": {factor: level}, "n_samples": int, "n_subjects": int | None}``.
        empty_cells: The subset of ``cells`` with no specimen.
    """

    factors: tuple[str, str]
    cells: list[dict[str, object]]
    empty_cells: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-safe dictionary."""
        return {
            "factors": list(self.factors),
            "cells": [dict(c) for c in self.cells],
            "empty_cells": [dict(c) for c in self.empty_cells],
        }


@dataclass(frozen=True)
class RandomEffectReport:
    """Grouping structure of one requested random-intercept variable.

    Args:
        column: Grouping variable (a metadata column or the specimen column).
        n_groups: Distinct groups.
        specimens_per_group: Specimens in each group.
        min_specimens: Smallest group size.
        max_specimens: Largest group size.
        n_groups_with_repeats: Groups holding more than one specimen.
    """

    column: str
    n_groups: int
    specimens_per_group: dict[str, int]
    min_specimens: int
    max_specimens: int
    n_groups_with_repeats: int

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-safe dictionary."""
        return {
            "column": self.column,
            "n_groups": self.n_groups,
            "specimens_per_group": dict(self.specimens_per_group),
            "min_specimens": self.min_specimens,
            "max_specimens": self.max_specimens,
            "n_groups_with_repeats": self.n_groups_with_repeats,
        }


@dataclass(frozen=True)
class FactorialDesignReport:
    """Descriptive report of what a specimen table supports for a declared design.

    The report states support; it passes no power judgement. ``errors`` are structural
    impossibilities, ``warnings`` are weak-but-fittable observations.

    Args:
        sample_col: Specimen column.
        subject_col: Subject column, if any.
        n_samples: Specimens.
        n_subjects: Distinct subjects; ``None`` without a subject column.
        factors: Per-factor support, in declaration order.
        interactions: Per-interaction cell support.
        random_effects: Per-grouping-variable structure.
        fixed_effect_columns: Treatment-coded fixed-effect columns (declared references
            dropped); random effects are never included.
        fixed_effect_rank: Numerical rank of the fixed-effect matrix.
        full_rank: Whether the fixed-effect matrix has full column rank.
        confounded_pairs: Perfectly aliased fixed-factor pairs.
        errors: Structural impossibilities.
        warnings: Weak replication / incomplete coverage notes.
    """

    sample_col: str
    subject_col: str | None
    n_samples: int
    n_subjects: int | None
    factors: list[FactorReport]
    interactions: list[InteractionReport]
    random_effects: list[RandomEffectReport]
    fixed_effect_columns: list[str]
    fixed_effect_rank: int
    full_rank: bool
    confounded_pairs: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether no structural impossibility was found."""
        return not self.errors

    def factor(self, name: str) -> FactorReport:
        """Return the report for one factor (``KeyError`` if undeclared)."""
        for report in self.factors:
            if report.name == name:
                return report
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        """Convert the report to a JSON-safe dictionary (provenance)."""
        return {
            "sample_col": self.sample_col,
            "subject_col": self.subject_col,
            "n_samples": self.n_samples,
            "n_subjects": self.n_subjects,
            "factors": [f.to_dict() for f in self.factors],
            "interactions": [i.to_dict() for i in self.interactions],
            "random_effects": [r.to_dict() for r in self.random_effects],
            "fixed_effect_columns": list(self.fixed_effect_columns),
            "fixed_effect_rank": self.fixed_effect_rank,
            "full_rank": self.full_rank,
            "confounded_pairs": [list(p) for p in self.confounded_pairs],
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _check_integrity(
    table: pd.DataFrame,
    *,
    design: FactorialDesignConfig,
    sample_col: str,
    subject_col: str | None,
) -> None:
    """Reject an input from which no meaningful report can be built.

    Raises:
        CellQuorumDataError: Missing/duplicate specimen ids, a specimen mapped to several
            subjects, or missing values in a declared column.
        CellQuorumConfigError: A declared column absent from the table.
    """

    if bool(pd.isna(table.index).any()):
        raise CellQuorumDataError(
            f"Specimen table has missing specimen ids (index '{sample_col}'); every specimen "
            "needs an identifier."
        )
    if table.index.has_duplicates:
        dupes = [str(s) for s in table.index[table.index.duplicated()].unique()]
        if subject_col is not None and subject_col in table.columns:
            n_subjects = table[subject_col].astype(str).groupby(level=0).nunique()
            multi = [str(s) for s in n_subjects.index[(n_subjects > 1).to_numpy()]]
            if multi:
                raise CellQuorumDataError(
                    f"Specimen(s) {multi[:_MAX_REPORTED]} map to more than one subject "
                    f"('{subject_col}'); a specimen belongs to exactly one subject."
                )
        raise CellQuorumDataError(
            f"Duplicate specimen rows for {dupes[:_MAX_REPORTED]}: the table must hold one row "
            "per specimen (a cell-level table would count cells as replicates)."
        )

    needed = [*([subject_col] if subject_col else []), *design.factors]
    needed += [c for c in design.random_effects if c != sample_col]
    absent = [c for c in dict.fromkeys(needed) if c not in table.columns]
    if absent:
        raise CellQuorumConfigError(
            f"Design column(s) {absent} are absent from the specimen table; available: "
            f"{list(table.columns)}."
        )
    for column in dict.fromkeys(needed):
        missing = table.index[table[column].isna().to_numpy()]
        if len(missing):
            shown = [str(m) for m in missing][:_MAX_REPORTED]
            raise CellQuorumDataError(
                f"Column '{column}' is missing for specimen(s) {shown} ({len(missing)} of "
                f"{len(table)}); declared design columns must be fully observed."
            )


def _factor_report(
    name: str,
    spec_scope: str,
    reference: str,
    table: pd.DataFrame,
    subject_col: str | None,
) -> FactorReport:
    """Summarize one factor's levels, replication and subject coverage."""

    levels = table[name].astype(str)
    observed = sorted(levels.unique())
    by_level = {lv: int((levels == lv).sum()) for lv in observed}

    report_kwargs: dict[str, object] = {}
    if subject_col is not None:
        subjects = table[subject_col].astype(str)
        per_subject = levels.groupby(subjects.to_numpy()).agg(lambda s: sorted(set(s)))
        subject_levels = {str(k): list(v) for k, v in per_subject.items()}
        varying = [s for s, lv in subject_levels.items() if len(lv) > 1]
        incomplete = {
            s: [lv for lv in observed if lv not in have]
            for s, have in subject_levels.items()
            if len(have) < len(observed)
        }
        report_kwargs = {
            "n_subjects_by_level": {lv: int(subjects[levels == lv].nunique()) for lv in observed},
            "subject_levels": subject_levels,
            "n_subjects_varying": len(varying),
            "constant_within_subject": not varying,
            "complete_subjects": [s for s in subject_levels if s not in incomplete],
            "incomplete_subjects": incomplete,
        }
    return FactorReport(
        name=name,
        scope=spec_scope,
        reference=reference,
        observed_levels=observed,
        reference_observed=reference in observed,
        n_samples_by_level=by_level,
        **report_kwargs,  # type: ignore[arg-type]
    )


def analyze_factorial_design(
    sample_metadata: pd.DataFrame,
    *,
    design: FactorialDesignConfig,
    sample_col: str,
    subject_col: str | None,
) -> FactorialDesignReport:
    """
    Describe what a specimen table supports for a declared factorial design, without
    raising on structural impossibilities.

    Structural impossibilities and weak-support observations are collected into the
    report's ``errors`` and ``warnings``; use :func:`validate_factorial_design` to halt
    on the former. Only input that cannot be interpreted at all (see the module
    docstring) raises here.

    Args:
        sample_metadata: One row per specimen, indexed by specimen id, holding the
            subject, every factor and every random-effect grouping column (see
            :func:`required_specimen_columns`).
        design: The factorial design declaration.
        sample_col: Resolved specimen column; names the index, and is accepted as a
            random-effect grouping variable.
        subject_col: Resolved subject column, or ``None``.

    Returns:
        A :class:`FactorialDesignReport`.

    Raises:
        CellQuorumDataError: Missing or duplicate specimen ids, a specimen mapped to
            several subjects, or missing values in a declared column.
        CellQuorumConfigError: A declared column absent from the table.
    """

    table = sample_metadata
    _check_integrity(table, design=design, sample_col=sample_col, subject_col=subject_col)

    errors: list[str] = []
    warnings: list[str] = []
    n_samples = len(table)
    n_subjects = int(table[subject_col].astype(str).nunique()) if subject_col else None

    # ---- per-factor support and role checks ---------------------------------------
    factor_reports: list[FactorReport] = []
    for name, spec in design.factors.items():
        rep = _factor_report(name, spec.scope, spec.reference, table, subject_col)
        factor_reports.append(rep)

        if not rep.reference_observed:
            errors.append(
                f"Factor '{name}': reference level '{spec.reference}' is not an observed "
                f"level; observed levels: {rep.observed_levels}."
            )
        if len(rep.observed_levels) < 2:
            errors.append(
                f"Factor '{name}' has a single observed level {rep.observed_levels}; its "
                "effect is not estimable."
            )

        if spec.scope in ("between_subject", "within_subject") and subject_col is None:
            errors.append(
                f"Factor '{name}' is declared {spec.scope} but no subject column is resolved "
                "(set design.factorial.subject_col or cohort.donor_key)."
            )
        elif spec.scope == "between_subject" and rep.subject_levels is not None:
            bad = {s: lv for s, lv in rep.subject_levels.items() if len(lv) > 1}
            if bad:
                shown = "; ".join(
                    f"subject '{s}' shows levels {lv}"
                    for s, lv in list(bad.items())[:_MAX_REPORTED]
                )
                errors.append(
                    f"Factor '{name}' is declared between_subject but varies within a subject: "
                    f"{shown}"
                    + (f" (+{len(bad) - _MAX_REPORTED} more)" if len(bad) > _MAX_REPORTED else "")
                    + "."
                )
        elif spec.scope == "within_subject" and rep.subject_levels is not None:
            if not rep.n_subjects_varying:
                errors.append(
                    f"Factor '{name}' is declared within_subject but no subject shows more than "
                    "one level; it is constant within every subject."
                )
            elif rep.incomplete_subjects:
                shown = ", ".join(
                    f"{s} (lacks {lv})"
                    for s, lv in list(rep.incomplete_subjects.items())[:_MAX_REPORTED]
                )
                warnings.append(
                    f"Within-subject factor '{name}': {len(rep.incomplete_subjects)} of "
                    f"{n_subjects} subject(s) do not show every level: {shown}."
                )

        # Replication is described, never gated, here.
        units = rep.n_subjects_by_level or rep.n_samples_by_level
        unit = "subject" if rep.n_subjects_by_level is not None else "specimen"
        thin = [lv for lv, n in units.items() if n < 2]
        if thin:
            warnings.append(
                f"Factor '{name}': level(s) {thin} are supported by a single independent {unit}."
            )

    # ---- interactions: factorial-cell support ------------------------------------
    interaction_reports: list[InteractionReport] = []
    for a, b in design.interactions:
        la, lb = sorted(table[a].astype(str).unique()), sorted(table[b].astype(str).unique())
        keyed = pd.DataFrame({a: table[a].astype(str), b: table[b].astype(str)})
        if subject_col:
            keyed["_subject"] = table[subject_col].astype(str)
        cells: list[dict[str, object]] = []
        for va, vb in itertools.product(la, lb):
            sel = keyed[(keyed[a] == va) & (keyed[b] == vb)]
            cells.append(
                {
                    "levels": {a: va, b: vb},
                    "n_samples": int(len(sel)),
                    "n_subjects": int(sel["_subject"].nunique()) if subject_col else None,
                }
            )
        empty = [c for c in cells if c["n_samples"] == 0]
        interaction_reports.append(
            InteractionReport(factors=(a, b), cells=cells, empty_cells=empty)
        )
        if empty:
            errors.append(
                f"Interaction {a}:{b} is not estimable: empty factorial cell(s) "
                f"{[c['levels'] for c in empty]}."
            )

    # ---- random-effect grouping structure ----------------------------------------
    re_reports: list[RandomEffectReport] = []
    for column in design.random_effects:
        groups = (
            pd.Series(table.index.astype(str), index=table.index)
            if column == sample_col
            else table[column].astype(str)
        )
        sizes = groups.value_counts()
        rep_re = RandomEffectReport(
            column=column,
            n_groups=int(len(sizes)),
            specimens_per_group={str(k): int(v) for k, v in sizes.sort_index().items()},
            min_specimens=int(sizes.min()),
            max_specimens=int(sizes.max()),
            n_groups_with_repeats=int((sizes > 1).sum()),
        )
        re_reports.append(rep_re)
        if rep_re.n_groups < 2:
            errors.append(
                f"Random effect '{column}' has {rep_re.n_groups} group; a random intercept needs "
                "at least two."
            )
        elif rep_re.n_groups_with_repeats == 0:
            errors.append(
                f"Random effect '{column}' has one specimen per group ({rep_re.n_groups} groups): "
                "group variance is not separable from residual variance, so a random intercept "
                "cannot be estimated."
            )
    within = [n for n, s in design.factors.items() if s.scope == "within_subject"]
    if within and subject_col and subject_col not in design.random_effects:
        warnings.append(
            f"Within-subject factor(s) {within} are declared but subject '{subject_col}' is not "
            "a random effect; repeated specimens would be treated as independent."
        )

    # ---- fixed-effect estimability (random effects are NOT in this matrix) -------
    observed_refs = {
        n: s.reference
        for n, s in design.factors.items()
        if s.reference in set(table[n].astype(str))
    }
    fixed = analyze_design(
        table,
        factors=list(design.factors),
        interactions=[(a, b) for a, b in design.interactions],
        references=observed_refs,
    )
    if not fixed.full_rank:
        detail = (
            " Confounded factor(s): "
            + ", ".join(f"{a} ~ {b}" for a, b in fixed.confounded_pairs)
            + "."
            if fixed.confounded_pairs
            else ""
        )
        errors.append(
            "Fixed-effect design is not estimable: the model matrix is rank-deficient "
            f"({fixed.n_columns} columns, rank {fixed.rank}, {n_samples} specimens)." + detail
        )
    elif n_samples <= fixed.rank:
        warnings.append(
            f"Fixed-effect design has {fixed.rank} parameters for {n_samples} specimens: no "
            "residual degrees of freedom among the fixed effects."
        )

    return FactorialDesignReport(
        sample_col=sample_col,
        subject_col=subject_col,
        n_samples=n_samples,
        n_subjects=n_subjects,
        factors=factor_reports,
        interactions=interaction_reports,
        random_effects=re_reports,
        fixed_effect_columns=list(fixed.columns),
        fixed_effect_rank=fixed.rank,
        full_rank=fixed.full_rank,
        confounded_pairs=list(fixed.confounded_pairs),
        errors=errors,
        warnings=warnings,
    )


def validate_factorial_design(
    sample_metadata: pd.DataFrame,
    *,
    design: FactorialDesignConfig,
    sample_col: str,
    subject_col: str | None,
) -> FactorialDesignReport:
    """
    Validate a factorial design against a specimen table, halting loudly if impossible.

    The raising counterpart of :func:`analyze_factorial_design`, for use at a method
    boundary. Warnings (weak replication, incomplete subjects) never raise.

    Args:
        sample_metadata: One row per specimen, indexed by specimen id.
        design: The factorial design declaration.
        sample_col: Resolved specimen column.
        subject_col: Resolved subject column, or ``None``.

    Returns:
        The :class:`FactorialDesignReport` when no structural impossibility exists.

    Raises:
        CellQuorumConfigError: If the declared design contradicts the data; the message
            lists every violation.
        CellQuorumDataError: See :func:`analyze_factorial_design`.
    """

    report = analyze_factorial_design(
        sample_metadata, design=design, sample_col=sample_col, subject_col=subject_col
    )
    if report.errors:
        raise CellQuorumConfigError(
            "Factorial design is not valid for these specimens: " + " | ".join(report.errors)
        )
    return report


__all__ = [
    "FactorReport",
    "FactorialDesignReport",
    "InteractionReport",
    "RandomEffectReport",
    "ResolvedFactorialColumns",
    "analyze_factorial_design",
    "required_specimen_columns",
    "resolve_factorial_columns",
    "validate_factorial_design",
]
