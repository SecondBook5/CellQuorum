"""Experimental design + contrasts — the biological question, as config.

DesignConfig captures the donor/condition columns and the primary case/control
comparison; ContrastsConfig lists named comparisons (each with its own
case/control, pairing, and a min_donors power guard). Downstream DE/composition
stages consume these so the biological question is declared once, per dataset.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from cellquorum.config.base import StrictBaseModel
from cellquorum.core.exceptions import CellQuorumConfigError


class DesignConfig(StrictBaseModel):
    """Primary experimental-design settings."""

    # obs column identifying donors (for pairing / pseudobulk grouping).
    donor_col: str = "patient_id"

    # obs column holding the condition label.
    condition_col: str = "condition"

    # Primary case / control condition tokens (optional at config time).
    case: str | None = None
    control: str | None = None

    # Whether the design is donor-paired (drives paired statistics).
    paired: bool = False

    # Optional specimen-level factorial / repeated-measures design. ``None`` (default)
    # leaves every legacy case/control workflow exactly as it was.
    factorial: FactorialDesignConfig | None = None


# Where a factor sits relative to the biological subject (donor, mouse, patient).
FactorScope = Literal["between_subject", "within_subject", "sample_level"]


class FactorSpec(StrictBaseModel):
    """One categorical model factor of a factorial design.

    The factor's name is the key it is declared under in
    :attr:`FactorialDesignConfig.factors`, and is the name of the specimen-metadata
    column holding its levels.

    ``scope`` states how the factor relates to the subject structure:

    * ``between_subject`` -- constant within every subject (e.g. genotype). A subject
      that shows two levels is a declared-vs-observed mismatch, not a rank problem.
    * ``within_subject`` -- intended to vary within subjects (e.g. tissue sampled
      from each animal). At least one subject must actually show more than one level;
      incomplete subjects are reported, not rejected.
    * ``sample_level`` -- no claim about subject structure; the factor is validated at
      the specimen level only (e.g. a processing batch). Default, so a factor is never
      silently treated as repeated-measures.

    Attributes:
        reference: Level every other level is compared against; must be an observed
            level (checked against the data at run time).
        scope: Relationship to the subject structure.
    """

    reference: str = Field(min_length=1)
    scope: FactorScope = "sample_level"


class FactorialDesignConfig(StrictBaseModel):
    """Specimen-level factorial / repeated-measures design.

    Describes the objects of a repeated-measures experiment explicitly::

        cell -> specimen (sample_col) -> subject (subject_col)

    Specimens carry factor values; a subject induces dependence among its repeated
    specimens. This model only *represents* the design; it fits nothing. The fixed-effect
    part is the main effects of ``factors`` plus ``interactions``; ``random_effects``
    names the grouping variables of the eventual random intercepts and is kept distinct
    from ``subject_col`` (they coincide for a simple repeated-measures study but need
    not: litter, site and visit are all legitimate grouping variables).

    Attributes:
        sample_col: Column identifying the specimen. ``None`` falls back to
            ``cohort.sample_key`` (see ``resolve_factorial_columns``).
        subject_col: Column identifying the biological subject. ``None`` falls back to
            ``cohort.donor_key``; may stay unresolved for a design with no subject
            structure (then no factor may claim a between/within scope).
        factors: Model factors keyed by specimen-metadata column name.
        interactions: Two-way interaction terms among declared factors.
        random_effects: Random-intercept grouping variables (a specimen-metadata column,
            or the sample column itself).
    """

    sample_col: str | None = Field(default=None, min_length=1)
    subject_col: str | None = Field(default=None, min_length=1)
    factors: dict[str, FactorSpec]
    interactions: list[tuple[str, str]] = []
    random_effects: list[str] = []

    @model_validator(mode="after")
    def _validate_structure(self) -> FactorialDesignConfig:
        """Reject structurally impossible declarations before any data is read."""

        if not self.factors:
            raise ValueError("a factorial design needs at least one factor.")
        if any(not name for name in self.factors):
            raise ValueError("factor names must be non-empty column names.")
        if self.sample_col is not None and self.sample_col == self.subject_col:
            raise ValueError(
                f"sample_col and subject_col are both '{self.sample_col}': a repeated-"
                "measures design needs distinct specimen and subject identifiers."
            )

        seen: set[frozenset[str]] = set()
        for a, b in self.interactions:
            for member in (a, b):
                if member not in self.factors:
                    raise ValueError(
                        f"interaction ({a}, {b}) references '{member}', which is not a "
                        f"declared factor ({sorted(self.factors)})."
                    )
            if a == b:
                raise ValueError(f"interaction ({a}, {b}) crosses a factor with itself.")
            key = frozenset((a, b))
            if key in seen:
                raise ValueError(f"interaction ({a}, {b}) is declared twice (order is irrelevant).")
            seen.add(key)

        if len(set(self.random_effects)) != len(self.random_effects):
            raise ValueError(f"random_effects contains duplicates: {self.random_effects}.")
        if any(not name for name in self.random_effects):
            raise ValueError("random_effects entries must be non-empty column names.")
        return self


DesignConfig.model_rebuild()


class Contrast(StrictBaseModel):
    """One named case-vs-control comparison."""

    # Stable contrast name.
    name: str

    # Case / control condition tokens for this comparison.
    case: str
    control: str

    # Whether this comparison is donor-paired. None means INHERIT `design.paired`,
    # and is the default on purpose: a contrast is a refinement of the design, not a
    # replacement for it, so a field the author never wrote must not decide a
    # statistical question. With a plain `False` default, adding a named contrast to a
    # matched cohort silently turned pairing OFF -- donor baseline variance stayed in
    # the residual and real effects came back as nulls. Only an explicit `paired: false`
    # overrides a paired design, which is a visible act a reviewer can see in the
    # config. Set it explicitly when a contrast genuinely is unpaired (e.g. a
    # cross-cohort comparison inside an otherwise within-donor study).
    paired: bool | None = None

    # Minimum distinct donors required for this comparison (power guard).
    min_donors: int = 0


class SimpleEffectContrast(StrictBaseModel):
    """A level-vs-level effect of one factor, optionally at fixed levels of others.

    Declared in biological level terms; no backend coefficient name ever appears.
    The estimand is ``E[numerator | at] - E[denominator | at]``.

    Attributes:
        kind: Discriminator, always ``"simple_effect"``.
        name: Stable contrast name (output files and result columns carry it).
        factor: The factor whose two levels are compared.
        numerator: Level on the positive side.
        denominator: Level on the negative side.
        at: Levels of other factors the effect is evaluated at. A factor that
            interacts with ``factor`` MUST be given here; the effect differs by its
            level, so it is never silently read at the reference.
    """

    kind: Literal["simple_effect"]
    name: str = Field(min_length=1)
    factor: str = Field(min_length=1)
    numerator: str = Field(min_length=1)
    denominator: str = Field(min_length=1)
    at: dict[str, str] = {}

    @model_validator(mode="after")
    def _validate_structure(self) -> SimpleEffectContrast:
        """Reject a contrast that compares a level with itself or fixes its own factor."""

        if self.numerator == self.denominator:
            raise ValueError(
                f"contrast '{self.name}': numerator and denominator are both "
                f"'{self.numerator}'; the effect is identically zero."
            )
        if self.factor in self.at:
            raise ValueError(
                f"contrast '{self.name}': 'at' fixes '{self.factor}', the factor being compared."
            )
        return self


class DifferenceOfDifferencesContrast(StrictBaseModel):
    """The interaction estimand: how the effect of factor A differs across factor B.

    ``(A=na,B=nb) - (A=na,B=db) - (A=da,B=nb) + (A=da,B=db)``, i.e. the effect of
    ``factor_a`` (numerator_a vs denominator_a) at ``numerator_b`` minus the same
    effect at ``denominator_b``. Reversing either pair flips the sign.

    Attributes:
        kind: Discriminator, always ``"difference_of_differences"``.
        name: Stable contrast name.
        factor_a: First factor.
        numerator_a: Positive level of ``factor_a``.
        denominator_a: Negative level of ``factor_a``.
        factor_b: Second factor.
        numerator_b: Positive level of ``factor_b``.
        denominator_b: Negative level of ``factor_b``.
        at: Levels of any further factors the contrast is evaluated at.
    """

    kind: Literal["difference_of_differences"]
    name: str = Field(min_length=1)
    factor_a: str = Field(min_length=1)
    numerator_a: str = Field(min_length=1)
    denominator_a: str = Field(min_length=1)
    factor_b: str = Field(min_length=1)
    numerator_b: str = Field(min_length=1)
    denominator_b: str = Field(min_length=1)
    at: dict[str, str] = {}

    @model_validator(mode="after")
    def _validate_structure(self) -> DifferenceOfDifferencesContrast:
        """Reject a degenerate difference of differences."""

        if self.factor_a == self.factor_b:
            raise ValueError(
                f"contrast '{self.name}': factor_a and factor_b are both '{self.factor_a}'; "
                "a difference of differences needs two distinct factors."
            )
        if self.numerator_a == self.denominator_a or self.numerator_b == self.denominator_b:
            raise ValueError(
                f"contrast '{self.name}': a numerator equals its denominator; the contrast "
                "is identically zero."
            )
        clash = {self.factor_a, self.factor_b}.intersection(self.at)
        if clash:
            raise ValueError(
                f"contrast '{self.name}': 'at' fixes compared factor(s) {sorted(clash)}."
            )
        return self


# Discriminated union of the structured factorial estimands.
FactorialContrast = Annotated[
    SimpleEffectContrast | DifferenceOfDifferencesContrast, Field(discriminator="kind")
]


class ContrastsConfig(StrictBaseModel):
    """A list of named comparisons."""

    # The registered legacy case/control contrasts (untouched by factorial designs).
    contrasts: list[Contrast] = []

    # Structured factorial estimands, declared in factor-level terms. Kept separate from
    # the legacy pairwise list: an old `Contrast` never means a factorial one.
    factorial: list[FactorialContrast] = []

    @model_validator(mode="after")
    def _validate_unique_names(self) -> ContrastsConfig:
        """Contrast names address outputs, so a repeat (in either list) is a config error."""

        # Duplicates among the legacy contrasts alone keep their existing, separate check
        # (CellQuorumConfig); only repeats that involve a factorial contrast are caught here.
        legacy = [c.name for c in self.contrasts]
        factorial = [c.name for c in self.factorial]
        duplicates = sorted({n for n in factorial if factorial.count(n) > 1 or n in legacy})
        if duplicates:
            raise ValueError(f"contrast name(s) {duplicates} are declared more than once.")
        return self

    def get(self, name: str) -> Contrast:
        """
        Return a named contrast, or raise if unknown.

        Args:
            name: Contrast name.

        Returns:
            The matching Contrast.

        Raises:
            CellQuorumConfigError: If no contrast has that name.
        """

        # Linear search is fine for the small number of contrasts per dataset.
        for contrast in self.contrasts:
            if contrast.name == name:
                return contrast
        known = [c.name for c in self.contrasts]
        raise CellQuorumConfigError(f"Unknown contrast '{name}'. Registered: {known}.")


@dataclass(frozen=True)
class DesignValidationResult:
    """
    Store the estimability checks for one declared comparison.

    Args:
        case: Case condition label.
        control: Control condition label.
        paired: Whether the comparison is paired.
        case_donors: Distinct donors contributing case observations.
        control_donors: Distinct donors contributing control observations.
        complete_pair_donors: Donors contributing both case and control observations.
        warnings: Non-fatal design caveats.
    """

    case: str

    control: str

    paired: bool

    case_donors: set[str] = field(default_factory=set)

    control_donors: set[str] = field(default_factory=set)

    complete_pair_donors: set[str] = field(default_factory=set)

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the validation result to a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the validation result.
        """

        return {
            "case": self.case,
            "control": self.control,
            "paired": self.paired,
            "case_donors": sorted(self.case_donors),
            "control_donors": sorted(self.control_donors),
            "complete_pair_donors": sorted(self.complete_pair_donors),
            "warnings": list(self.warnings),
        }


def validate_design_against_obs(
    obs: pd.DataFrame,
    *,
    design: DesignConfig,
    contrast: Contrast | None = None,
    min_donors_per_arm: int = 2,
) -> DesignValidationResult:
    """
    Validate that a declared comparison is estimable from observation metadata.

    This guard is intended for DE, DA, pseudobulk, and other statistical stages.
    It checks for missing metadata, absent condition levels, donor replication,
    minimum donor counts, paired completeness, and obvious non-estimable donor
    structures before a statistical method can run.

    Args:
        obs: AnnData ``obs`` table.
        design: Project-level design declaration.
        contrast: Optional named contrast. When absent, ``design.case`` and
            ``design.control`` define the comparison.
        min_donors_per_arm: Minimum donors per condition arm when the contrast
            does not declare a stricter ``min_donors`` value.

    Returns:
        Structured validation result.

    Raises:
        CellQuorumConfigError: If the comparison is not estimable.
    """

    # Validate required metadata columns before looking at the contrast.
    for column in (design.donor_col, design.condition_col):
        if column not in obs.columns:
            raise CellQuorumConfigError(
                f"Design column '{column}' is missing from obs; available columns: "
                f"{list(obs.columns)}."
            )

    # Resolve the active comparison.
    case = contrast.case if contrast is not None else design.case
    control = contrast.control if contrast is not None else design.control
    # Resolve pairing by INHERITANCE, not by override. `Contrast.paired` defaults to
    # None ("as the design says") so a contrast that never mentions pairing cannot
    # downgrade a matched design to an unpaired fit; see Contrast.paired.
    paired = design.paired if contrast is None or contrast.paired is None else contrast.paired
    declared_min_donors = contrast.min_donors if contrast is not None else 0
    if not case or not control:
        raise CellQuorumConfigError(
            "A statistical comparison requires case and control labels in design "
            "or in a named contrast."
        )
    if case == control:
        raise CellQuorumConfigError("Design case and control labels must be different.")

    # Drop rows that cannot contribute to the comparison.
    donor = obs[design.donor_col]
    condition = obs[design.condition_col]
    comparison_mask = condition.isin([case, control]) & donor.notna() & condition.notna()
    comparison_obs = obs.loc[comparison_mask, [design.donor_col, design.condition_col]].copy()
    if comparison_obs.empty:
        raise CellQuorumConfigError(
            f"No observations match case/control labels '{case}' and '{control}'."
        )

    # Confirm both arms exist.
    observed_conditions = set(comparison_obs[design.condition_col].astype(str))
    missing_conditions = {case, control} - observed_conditions
    if missing_conditions:
        raise CellQuorumConfigError(
            f"Comparison is missing condition level(s): {sorted(missing_conditions)}."
        )

    # Convert donor ids to strings for stable reporting.
    comparison_obs[design.donor_col] = comparison_obs[design.donor_col].astype(str)
    comparison_obs[design.condition_col] = comparison_obs[design.condition_col].astype(str)

    # Compute donor support by condition.
    case_donors = set(
        comparison_obs.loc[
            comparison_obs[design.condition_col] == case,
            design.donor_col,
        ]
    )
    control_donors = set(
        comparison_obs.loc[
            comparison_obs[design.condition_col] == control,
            design.donor_col,
        ]
    )
    complete_pair_donors = case_donors & control_donors

    # Require independent donor replication in each arm.
    required_per_arm = max(1, min_donors_per_arm)
    if len(case_donors) < required_per_arm or len(control_donors) < required_per_arm:
        raise CellQuorumConfigError(
            "Comparison lacks donor replication per condition arm: "
            f"{case} has {len(case_donors)} donor(s), {control} has "
            f"{len(control_donors)} donor(s), required per arm is {required_per_arm}."
        )

    # Enforce the contrast-level power guard against total distinct donors.
    total_donors = case_donors | control_donors
    if declared_min_donors and len(total_donors) < declared_min_donors:
        raise CellQuorumConfigError(
            f"Contrast requires at least {declared_min_donors} distinct donors, "
            f"but only {len(total_donors)} are present."
        )

    warnings: list[str] = []

    # Paired designs must have complete donor pairs.
    if paired:
        incomplete_case = case_donors - complete_pair_donors
        incomplete_control = control_donors - complete_pair_donors
        if incomplete_case or incomplete_control:
            raise CellQuorumConfigError(
                "Paired comparison has incomplete donor pairs. "
                f"Case-only donors: {sorted(incomplete_case)}; "
                f"control-only donors: {sorted(incomplete_control)}."
            )
        if declared_min_donors and len(complete_pair_donors) < declared_min_donors:
            raise CellQuorumConfigError(
                f"Paired contrast requires at least {declared_min_donors} complete "
                f"donor pairs, but only {len(complete_pair_donors)} are present."
            )
    elif complete_pair_donors:
        warnings.append(
            "Some donors appear in both condition arms but the contrast is not marked paired."
        )

    # Detect a design that cannot separate donor identity from condition.
    donor_condition_counts = comparison_obs.groupby(design.donor_col)[
        design.condition_col
    ].nunique()
    if paired and (donor_condition_counts < 2).any():
        bad_donors = sorted(donor_condition_counts[donor_condition_counts < 2].index)
        raise CellQuorumConfigError(
            "Paired design is confounded by donor identity for donor(s): " f"{bad_donors}."
        )

    return DesignValidationResult(
        case=case,
        control=control,
        paired=paired,
        case_donors=case_donors,
        control_donors=control_donors,
        complete_pair_donors=complete_pair_donors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# General (multi-factor / factorial) design estimability
#
# The pairwise path above answers "is this two-level case/control comparison
# estimable?". A factorial design (multiple crossed factors, optional
# interactions) can fail in ways a two-level check cannot see: a rank-deficient
# model matrix, two factors that are perfectly confounded (aliased), or an empty
# cell in the factorial grid that leaves an interaction inestimable. The helpers
# below build the treatment-coded model matrix from arbitrary factor columns and
# surface those failures explicitly, in the same fail-loud spirit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignMatrixReport:
    """Estimability diagnostics for a multi-factor design matrix.

    Args:
        factors: The main-effect factor columns, in order.
        interactions: Two-way interaction terms as ``(factor_a, factor_b)`` pairs.
        n_samples: Number of rows (samples/pseudo-samples) the matrix was built from.
        n_columns: Number of model-matrix columns (intercept + coded terms).
        rank: Numerical rank of the model matrix.
        full_rank: Whether ``rank == n_columns`` (the design is estimable).
        confounded_pairs: Main-effect factor pairs that are perfectly aliased
            (one factor nested within the other), so their effects cannot be
            separated.
        empty_cells: Factorial-grid cells (as ``{factor: level}`` dicts) with
            fewer than ``min_per_cell`` samples.
        warnings: Human-readable summaries of any problems found.
    """

    # Main-effect factor columns, in order.
    factors: list[str]

    # Two-way interaction terms, as ordered factor pairs.
    interactions: list[tuple[str, str]] = field(default_factory=list)

    # Rows the matrix was built from.
    n_samples: int = 0

    # Model-matrix columns (intercept + coded terms).
    n_columns: int = 0

    # Numerical rank of the model matrix.
    rank: int = 0

    # Whether the matrix has full column rank (estimable).
    full_rank: bool = True

    # Perfectly-aliased main-effect factor pairs.
    confounded_pairs: list[tuple[str, str]] = field(default_factory=list)

    # Factorial-grid cells below the per-cell minimum.
    empty_cells: list[dict[str, str]] = field(default_factory=list)

    # Non-empty when a problem was found.
    warnings: list[str] = field(default_factory=list)

    # Model-matrix column names, in order (Intercept first).
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert the report to a JSON-friendly dictionary."""

        return {
            "columns": list(self.columns),
            "factors": list(self.factors),
            "interactions": [list(pair) for pair in self.interactions],
            "n_samples": self.n_samples,
            "n_columns": self.n_columns,
            "rank": self.rank,
            "full_rank": self.full_rank,
            "confounded_pairs": [list(pair) for pair in self.confounded_pairs],
            "empty_cells": [dict(cell) for cell in self.empty_cells],
            "warnings": list(self.warnings),
        }


def build_design_matrix(
    sample_meta: pd.DataFrame,
    factors: list[str],
    interactions: list[tuple[str, str]] | None = None,
    references: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a treatment-coded model matrix from categorical factor columns.

    Every factor is dummy-coded with the first level dropped (treatment coding),
    an intercept column is prepended, and each requested two-way interaction adds
    the products of its two factors' surviving dummy columns. This is the same
    parameterization ``model.matrix(~ a + b + a:b)`` produces in R, so the rank of
    the returned matrix is the rank the downstream fit will see.

    Args:
        sample_meta: Per-sample metadata (rows are samples/pseudo-samples).
        factors: Main-effect factor columns to include, in order.
        interactions: Optional two-way interaction terms as ``(a, b)`` pairs; each
            member is coded even if it is not also listed in ``factors``.
        references: Optional ``{factor: level}`` reference levels to drop instead of
            the alphabetically first one, so the coded columns are contrasts against
            the declared reference. Rank is unaffected; the column names are.

    Returns:
        A DataFrame of model-matrix columns aligned to ``sample_meta.index``.

    Raises:
        CellQuorumConfigError: If any referenced column is absent from
            ``sample_meta``, or a declared reference is not an observed level.
    """

    interactions = list(interactions or [])

    # Every referenced column (main effects + interaction members) must exist.
    referenced = list(dict.fromkeys([*factors, *(m for pair in interactions for m in pair)]))
    missing = [c for c in referenced if c not in sample_meta.columns]
    if missing:
        raise CellQuorumConfigError(
            f"Design factor column(s) missing from metadata: {missing}; "
            f"available columns: {list(sample_meta.columns)}."
        )

    n = len(sample_meta)
    index = sample_meta.index

    # Treatment-coded dummies per referenced factor (first level dropped).
    dummies: dict[str, pd.DataFrame] = {}
    references = references or {}
    for column in referenced:
        levels = sample_meta[column].astype(str)
        reference = references.get(column)
        if reference is None:
            coded = pd.get_dummies(levels, prefix=column, drop_first=True).astype(float)
        else:
            if reference not in set(levels):
                raise CellQuorumConfigError(
                    f"Reference level '{reference}' of factor '{column}' is not observed; "
                    f"observed levels: {sorted(set(levels))}."
                )
            coded = pd.get_dummies(levels, prefix=column).astype(float)
            coded = coded.drop(columns=f"{column}_{reference}")
        coded.index = index
        dummies[column] = coded

    blocks: list[pd.DataFrame] = [pd.DataFrame({"Intercept": np.ones(n, dtype=float)}, index=index)]

    # Main-effect blocks (a single-level factor contributes no columns).
    for column in factors:
        if not dummies[column].empty:
            blocks.append(dummies[column])

    # Two-way interaction blocks: products of the coded dummy columns.
    for a, b in interactions:
        da, db = dummies[a], dummies[b]
        products: dict[str, np.ndarray] = {}
        for ca in da.columns:
            for cb in db.columns:
                products[f"{ca}:{cb}"] = da[ca].to_numpy() * db[cb].to_numpy()
        if products:
            blocks.append(pd.DataFrame(products, index=index))

    return pd.concat(blocks, axis=1)


def _factors_confounded(sample_meta: pd.DataFrame, a: str, b: str) -> bool:
    """Return whether two multi-level factors are perfectly aliased (nested)."""

    sa = sample_meta[a].astype(str)
    sb = sample_meta[b].astype(str)

    # A single-level factor absorbs into the intercept; it is degenerate, not
    # confounding, so it takes two informative factors to be aliased.
    if sa.nunique() < 2 or sb.nunique() < 2:
        return False

    # a is nested in b when every level of a co-occurs with exactly one level of
    # b (and vice versa); either direction makes the two effects inseparable.
    a_nested_in_b = bool((sample_meta.groupby(sa)[b].nunique() <= 1).all())
    b_nested_in_a = bool((sample_meta.groupby(sb)[a].nunique() <= 1).all())
    return a_nested_in_b or b_nested_in_a


def _empty_factorial_cells(
    sample_meta: pd.DataFrame, factors: list[str], min_per_cell: int
) -> list[dict[str, str]]:
    """List cells of the full factorial grid with fewer than ``min_per_cell`` rows."""

    if not factors:
        return []

    levels = [sorted(sample_meta[f].astype(str).unique()) for f in factors]
    coded = sample_meta[list(factors)].astype(str)
    counts = coded.groupby(list(factors)).size()

    empty: list[dict[str, str]] = []
    for combo in itertools.product(*levels):
        key = combo if len(factors) > 1 else combo[0]
        if int(counts.get(key, 0)) < min_per_cell:
            empty.append(dict(zip(factors, combo, strict=True)))
    return empty


def analyze_design(
    sample_meta: pd.DataFrame,
    *,
    factors: list[str],
    interactions: list[tuple[str, str]] | None = None,
    min_per_cell: int = 1,
    references: dict[str, str] | None = None,
) -> DesignMatrixReport:
    """
    Diagnose the estimability of a multi-factor design without raising.

    Builds the model matrix, measures its rank, and reports confounded factor
    pairs and empty factorial cells. This is the non-raising analyzer; use
    :func:`validate_design_matrix` at a stage boundary to halt on a non-estimable
    design.

    Args:
        sample_meta: Per-sample metadata.
        factors: Main-effect factor columns.
        interactions: Optional two-way interaction terms.
        min_per_cell: Minimum samples required per factorial-grid cell.
        references: Optional ``{factor: level}`` reference levels (see
            :func:`build_design_matrix`).

    Returns:
        A :class:`DesignMatrixReport` describing the design.

    Raises:
        CellQuorumConfigError: If a referenced column is absent (from
            :func:`build_design_matrix`).
    """

    interactions = list(interactions or [])
    design_matrix = build_design_matrix(sample_meta, factors, interactions, references)

    matrix = design_matrix.to_numpy(dtype=float)
    n_columns = int(design_matrix.shape[1])
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    full_rank = rank == n_columns

    confounded: list[tuple[str, str]] = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            if _factors_confounded(sample_meta, factors[i], factors[j]):
                confounded.append((factors[i], factors[j]))

    empty_cells = _empty_factorial_cells(sample_meta, factors, min_per_cell)

    warnings: list[str] = []
    if not full_rank:
        warnings.append(f"Design matrix is rank-deficient: {n_columns} columns but rank {rank}.")
    if confounded:
        warnings.append(
            "Confounded (aliased) factor pair(s): "
            + ", ".join(f"{a} ~ {b}" for a, b in confounded)
            + "."
        )
    if empty_cells:
        warnings.append(
            f"{len(empty_cells)} empty factorial cell(s) with < {min_per_cell} "
            "sample(s); interactions involving them are not estimable."
        )

    return DesignMatrixReport(
        factors=list(factors),
        interactions=[(a, b) for a, b in interactions],
        n_samples=len(sample_meta),
        n_columns=n_columns,
        rank=rank,
        full_rank=full_rank,
        confounded_pairs=confounded,
        empty_cells=empty_cells,
        warnings=warnings,
        columns=[str(c) for c in design_matrix.columns],
    )


def validate_design_matrix(
    sample_meta: pd.DataFrame,
    *,
    factors: list[str],
    interactions: list[tuple[str, str]] | None = None,
    min_per_cell: int = 1,
    references: dict[str, str] | None = None,
) -> DesignMatrixReport:
    """
    Validate that a multi-factor design is estimable, halting loudly if not.

    Args:
        sample_meta: Per-sample metadata.
        factors: Main-effect factor columns.
        interactions: Optional two-way interaction terms.
        min_per_cell: Minimum samples required per factorial-grid cell.
        references: Optional ``{factor: level}`` reference levels (see
            :func:`build_design_matrix`).

    Returns:
        The :class:`DesignMatrixReport` when the design has full column rank.

    Raises:
        CellQuorumConfigError: If a referenced column is absent, or the model
            matrix is rank-deficient (a non-estimable design).
    """

    report = analyze_design(
        sample_meta,
        factors=factors,
        interactions=interactions,
        min_per_cell=min_per_cell,
        references=references,
    )
    if not report.full_rank:
        detail = ""
        if report.confounded_pairs:
            detail += " Confounded factor(s): " + ", ".join(
                f"{a} ~ {b}" for a, b in report.confounded_pairs
            )
        if report.empty_cells:
            detail += f" Empty factorial cell(s): {report.empty_cells}."
        raise CellQuorumConfigError(
            "Design is not estimable: the model matrix is rank-deficient "
            f"({report.n_columns} columns, rank {report.rank})." + detail
        )
    return report


__all__ = [
    "Contrast",
    "ContrastsConfig",
    "DesignConfig",
    "FactorScope",
    "FactorSpec",
    "FactorialDesignConfig",
    "DesignMatrixReport",
    "DesignValidationResult",
    "DifferenceOfDifferencesContrast",
    "FactorialContrast",
    "SimpleEffectContrast",
    "analyze_design",
    "build_design_matrix",
    "validate_design_against_obs",
    "validate_design_matrix",
]
