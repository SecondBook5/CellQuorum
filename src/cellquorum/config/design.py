"""Experimental design + contrasts — the biological question, as config.

DesignConfig captures the donor/condition columns and the primary case/control
comparison; ContrastsConfig lists named comparisons (each with its own
case/control, pairing, and a min_donors power guard). Downstream DE/composition
stages consume these so the biological question is declared once, per dataset.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


class Contrast(StrictBaseModel):
    """One named case-vs-control comparison."""

    # Stable contrast name.
    name: str

    # Case / control condition tokens for this comparison.
    case: str
    control: str

    # Whether this comparison is donor-paired.
    paired: bool = False

    # Minimum distinct donors required for this comparison (power guard).
    min_donors: int = 0


class ContrastsConfig(StrictBaseModel):
    """A list of named comparisons."""

    # The registered contrasts.
    contrasts: list[Contrast] = []

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

    # Store the compared case condition.
    case: str

    # Store the compared control condition.
    control: str

    # Store whether the comparison is paired.
    paired: bool

    # Store distinct case donors.
    case_donors: set[str] = field(default_factory=set)

    # Store distinct control donors.
    control_donors: set[str] = field(default_factory=set)

    # Store donors with both case and control observations.
    complete_pair_donors: set[str] = field(default_factory=set)

    # Store non-fatal caveats.
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
    paired = contrast.paired if contrast is not None else design.paired
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

    def to_dict(self) -> dict[str, object]:
        """Convert the report to a JSON-friendly dictionary."""

        return {
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

    Returns:
        A DataFrame of model-matrix columns aligned to ``sample_meta.index``.

    Raises:
        CellQuorumConfigError: If any referenced column is absent from
            ``sample_meta``.
    """

    interactions = list(interactions or [])

    # Every referenced column (main effects + interaction members) must exist.
    referenced = list(
        dict.fromkeys([*factors, *(m for pair in interactions for m in pair)])
    )
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
    for column in referenced:
        coded = pd.get_dummies(
            sample_meta[column].astype(str), prefix=column, drop_first=True
        ).astype(float)
        coded.index = index
        dummies[column] = coded

    blocks: list[pd.DataFrame] = [
        pd.DataFrame({"Intercept": np.ones(n, dtype=float)}, index=index)
    ]

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

    Returns:
        A :class:`DesignMatrixReport` describing the design.

    Raises:
        CellQuorumConfigError: If a referenced column is absent (from
            :func:`build_design_matrix`).
    """

    interactions = list(interactions or [])
    design_matrix = build_design_matrix(sample_meta, factors, interactions)

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
        warnings.append(
            f"Design matrix is rank-deficient: {n_columns} columns but rank {rank}."
        )
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
        interactions=[tuple(pair) for pair in interactions],
        n_samples=len(sample_meta),
        n_columns=n_columns,
        rank=rank,
        full_rank=full_rank,
        confounded_pairs=confounded,
        empty_cells=empty_cells,
        warnings=warnings,
    )


def validate_design_matrix(
    sample_meta: pd.DataFrame,
    *,
    factors: list[str],
    interactions: list[tuple[str, str]] | None = None,
    min_per_cell: int = 1,
) -> DesignMatrixReport:
    """
    Validate that a multi-factor design is estimable, halting loudly if not.

    Args:
        sample_meta: Per-sample metadata.
        factors: Main-effect factor columns.
        interactions: Optional two-way interaction terms.
        min_per_cell: Minimum samples required per factorial-grid cell.

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
            f"({report.n_columns} columns, rank {report.rank})."
            + detail
        )
    return report


__all__ = [
    "Contrast",
    "ContrastsConfig",
    "DesignConfig",
    "DesignMatrixReport",
    "DesignValidationResult",
    "analyze_design",
    "build_design_matrix",
    "validate_design_against_obs",
    "validate_design_matrix",
]
