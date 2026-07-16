"""Experimental design + contrasts — the biological question, as config.

DesignConfig captures the donor/condition columns and the primary case/control
comparison; ContrastsConfig lists named comparisons (each with its own
case/control, pairing, and a min_donors power guard). Downstream DE/composition
stages consume these so the biological question is declared once, per dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


__all__ = [
    "Contrast",
    "ContrastsConfig",
    "DesignConfig",
    "DesignValidationResult",
    "validate_design_against_obs",
]
