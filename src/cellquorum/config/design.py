"""Experimental design + contrasts — the biological question, as config.

DesignConfig captures the donor/condition columns and the primary case/control
comparison; ContrastsConfig lists named comparisons (each with its own
case/control, pairing, and a min_donors power guard). Downstream DE/composition
stages consume these so the biological question is declared once, per dataset.
"""

from __future__ import annotations

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


__all__ = ["Contrast", "ContrastsConfig", "DesignConfig"]
