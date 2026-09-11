"""Two-group inference over biological replicates with explicit study design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from cellquorum.core.exceptions import CellQuorumDataError


@dataclass(frozen=True)
class TwoGroupTest:
    """A donor-level test with its contrast and analyzed population."""

    p_value: float
    test: str
    n_group1: int
    n_group2: int
    group1: str
    group2: str
    donors_group1: tuple[str, ...]
    donors_group2: tuple[str, ...]
    effect: float
    n_excluded_rows: int
    n_unmatched_donors: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.p_value) or not 0 <= self.p_value <= 1:
            raise CellQuorumDataError("Comparison p-value must be a finite probability.")
        if not np.isfinite(self.effect):
            raise CellQuorumDataError("Comparison effect must be finite.")
        if self.test not in {"wilcoxon_signed_rank", "mann_whitney"}:
            raise CellQuorumDataError(f"Unsupported donor test: {self.test}")
        if self.group1 == self.group2 or any(
            not isinstance(group, str) or not group.strip() for group in (self.group1, self.group2)
        ):
            raise CellQuorumDataError("Comparison groups must be distinct, nonblank labels.")
        for count, donors in (
            (self.n_group1, self.donors_group1),
            (self.n_group2, self.donors_group2),
        ):
            if type(count) is not int or count < 2:
                raise CellQuorumDataError("Each comparison group requires at least two donors.")
            if not isinstance(donors, tuple) or any(
                not isinstance(donor, str) or not donor.strip() for donor in donors
            ):
                raise CellQuorumDataError("Donor identities must be a tuple of nonblank labels.")
            if len(donors) != count or len(set(donors)) != count:
                raise CellQuorumDataError("Donor counts must match unique donor identities.")
        if self.test == "wilcoxon_signed_rank":
            if self.donors_group1 != self.donors_group2:
                raise CellQuorumDataError("Paired results require identically ordered donors.")
        elif set(self.donors_group1) & set(self.donors_group2):
            raise CellQuorumDataError("Independent results cannot share donors.")
        for count in (self.n_excluded_rows, self.n_unmatched_donors):
            if type(count) is not int or count < 0:
                raise CellQuorumDataError("Exclusion counts must be nonnegative integers.")
        if self.test == "mann_whitney" and self.n_unmatched_donors:
            raise CellQuorumDataError("Unmatched donor counts apply only to paired results.")

    @property
    def label(self) -> str:
        """Describe the test and actual biological replicate counts."""
        if self.test == "wilcoxon_signed_rank":
            return (
                f"Wilcoxon signed-rank p = {self.p_value:.2g}\n"
                f"donor medians, n = {self.n_group1} paired"
            )
        return (
            f"Mann–Whitney p = {self.p_value:.2g}\n"
            f"donor medians, n = {self.n_group1} vs {self.n_group2}"
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the result independently of any figure."""
        return asdict(self)


def two_group_test_on_donor_medians(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    donor_col: str,
    group1: str,
    group2: str,
    paired: bool,
    min_donors: int = 3,
    incomplete_pairs: Literal["error", "drop"] = "error",
) -> TwoGroupTest | None:
    """Test donor medians; return None when usable donor counts are insufficient.

    Missing labels and nonfinite observations are excluded before grouping. Paired
    designs reject unmatched donors unless complete-pair analysis is explicitly
    requested. Independent designs reject overlapping donors. The effect is group2
    minus group1: median paired difference or difference of independent medians.
    """
    if type(paired) is not bool:
        raise ValueError("paired must explicitly be True or False.")
    if type(min_donors) is not int or min_donors < 2:
        raise ValueError("min_donors must be an integer of at least two.")
    if incomplete_pairs not in ("error", "drop"):
        raise ValueError("incomplete_pairs must be 'error' or 'drop'.")
    if group1 == group2 or not group1.strip() or not group2.strip():
        raise ValueError("The comparison requires two distinct, nonblank group labels.")
    required = [value_col, group_col, donor_col]
    if len(set(required)) != 3:
        raise ValueError("Value, condition, and donor columns must be distinct.")
    missing = set(required) - set(frame.columns)
    if missing:
        raise CellQuorumDataError(f"Donor comparison requires columns: {sorted(missing)}")
    table = frame.loc[:, required].copy()
    table[value_col] = pd.to_numeric(table[value_col], errors="coerce")
    valid = np.isfinite(table[value_col]) & table[group_col].notna() & table[donor_col].notna()
    for key in (group_col, donor_col):
        valid &= table[key].astype("string").str.strip().ne("").fillna(False)
    n_excluded = int((~valid).sum())
    table = table.loc[valid]
    for key in (group_col, donor_col):
        if table[key].nunique() != table[key].astype(str).nunique():
            raise CellQuorumDataError(f"Ambiguous mixed-type identifiers in {key}.")
        table[key] = table[key].astype(str)
    table = table[table[group_col].isin([group1, group2])]
    medians = table.groupby([group_col, donor_col], observed=True)[value_col].median()
    if group1 not in medians.index.get_level_values(
        0
    ) or group2 not in medians.index.get_level_values(0):
        return None
    first, second = medians.loc[group1], medians.loc[group2]
    shared = first.index.intersection(second.index).sort_values()
    unmatched = len(first.index.symmetric_difference(second.index)) if paired else 0
    if paired and unmatched and incomplete_pairs == "error":
        raise CellQuorumDataError(
            "Paired comparison has unmatched donors; "
            "explicitly select complete pairs to exclude them."
        )
    if not paired and len(shared):
        raise CellQuorumDataError("Independent groups contain overlapping donor identifiers.")
    if paired:
        first, second = first.loc[shared], second.loc[shared]
    if min(len(first), len(second)) < min_donors:
        return None
    if paired:
        differences = second.to_numpy() - first.to_numpy()
        pvalue = 1.0 if np.all(differences == 0) else float(stats.wilcoxon(differences).pvalue)
        method = "wilcoxon_signed_rank"
        effect = float(np.median(differences))
    else:
        pvalue = float(stats.mannwhitneyu(first, second, alternative="two-sided").pvalue)
        method = "mann_whitney"
        effect = float(second.median() - first.median())
    if not np.isfinite(pvalue) or not 0 <= pvalue <= 1:
        raise CellQuorumDataError("Donor comparison did not produce a finite probability.")
    return TwoGroupTest(
        p_value=pvalue,
        test=method,
        n_group1=len(first),
        n_group2=len(second),
        group1=group1,
        group2=group2,
        donors_group1=tuple(first.index),
        donors_group2=tuple(second.index),
        effect=effect,
        n_excluded_rows=n_excluded,
        n_unmatched_donors=unmatched,
    )
