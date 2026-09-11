"""Donor identity and study-design invariants for two-group inference."""

import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stats.donor_comparison import two_group_test_on_donor_medians

# ---------------------------------------------------------------------------
# Donor-level two-group testing. These pin the unit of analysis: a figure
# p-value must never be computed over cells, because cells from one donor are
# not independent replicates.
# ---------------------------------------------------------------------------


def _cell_frame(
    *,
    n_donors_per_arm: int = 9,
    cells_per_donor: int = 300,
    shift: float = 0.4,
    paired: bool = True,
) -> pd.DataFrame:
    """Build a cell-level frame with a donor-level offset and per-cell noise.

    The arm-level effect is deliberately small relative to per-cell spread, so a
    cell-level test finds an overwhelming p-value while the donor-level test
    reports something honest.
    """

    rng = np.random.default_rng(0)
    rows = []
    for arm, offset in (("Normal", 0.0), ("LE", shift)):
        for donor_index in range(n_donors_per_arm):
            donor = f"d{donor_index}" if paired else f"{arm}_d{donor_index}"
            donor_effect = rng.normal(0.0, 0.05)
            values = rng.normal(5.0 + offset + donor_effect, 1.0, size=cells_per_donor)
            rows.append(pd.DataFrame({"metric": values, "condition": arm, "donor_id": donor}))
    return pd.concat(rows, ignore_index=True)


def test_donor_level_test_uses_donors_not_cells_as_n():
    frame = _cell_frame()
    result = two_group_test_on_donor_medians(
        frame,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=True,
    )
    assert result is not None
    # n is donors (9), not the 2700 cells per arm.
    assert result.n_group1 == 9
    assert result.n_group2 == 9
    assert "donor medians" in result.label
    assert "n = 9" in result.label


def test_donor_level_test_is_paired_when_donors_appear_in_both_arms():
    result = two_group_test_on_donor_medians(
        _cell_frame(paired=True),
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=True,
    )
    assert result is not None
    assert result.test == "wilcoxon_signed_rank"
    assert "signed-rank" in result.label


def test_donor_level_test_is_unpaired_when_donor_sets_are_disjoint():
    result = two_group_test_on_donor_medians(
        _cell_frame(paired=False),
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=False,
    )
    assert result is not None
    assert result.test == "mann_whitney"


def test_donor_level_p_value_is_not_the_pseudoreplicated_cell_level_one():
    """The donor-level p-value must be orders of magnitude less extreme."""

    from scipy import stats

    frame = _cell_frame()
    cell_level = stats.mannwhitneyu(
        frame.loc[frame["condition"].eq("Normal"), "metric"],
        frame.loc[frame["condition"].eq("LE"), "metric"],
        alternative="two-sided",
    ).pvalue
    donor_level = two_group_test_on_donor_medians(
        frame,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=True,
    )
    assert donor_level is not None
    # Cell-level pseudoreplication buys many orders of magnitude of fake
    # confidence; the donor-level test cannot exceed 1/2**9 for n=9 paired.
    assert cell_level < 1e-20
    assert donor_level.p_value > cell_level * 1e10


def test_donor_level_test_returns_none_when_underpowered():
    frame = _cell_frame(n_donors_per_arm=2)
    assert (
        two_group_test_on_donor_medians(
            frame,
            value_col="metric",
            group_col="condition",
            donor_col="donor_id",
            group1="Normal",
            group2="LE",
            paired=True,
        )
        is None
    )


def test_missing_required_column_raises():
    with pytest.raises(CellQuorumDataError, match="requires columns"):
        two_group_test_on_donor_medians(
            _cell_frame().drop(columns=["donor_id"]),
            value_col="metric",
            group_col="condition",
            donor_col="donor_id",
            group1="Normal",
            group2="LE",
            paired=True,
        )


def test_donor_level_test_ignores_non_finite_cells():
    frame = _cell_frame()
    poisoned = frame.copy()
    poisoned.loc[poisoned.index[:50], "metric"] = np.inf
    poisoned.loc[poisoned.index[50:100], "metric"] = np.nan
    clean = two_group_test_on_donor_medians(
        frame,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=True,
    )
    dirty = two_group_test_on_donor_medians(
        poisoned,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=True,
    )
    assert clean is not None and dirty is not None
    # Non-finite values must not become +inf medians or drop a whole donor.
    assert dirty.n_group1 == clean.n_group1
    assert np.isfinite(dirty.p_value)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", "   "])
def test_missing_donors_cannot_satisfy_minimum_replication(missing):
    frame = pd.DataFrame(
        {
            "condition": ["control"] * 3 + ["case"] * 3,
            "donor": ["A", "B", missing, "A", "B", missing],
            "value": [1, 2, 3, 2, 3, 4],
        }
    )
    result = two_group_test_on_donor_medians(
        frame,
        value_col="value",
        group_col="condition",
        donor_col="donor",
        group1="control",
        group2="case",
        paired=True,
    )
    assert result is None


def test_partial_pairing_requires_explicit_complete_pair_policy():
    frame = _cell_frame(n_donors_per_arm=4)
    frame = frame[~((frame.condition == "LE") & (frame.donor_id == "d3"))]
    args = dict(
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
        paired=True,
    )
    with pytest.raises(CellQuorumDataError, match="unmatched"):
        two_group_test_on_donor_medians(frame, **args)
    result = two_group_test_on_donor_medians(frame, **args, incomplete_pairs="drop")
    assert result.n_group1 == result.n_group2 == 3
    assert result.n_unmatched_donors == 1
    assert result.donors_group1 == result.donors_group2 == ("d0", "d1", "d2")
    with pytest.raises(CellQuorumDataError, match="overlapping"):
        two_group_test_on_donor_medians(frame, **{**args, "paired": False})


def test_zero_differences_and_contrast_direction():
    frame = pd.DataFrame(
        {
            "condition": ["control"] * 3 + ["case"] * 3,
            "donor": ["A", "B", "C"] * 2,
            "value": [1, 2, 3] * 2,
        }
    )
    args = dict(value_col="value", group_col="condition", donor_col="donor", paired=True)
    result = two_group_test_on_donor_medians(frame, group1="control", group2="case", **args)
    assert result.p_value == 1 and result.effect == 0
    frame.loc[frame.condition == "case", "value"] += 1
    forward = two_group_test_on_donor_medians(frame, group1="control", group2="case", **args)
    reverse = two_group_test_on_donor_medians(frame, group1="case", group2="control", **args)
    assert forward.effect == -reverse.effect == 1
    assert forward.p_value == reverse.p_value
    assert forward.to_dict()["n_group1"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"p_value": float("nan")},
        {"p_value": 1.1},
        {"effect": float("inf")},
        {"n_group1": 4},
        {"donors_group1": ("A", "A", "C")},
        {"donors_group2": ("C", "B", "A")},
        {"donors_group1": ["A", "B", "C"]},
        {"test": "unknown"},
        {"test": "mann_whitney"},
        {"group2": "control"},
        {"n_excluded_rows": -1},
    ],
)
def test_result_rejects_inconsistent_scientific_metadata(change):
    from cellquorum.stats.donor_comparison import TwoGroupTest

    fields = dict(
        p_value=0.25,
        test="wilcoxon_signed_rank",
        n_group1=3,
        n_group2=3,
        group1="control",
        group2="case",
        donors_group1=("A", "B", "C"),
        donors_group2=("A", "B", "C"),
        effect=1.0,
        n_excluded_rows=0,
        n_unmatched_donors=0,
    )
    fields.update(change)
    with pytest.raises(CellQuorumDataError):
        TwoGroupTest(**fields)
