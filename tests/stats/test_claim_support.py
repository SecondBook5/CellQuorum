"""Tests for the claim-support primitives: FDR reachability and group resolution.

These exist because a real analysis in this project reported "nothing clears FDR" from a
family whose floor put a lone significant result out of reach, and ranked a fold-change
computed from a median of 1.5 cells per sample at the top of a figure. Both failures are
properties of the design, so both are testable without any data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.claim_support import (
    FDR_REACHABILITY_COLUMNS,
    GROUP_RESOLUTION_COLUMNS,
    annotate_fdr_reachability,
    group_resolution,
)


def _paired(n_donors: int) -> tuple[list[str], list[bool]]:
    donors = [f"D{i}" for i in range(n_donors)]
    return [*donors, *donors], [False] * n_donors + [True] * n_donors


# --------------------------------------------------------------------------- #
# FDR reachability
# --------------------------------------------------------------------------- #


def test_the_nine_donor_thirteen_test_family_that_produced_the_wrong_conclusion():
    """The exact arrangement that got written into a manuscript note as a null.

    Nine paired donors floor an assumption-free two-sided p at 2/2**9 = 0.0039. BH over
    thirteen tests puts a lone result's best FDR at 0.0508 -- above 0.05 -- so no single
    signed-rank result could have been called however large the shift. Two together could:
    0.0039 * 13 / 2 = 0.0254.
    """
    donors, is_case = _paired(9)
    table = pd.DataFrame({"pvalue": np.linspace(0.001, 0.9, 13)})

    out = annotate_fdr_reachability(table, donors=donors, is_case=is_case)

    assert out["design_floor_p"].iloc[0] == pytest.approx(2 / 2**9)
    assert out["family_size"].iloc[0] == 13
    # A lone result is unreachable, a pair is: that is what min_concordant == 2 means.
    assert out["family_min_concordant"].iloc[0] == 2
    floor = out["design_floor_p"].iloc[0]
    assert floor * 13 / 1 > 0.05
    assert floor * 13 / 2 < 0.05
    # Two is available inside a thirteen-test family, so the family is not hopeless.
    assert bool(out["family_floor_reachable"].iloc[0]) is True


def test_a_small_family_can_be_unreachable_outright():
    """Four donor pairs in a family of twenty cannot call anything, alone or together."""
    donors, is_case = _paired(4)
    table = pd.DataFrame({"pvalue": np.full(20, 0.01)})

    out = annotate_fdr_reachability(table, donors=donors, is_case=is_case)

    assert out["design_floor_p"].iloc[0] == pytest.approx(2 / 2**4)  # 0.125
    # 0.125 * 20 / k <= 0.05 needs k >= 50, and there are only 20 tests.
    assert out["family_min_concordant"].iloc[0] == 50
    assert bool(out["family_floor_reachable"].iloc[0]) is False


def test_the_floor_is_a_scale_and_not_a_veto():
    """A parametric p below the floor is flagged, not rewritten.

    This is the property that keeps the column honest: the whole point of a
    distributional assumption is that it can go below the randomization floor, so the flag
    reports the fact and leaves every p-value and FDR exactly as the test produced them.
    """
    donors, is_case = _paired(9)
    table = pd.DataFrame({"pvalue": [1e-9, 0.5], "fdr": [2e-9, 0.5]})

    out = annotate_fdr_reachability(table, donors=donors, is_case=is_case)

    assert list(out["p_below_design_floor"]) == [True, False]
    assert list(out["pvalue"]) == [1e-9, 0.5]
    assert list(out["fdr"]) == [2e-9, 0.5]


def test_unfittable_rows_are_not_counted_in_the_family():
    """A NaN p-value was held out of the correction, so it is held out of the family too.

    Counting it would inflate ``family_size``, which inflates ``family_min_concordant``,
    which overstates how many rows must move together -- the failure mode this column
    exists to prevent, reintroduced one step down.
    """
    donors, is_case = _paired(9)
    table = pd.DataFrame({"pvalue": [0.001, 0.02, np.nan, np.nan]})

    out = annotate_fdr_reachability(table, donors=donors, is_case=is_case)

    assert out["family_size"].iloc[0] == 2


def test_an_unpaired_design_has_a_far_lower_floor():
    """With no donor spanning both arms the randomization set is C(n, k), not 2**pairs."""
    donors = [f"S{i}" for i in range(18)]
    is_case = [True] * 9 + [False] * 9
    table = pd.DataFrame({"pvalue": np.full(13, 0.01)})

    out = annotate_fdr_reachability(table, donors=donors, is_case=is_case)

    # C(18, 9) = 48620 assignments, versus 512 for nine pairs.
    assert out["design_floor_p"].iloc[0] == pytest.approx(2 / 48620)
    assert out["family_min_concordant"].iloc[0] == 1
    assert bool(out["family_floor_reachable"].iloc[0]) is True


def test_an_empty_table_still_carries_the_schema():
    donors, is_case = _paired(9)
    out = annotate_fdr_reachability(
        pd.DataFrame(columns=["pvalue"]), donors=donors, is_case=is_case
    )
    assert list(out.columns) == ["pvalue", *FDR_REACHABILITY_COLUMNS]
    assert out.empty


def test_a_missing_p_value_column_is_an_error_not_a_silent_skip():
    donors, is_case = _paired(9)
    with pytest.raises(KeyError, match="no p-value column"):
        annotate_fdr_reachability(pd.DataFrame({"fdr": [0.1]}), donors=donors, is_case=is_case)


# --------------------------------------------------------------------------- #
# Group resolution
# --------------------------------------------------------------------------- #


def _counts() -> tuple[pd.DataFrame, pd.Series]:
    """Six samples: an abundant group, a group at the floor, and a barely-present one."""
    counts = pd.DataFrame(
        {
            "Abundant": [500, 480, 520, 510, 495, 505],
            "AtFloor": [12, 11, 10, 13, 9, 14],
            "Sparse": [2, 0, 1, 3, 0, 2],
        },
        index=[f"S{i}" for i in range(6)],
    )
    conditions = pd.Series(["control"] * 3 + ["case"] * 3, index=counts.index)
    return counts, conditions


def test_a_group_below_the_per_sample_floor_is_flagged_not_dropped():
    """The sparse group keeps its row -- hiding it to tidy a ranking is the worse failure."""
    counts, conditions = _counts()

    out = group_resolution(counts, conditions, case="case", control="control")

    assert set(out["cell_type"]) == {"Abundant", "AtFloor", "Sparse"}
    flags = out.set_index("cell_type")["ratio_rankable"]
    assert bool(flags["Abundant"]) is True
    assert bool(flags["AtFloor"]) is True  # median 11.5 >= 10
    assert bool(flags["Sparse"]) is False
    assert "below the 10-cell floor" in out.set_index("cell_type").loc["Sparse", "resolution_note"]


def test_the_floor_is_the_median_per_sample_not_the_pooled_total():
    """Eight cells pooled over six samples looks like data; per sample it is 1.5."""
    counts, conditions = _counts()
    out = group_resolution(counts, conditions, case="case", control="control").set_index(
        "cell_type"
    )
    assert out.loc["Sparse", "total_cells"] == 8
    assert out.loc["Sparse", "median_cells_per_sample"] == pytest.approx(1.5)


def test_samples_with_none_of_a_group_are_reported_separately_from_the_minimum():
    """A zero share makes a fold-change infinite, which is a different failure from noise."""
    counts, conditions = _counts()
    out = group_resolution(counts, conditions, case="case", control="control").set_index(
        "cell_type"
    )
    assert out.loc["Sparse", "n_samples_zero"] == 2
    assert out.loc["Sparse", "min_cells_per_sample"] == 0
    assert "contain none of this group" in out.loc["Sparse", "resolution_note"]
    assert out.loc["Abundant", "n_samples_zero"] == 0
    assert out.loc["Abundant", "resolution_note"] == ""


def test_arm_medians_are_reported_and_optional():
    counts, conditions = _counts()

    with_arms = group_resolution(counts, conditions, case="case", control="control").set_index(
        "cell_type"
    )
    assert with_arms.loc["Abundant", "median_cells_control"] == pytest.approx(500)
    assert with_arms.loc["Abundant", "median_cells_case"] == pytest.approx(505)

    without = group_resolution(counts).set_index("cell_type")
    assert np.isnan(without.loc["Abundant", "median_cells_case"])
    assert without.loc["Abundant", "median_cells_per_sample"] == pytest.approx(502.5)


def test_the_floor_is_a_stated_argument_not_a_hidden_constant():
    counts, conditions = _counts()
    strict = group_resolution(
        counts, conditions, case="case", control="control", min_cells_per_sample=100
    ).set_index("cell_type")
    assert bool(strict.loc["AtFloor", "ratio_rankable"]) is False
    assert bool(strict.loc["Abundant", "ratio_rankable"]) is True


def test_rows_are_ordered_by_resolution_and_carry_the_schema():
    counts, conditions = _counts()
    out = group_resolution(counts, conditions, case="case", control="control")
    assert list(out.columns) == ["cell_type", *GROUP_RESOLUTION_COLUMNS]
    assert list(out["cell_type"]) == ["Abundant", "AtFloor", "Sparse"]


def test_an_empty_count_matrix_returns_the_schema():
    out = group_resolution(pd.DataFrame())
    assert list(out.columns) == ["cell_type", *GROUP_RESOLUTION_COLUMNS]
    assert out.empty
