"""Tests for the donor-level concordance audit of abundance claims.

Each cohort is built so that the correct verdict is known by construction: a
per-donor change is specified directly, one cell type absorbs it, and the pattern
that should come out is a consequence of the numbers rather than a judgement about
real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cellquorum.stats.paired_concordance import (
    CONCORDANCE_COLUMNS,
    VALUE_CONCORDANCE_COLUMNS,
    donor_unanimous,
    mark_called,
    paired_abundance_concordance,
    paired_value_concordance,
    qualify_abundance_calls,
)

# Baseline counts per cell type per sample. Equal shares keep the arithmetic
# transparent: the control CLR is zero for every type, so a donor's CLR delta is
# just the case sample's CLR.
BASE = 1000


def build_pairs(
    deltas: dict[str, int],
    *,
    target: str = "A",
    absorber: str = "B",
    filler: tuple[str, ...] = ("C",),
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build a donor-paired count matrix in which one cell type moves by a set amount.

    Args:
        deltas: Maps donor to the case-arm count change in ``target``. The change
            is taken out of ``absorber`` so every sample keeps the same total, which
            makes the log-ratio delta depend only on the specified change.
        target: Cell type that moves.
        absorber: Cell type that compensates.
        filler: Additional cell types held constant, present so the composition has
            somewhere to sit.

    Returns:
        ``(counts, donors, conditions)`` ready for
        :func:`paired_abundance_concordance`.
    """

    cell_types = [target, absorber, *filler]
    rows: dict[str, dict[str, int]] = {}
    donor_of: dict[str, str] = {}
    condition_of: dict[str, str] = {}

    for donor, delta in deltas.items():
        for arm, shift in (("Control", 0), ("Case", delta)):
            sample = f"{donor}_{arm}"
            rows[sample] = dict.fromkeys(cell_types, BASE)
            rows[sample][target] = BASE + shift
            rows[sample][absorber] = BASE - shift
            donor_of[sample] = donor
            condition_of[sample] = arm

    counts = pd.DataFrame(rows).T[cell_types]
    return counts, pd.Series(donor_of), pd.Series(condition_of)


def audit(deltas: dict[str, int], **kwargs) -> pd.Series:
    """Run the audit on a fixture and return the row for the moving cell type."""

    counts, donors, conditions = build_pairs(deltas)
    table = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control", **kwargs
    )
    return table.set_index("cell_type").loc["A"]


def test_a_unanimous_shift_is_consistent():
    """When every donor moves the same way the pattern is consistent."""

    row = audit({f"D{i}": 60 + 5 * i for i in range(9)})

    assert row["pattern"] == "consistent"
    assert row["n_pairs"] == 9
    assert row["n_agree"] == 9
    assert row["direction"] == 1
    assert row["loo_sign_stable"]
    # Unanimity on 9 pairs is the smallest attainable one-sided p, 2**-9.
    assert row["sign_test_p"] == pytest.approx(2.0**-9)
    assert "9 of 9 donors" in row["reason"]


def test_a_minority_direction_carried_by_large_movers_is_not_consistent():
    """A mean built from a few big donors is reported as a subgroup, not a change.

    This is the motivating case: on a real cohort the one cell type a compositional
    model called credible moved in the reported direction in a minority of donors,
    the gains being large and the losses small. The mean was real; the cohort-wide
    claim was not.
    """

    deltas = {"D0": 700, "D1": 650, "D2": 600, "D3": 550}
    deltas.update({f"D{i}": -60 for i in range(4, 9)})

    row = audit(deltas)

    assert row["pattern"] in {"heterogeneous", "direction_inconsistent"}
    assert row["pattern"] != "consistent"
    assert row["direction"] == 1
    assert row["n_agree"] == 4
    assert row["sign_test_p"] > 0.5
    assert "of 9 donors" in row["reason"]


def test_donors_moving_against_the_mean_never_read_as_consistent():
    """Pins the one-sided sign test.

    Eight of nine donors move one way and a single enormous donor drags the cohort
    mean the other way, so the mean's direction is shared by exactly one donor. A
    two-sided sign test calls that significant -- ``binomtest(1, 9, 0.5)`` is
    p=0.039 -- and would have labelled this cell type consistent while the donors
    contradicted it. The test used here is one-sided on the reported direction, so
    it cannot make that mistake.
    """

    deltas = {f"D{i}": 5 for i in range(8)}
    deltas["D8"] = -800

    row = audit(deltas)

    # The precondition: the mean points the opposite way to eight of nine donors.
    assert row["direction"] == -1
    assert row["n_agree"] == 1

    # What a two-sided test would have said about that same count.
    two_sided = stats.binomtest(1, 9, 0.5, alternative="two-sided").pvalue
    assert two_sided < 0.05, "fixture no longer exercises the two-sided failure mode"

    assert row["sign_test_p"] > 0.5, "the sign test must not reward disagreement"
    assert row["pattern"] != "consistent"
    assert row["pattern"] == "direction_inconsistent"


def test_one_dissenting_donor_setting_the_effect_size_is_flagged():
    """Donors agree on direction, but one is big enough to cancel them.

    Distinct from the case above: here the sign test is satisfied, so the direction
    is the group's. It is the magnitude that belongs to one donor, which the
    leave-one-out range catches and the sign test cannot.
    """

    deltas = {f"D{i}": 100 for i in range(8)}
    # Tuned so the cohort mean log-ratio stays positive while dropping any single
    # agreeing donor takes it negative.
    deltas["D8"] = -580

    row = audit(deltas)

    assert row["n_agree"] == 8, "the direction should still be the majority's"
    assert row["direction"] == 1
    assert row["sign_test_p"] < 0.05, "8 of 9 clears the one-sided bar"
    assert not row["loo_sign_stable"]
    assert row["pattern"] == "single_donor_driven"
    assert "leave-one-out" in row["reason"]


def test_too_few_pairs_is_underpowered_not_negative():
    """Below the paired floor no pattern is claimed either way."""

    row = audit({f"D{i}": 200 for i in range(4)})

    assert row["pattern"] == "underpowered"
    assert row["n_pairs"] == 4
    assert "4 donor(s)" in row["reason"]

    # The same cohort passes once the floor is lowered to what it can support.
    relaxed = audit({f"D{i}": 200 for i in range(4)}, min_pairs=4)
    assert relaxed["pattern"] != "underpowered"


def test_unpaired_donors_are_dropped_and_replicates_do_not_double_count():
    """Only donors contributing both arms count, and each counts once."""

    counts, donors, conditions = build_pairs({f"D{i}": 100 for i in range(7)})

    # A donor with a case sample but no control cannot be paired.
    counts.loc["D7_Case"] = {"A": 5000, "B": 1000, "C": 1000}
    donors["D7_Case"] = "D7"
    conditions["D7_Case"] = "Case"

    # A second case sample for D0, moving the other way. Averaged within the arm,
    # so D0 still contributes one paired observation rather than two.
    counts.loc["D0_Case_rep"] = {"A": 900, "B": 1100, "C": 1000}
    donors["D0_Case_rep"] = "D0"
    conditions["D0_Case_rep"] = "Case"

    table = paired_abundance_concordance(counts, donors, conditions, case="Case", control="Control")
    row = table.set_index("cell_type").loc["A"]

    assert row["n_pairs"] == 7, "the unpaired donor must not add a pair"
    # D0's two case samples average to no change, so it stops agreeing.
    assert row["n_agree"] == 6


def test_degenerate_inputs_return_the_schema_not_an_exception():
    """Empty and all-zero matrices produce an empty table with the full schema."""

    empty = paired_abundance_concordance(
        pd.DataFrame(), pd.Series(dtype=str), pd.Series(dtype=str), case="Case", control="Control"
    )
    assert empty.empty
    assert list(empty.columns) == list(CONCORDANCE_COLUMNS)

    zeros = pd.DataFrame({"A": [0, 0], "B": [0, 0]}, index=["s1", "s2"])
    result = paired_abundance_concordance(
        zeros,
        pd.Series({"s1": "D0", "s2": "D0"}),
        pd.Series({"s1": "Case", "s2": "Control"}),
        case="Case",
        control="Control",
    )
    assert result.empty


def test_every_cell_type_appears_once_with_a_pattern():
    """The table covers the whole composition, not only the moving type."""

    counts, donors, conditions = build_pairs({f"D{i}": 100 for i in range(8)})
    table = paired_abundance_concordance(counts, donors, conditions, case="Case", control="Control")

    assert sorted(table["cell_type"]) == ["A", "B", "C"]
    assert table["pattern"].notna().all()
    assert set(CONCORDANCE_COLUMNS) == set(table.columns)
    # A absorbs the gain and B the loss, so they must be called opposite ways.
    signed = table.set_index("cell_type")["direction"]
    assert signed["A"] == -signed["B"]


def test_qualify_flags_called_effects_the_donors_do_not_support():
    """Notes are emitted for positive calls whose pattern is not consistent."""

    counts, donors, conditions = build_pairs(
        {"D0": 700, "D1": 650, "D2": 600, "D3": 550, **{f"D{i}": -60 for i in range(4, 9)}}
    )
    concordance = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control"
    )

    effects = pd.DataFrame(
        {
            "cell_type": ["A", "B", "C"],
            "credible_effect": [True, False, False],
            "inclusion_probability": [0.91, 0.2, 0.1],
        }
    )

    annotated, notes = qualify_abundance_calls(effects, concordance)

    assert len(annotated) == 3, "annotation must not add or drop rows"
    assert "pattern" in annotated.columns
    assert len(notes) == 1
    assert notes[0].startswith("A: called abundance change is")
    assert "of 9 donors" in notes[0]


def test_qualify_is_silent_when_the_call_is_donor_consistent():
    """A unanimous effect that was called needs no qualifier."""

    counts, donors, conditions = build_pairs({f"D{i}": 60 + 5 * i for i in range(9)})
    concordance = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control"
    )
    effects = pd.DataFrame({"cell_type": ["A"], "credible_effect": [True]})

    annotated, notes = qualify_abundance_calls(effects, concordance)

    assert annotated.loc[0, "pattern"] == "consistent"
    assert notes == []


def test_qualify_passes_through_when_there_is_nothing_to_join():
    """Missing or empty inputs leave the effects table alone."""

    effects = pd.DataFrame({"cell_type": ["A"], "credible_effect": [True]})

    unchanged, notes = qualify_abundance_calls(effects, pd.DataFrame())
    assert notes == []
    assert unchanged.equals(effects)

    empty, notes = qualify_abundance_calls(pd.DataFrame(), pd.DataFrame())
    assert empty.empty
    assert notes == []


def test_percentage_points_and_log_ratio_are_both_reported():
    """Both scales are carried, since they can disagree and readers need both."""

    counts, donors, conditions = build_pairs({f"D{i}": 100 for i in range(8)})
    table = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control"
    ).set_index("cell_type")

    row = table.loc["A"]
    # 100 more cells out of 3000 is a 3.33 point gain on a 33.3% baseline.
    assert row["mean_delta_pp"] == pytest.approx(100 / 3000 * 100, rel=1e-6)
    assert row["mean_delta_clr"] > 0
    assert np.isfinite(row["median_delta_clr"])


def test_a_numeric_state_label_is_named_the_same_way_on_both_sides_of_the_join():
    """A label that looks like a number must not become two names for one state.

    A subcluster column is float64 because NaN forces it, so the state a model calls
    ``1.0`` is the state the count matrix calls ``1``. Pandas does not join those: it
    raises "You are trying to merge on float64 and object columns" and takes the whole
    abundance stage down with it, after the methods have already written their CSVs.
    """

    counts, donors, conditions = build_pairs(
        {f"D{i}": 60 + 5 * i for i in range(8)}, target=1.0, absorber=2.0, filler=(3.0,)
    )
    concordance = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control"
    )

    # The audit names an integral state without a decimal point.
    assert set(concordance["cell_type"]) == {"1", "2", "3"}

    # The effects table arrives from a method that read the float column directly.
    effects = pd.DataFrame({"cell_type": [1.0, 2.0, 3.0], "credible_effect": [True, False, False]})

    annotated, notes = qualify_abundance_calls(effects, concordance)

    # Every row found its partner, so the pattern column is populated throughout.
    assert len(annotated) == 3
    assert annotated["pattern"].notna().all()
    assert annotated.loc[annotated["cell_type"] == "1", "pattern"].iloc[0] == "consistent"
    assert notes == []


# --------------------------------------------------------------------------- #
# the general paired-value audit
# --------------------------------------------------------------------------- #


def build_values(
    per_item: dict[str, dict[str, tuple[float | None, float | None]]],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build samples x items from ``item -> donor -> (case, control)``.

    ``None`` on a side means the item was not measured for that arm, which is the
    state a composition cannot be in and the reason this audit exists.
    """

    donors = sorted({donor for arms in per_item.values() for donor in arms})
    index = [f"{donor}_{arm}" for donor in donors for arm in ("Case", "Control")]
    frame = pd.DataFrame(np.nan, index=index, columns=list(per_item))
    for item, arms in per_item.items():
        for donor, (case_value, control_value) in arms.items():
            if case_value is not None:
                frame.loc[f"{donor}_Case", item] = float(case_value)
            if control_value is not None:
                frame.loc[f"{donor}_Control", item] = float(control_value)
    donor_series = pd.Series([name.rsplit("_", 1)[0] for name in index], index=index)
    condition_series = pd.Series([name.rsplit("_", 1)[1] for name in index], index=index)
    return frame, donor_series, condition_series


def test_a_unanimous_rise_is_consistent_and_reports_both_levels():
    values, donors, conditions = build_values(
        {"edge": {f"D{i}": (1.0 + 0.1 * i, 0.5) for i in range(8)}}
    )
    out = paired_value_concordance(
        values, donors, conditions, case="Case", control="Control", item_label="partner"
    )
    row = out.iloc[0]
    assert list(out.columns)[0] == "partner"
    assert (row["n_pairs"], row["n_agree"], row["pattern"]) == (8, 8, "consistent")
    assert row["mean_control"] == pytest.approx(0.5)
    assert row["mean_delta"] == pytest.approx(row["mean_case"] - row["mean_control"])


def test_a_donor_present_in_one_arm_only_is_dropped_for_that_item_and_counted():
    """The failure this rule prevents: an 8-donor case arm against a 5-donor control one."""
    arms = {f"D{i}": (2.0, 1.0) for i in range(5)}
    arms.update({f"D{i}": (9.0, None) for i in range(5, 8)})
    values, donors, conditions = build_values({"edge": arms})
    out = paired_value_concordance(values, donors, conditions, case="Case", control="Control")
    row = out.iloc[0]
    assert (row["n_pairs"], row["n_donors_one_arm"]) == (5, 3)
    # The three case-only donors sit at 9.0 and would quadruple the mean if averaged in.
    assert row["mean_case"] == pytest.approx(2.0)
    assert row["mean_delta"] == pytest.approx(1.0)


def test_the_pair_count_is_per_item_not_per_table():
    """A composition is complete; a measurement is not, which is the whole difference."""
    values, donors, conditions = build_values(
        {
            "everywhere": {f"D{i}": (2.0, 1.0) for i in range(8)},
            "patchy": {f"D{i}": (2.0, 1.0) if i < 6 else (None, None) for i in range(8)},
        }
    )
    out = paired_value_concordance(values, donors, conditions, case="Case", control="Control")
    pairs = dict(zip(out["item"], out["n_pairs"], strict=True))
    assert pairs == {"everywhere": 8, "patchy": 6}


def test_an_item_no_donor_could_pair_is_kept_as_underpowered_not_dropped():
    """A dropped row makes a detection failure look like a question nobody asked."""
    values, donors, conditions = build_values(
        {
            "paired": {f"D{i}": (2.0, 1.0) for i in range(8)},
            "case_only": {f"D{i}": (2.0, None) for i in range(8)},
        }
    )
    out = paired_value_concordance(values, donors, conditions, case="Case", control="Control")
    row = out[out["item"] == "case_only"].iloc[0]
    assert (row["n_pairs"], row["n_donors_one_arm"], row["pattern"]) == (0, 8, "underpowered")
    assert np.isnan(row["mean_delta"])
    assert "both arms" in row["reason"]


def test_the_design_floor_follows_the_items_own_missingness():
    """Six pairs cannot reach what eight can, and the column has to say so per item."""
    values, donors, conditions = build_values(
        {
            "eight": {f"D{i}": (2.0, 1.0) for i in range(8)},
            "six": {f"D{i}": (2.0, 1.0) if i < 6 else (None, None) for i in range(8)},
        }
    )
    out = paired_value_concordance(values, donors, conditions, case="Case", control="Control")
    floors = dict(zip(out["item"], out["design_floor_p"], strict=True))
    assert floors["eight"] == pytest.approx(2.0 / 2**8)
    assert floors["six"] == pytest.approx(2.0 / 2**6)


def test_a_missing_value_is_not_read_as_a_zero():
    """Zero means "measured, and nothing there"; NaN means "not measured". Opposite claims."""
    absent = {f"D{i}": (1.0, None) if i < 4 else (1.0, 1.0) for i in range(8)}
    zeroed = {f"D{i}": (1.0, 0.0) if i < 4 else (1.0, 1.0) for i in range(8)}
    out_absent = paired_value_concordance(
        *build_values({"edge": absent}), case="Case", control="Control"
    )
    out_zeroed = paired_value_concordance(
        *build_values({"edge": zeroed}), case="Case", control="Control"
    )
    # Treated as absent, the four donors leave the test and the rest show no change.
    assert out_absent.iloc[0]["mean_delta"] == pytest.approx(0.0)
    # Treated as zero, the same four donors make it a unanimous-in-the-movers rise.
    assert out_zeroed.iloc[0]["mean_delta"] == pytest.approx(0.5)


def test_the_two_audits_agree_on_the_pattern_they_call():
    """One classifier, two vocabularies: a shift both can see must get the same name.

    Only for the types that actually move. ``C`` is deliberately excluded and gets its
    own test below: its count and its raw share are both flat, and its *log-ratio* is
    not, because two other types moved. That is a property of the scale rather than of
    the classifier, and it is the reason the compositional audit is not simply this
    function called on a table of shares.
    """

    counts, donors, conditions = build_pairs({f"D{i}": 60 + 5 * i for i in range(8)})
    shares = counts.div(counts.sum(axis=1), axis=0)
    composition = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control"
    )
    general = paired_value_concordance(
        shares, donors, conditions, case="Case", control="Control", unit="share"
    )
    for cell_type in ("A", "B"):
        expected = composition.loc[composition["cell_type"] == cell_type, "pattern"].iloc[0]
        assert expected == "consistent"
        assert general.loc[general["item"] == cell_type, "pattern"].iloc[0] == expected


def test_a_flat_share_is_not_a_flat_log_ratio():
    """The scales disagree by construction, so the general audit must not be used as a
    compositional one: ``C``'s share never moves, while its share *relative to the rest of
    the composition* rises in every donor because A gained and B lost."""

    counts, donors, conditions = build_pairs({f"D{i}": 60 + 5 * i for i in range(8)})
    shares = counts.div(counts.sum(axis=1), axis=0)
    composition = paired_abundance_concordance(
        counts, donors, conditions, case="Case", control="Control"
    )
    general = paired_value_concordance(shares, donors, conditions, case="Case", control="Control")
    assert composition.loc[composition["cell_type"] == "C", "pattern"].iloc[0] == "consistent"
    row = general[general["item"] == "C"].iloc[0]
    assert (row["direction"], row["mean_delta"]) == (0, 0.0)
    assert row["reason"].startswith("the cohort mean change is exactly zero")


def _alternating(n_donors: int) -> dict[str, tuple[float, float]]:
    """A donor set that moves both ways: five up, four down at nine donors."""

    return {f"D{i}": (1.0 + (0.1 if i % 2 == 0 else -0.1), 1.0) for i in range(n_donors)}


def test_the_conservative_sign_p_is_the_doubled_one_and_caps_at_one():
    """The direction is read off the same deltas, so the nominal one-sided p is halved.

    At nine unanimous donors the nominal value reaches ``2**-9``, below the ``2/2**9``
    that is the smallest p any relabelling of nine donors can produce. Doubling puts the
    column on the scale ``randomization_floor`` reports.
    """

    values, donors, conditions = build_values(
        {
            "real": {f"D{i}": (1.0 + 0.05 * i, 0.5) for i in range(9)},
            "noise": _alternating(9),
        }
    )
    out = paired_value_concordance(
        values, donors, conditions, case="Case", control="Control"
    ).set_index("item")
    assert out.loc["real", "sign_test_p"] == pytest.approx(2.0**-9)
    assert out.loc["real", "sign_test_p_conservative"] == pytest.approx(2.0**-8)
    assert out.loc["real", "sign_test_p_conservative"] == pytest.approx(
        out.loc["real", "design_floor_p"]
    )
    # Doubling a p of 0.5 must not produce a p of 1.5.
    assert out.loc["noise", "sign_test_p"] == pytest.approx(0.5)
    assert out.loc["noise", "sign_test_p_conservative"] == pytest.approx(1.0)


def test_an_item_can_be_pattern_consistent_and_not_survive_the_conservative_family():
    """``pattern`` is a donor-agreement verdict, not a significance call.

    The failure this pins is a caller reading ``pattern == "consistent"`` as "changed".
    Here one unanimous item sits in a family of twenty: it is consistent, it clears the
    nominal family bar, and it does not clear the same bar once the one-sided p is put on
    the two-sided scale. A "called" result needs both.
    """

    items = {"real": {f"D{i}": (1.0 + 0.05 * i, 0.5) for i in range(9)}}
    items.update({f"noise{i}": _alternating(9) for i in range(19)})
    values, donors, conditions = build_values(items)
    row = (
        paired_value_concordance(values, donors, conditions, case="Case", control="Control")
        .set_index("item")
        .loc["real"]
    )
    assert row["pattern"] == "consistent"
    assert row["sign_test_fdr"] < 0.05
    assert row["sign_test_fdr_conservative"] > 0.05


def test_an_item_with_no_direction_has_no_conservative_p_either():
    """NaN means there was nothing to make conservative, not a p of zero."""

    values, donors, conditions = build_values({"flat": {f"D{i}": (1.0, 1.0) for i in range(9)}})
    row = paired_value_concordance(values, donors, conditions, case="Case", control="Control").iloc[
        0
    ]
    assert np.isnan(row["sign_test_p"]) and np.isnan(row["sign_test_p_conservative"])


def test_degenerate_input_returns_the_schema_not_an_exception():
    empty = paired_value_concordance(
        pd.DataFrame(), pd.Series(dtype=str), pd.Series(dtype=str), case="Case", control="Control"
    )
    assert list(empty.columns) == list(VALUE_CONCORDANCE_COLUMNS)
    assert empty.empty


# --------------------------------------------------------------------------- #
# reading the verdict: two hurdles, and a unanimity that can be lost            #
# --------------------------------------------------------------------------- #
def test_a_call_needs_both_the_donors_and_the_family_correction():
    table = pd.DataFrame(
        {
            "item": ["both", "donors only", "fdr only", "neither"],
            "pattern": ["consistent", "consistent", "heterogeneous", "underpowered"],
            "sign_test_fdr_conservative": [0.01, 0.13, 0.01, 0.9],
        }
    )
    called = mark_called(table).set_index("item")["called"]
    assert called["both"]
    # The two failures this exists to prevent, one on each side.
    assert not called["donors only"]  # an FDR of 0.13 quoted as a finding
    assert not called["fdr only"]  # one outlying donor carrying the family
    assert not called["neither"]


def test_marking_a_table_that_never_ran_the_test_gives_a_column_of_false():
    """A figure reading ``called`` should draw nothing, not raise on a missing key."""
    out = mark_called(pd.DataFrame({"item": ["a", "b"]}))
    assert list(out["called"]) == [False, False]


def test_unanimity_cannot_be_won_by_having_almost_no_donors():
    """Pin the bug: ``n_agree >= n_pairs`` is true at 0/0 and nearly free at 3/3.

    Selecting rows this way puts the least-supported rows at the top of a table and draws
    empty rows in a figure, which is the opposite of what unanimity is for.
    """
    table = pd.DataFrame(
        {
            "item": ["never detected", "three", "six", "nine", "eight of nine"],
            "n_pairs": [0, 3, 6, 9, 9],
            "n_agree": [0, 3, 6, 9, 8],
        }
    )
    flag = donor_unanimous(table).to_numpy()
    assert list(flag) == [False, False, True, True, False]


def test_the_unanimity_floor_is_the_house_threshold_and_is_overridable():
    table = pd.DataFrame({"n_pairs": [4], "n_agree": [4]})
    assert not donor_unanimous(table).iloc[0]
    assert donor_unanimous(table, min_pairs=4).iloc[0]


def test_unanimity_on_a_frame_without_the_counts_is_false_not_an_error():
    assert not donor_unanimous(pd.DataFrame({"item": ["a"]})).iloc[0]
