"""TDD tests for cellquorum.stages.composition.compute (pure pandas, no AnnData)."""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.stages.composition.compute import grouped_composition, overall_composition


def test_overall_composition_counts_and_fractions():
    cell_type = pd.Series(["A", "A", "B", "A", "B", "B"])
    table = overall_composition(cell_type).set_index("cell_type")
    assert table.loc["A", "n_cells"] == 3
    assert table.loc["B", "n_cells"] == 3
    assert table.loc["A", "fraction"] == 0.5
    assert table["fraction"].sum() == 1.0


def test_overall_composition_orders_by_descending_abundance():
    cell_type = pd.Series(["rare"] + ["common"] * 5 + ["mid"] * 2)
    table = overall_composition(cell_type)
    assert list(table["cell_type"]) == ["common", "mid", "rare"]


def test_grouped_composition_rows_sum_to_100_percent():
    cell_type = pd.Series(["A", "A", "B", "A", "B", "B"])
    group = pd.Series(["g1", "g1", "g1", "g2", "g2", "g2"])
    table = grouped_composition(cell_type, group)
    assert (table.sum(axis=1).round(6) == 100.0).all()


def test_grouped_composition_reports_the_exact_percentage():
    cell_type = pd.Series(["A", "A", "B"])  # one group: 2/3 A, 1/3 B
    group = pd.Series(["g1", "g1", "g1"])
    table = grouped_composition(cell_type, group)
    assert table.loc["g1", "A"] == pytest.approx(200 / 3)
    assert table.loc["g1", "B"] == pytest.approx(100 / 3)


def test_grouped_composition_columns_are_cell_types_rows_are_groups():
    cell_type = pd.Series(["A", "B", "A", "B"])
    group = pd.Series(["x", "x", "y", "y"])
    table = grouped_composition(cell_type, group)
    assert set(table.columns) == {"A", "B"}
    assert set(table.index) == {"x", "y"}


def test_grouped_composition_fills_a_cell_type_absent_from_one_group_with_zero():
    """A group with zero cells of some cell type must show 0%, not be dropped."""
    cell_type = pd.Series(["A", "A", "A", "A", "A", "B"])
    group = pd.Series(["g1", "g1", "g1", "g2", "g2", "g1"])
    table = grouped_composition(cell_type, group)
    assert table.loc["g2", "B"] == 0.0


def test_grouped_composition_column_order_matches_overall_abundance():
    """Columns should be ordered by overall abundance, matching overall_composition,
    so a caller can reindex a stacked-bar figure consistently across both tables."""
    cell_type = pd.Series(["rare", "common", "common", "common", "mid", "mid"])
    group = pd.Series(["g1"] * 6)
    table = grouped_composition(cell_type, group)
    assert list(table.columns) == ["common", "mid", "rare"]


def test_groups_follow_the_requested_order_not_the_alphabet():
    from cellquorum.stages.composition.compute import grouped_composition

    cell_type = pd.Series(["A", "B", "A", "B"])
    group = pd.Series(["Lymphedema", "Lymphedema", "Normal", "Normal"])

    table = grouped_composition(cell_type, group, group_order=["Normal", "Lymphedema"])

    assert list(table.index) == ["Normal", "Lymphedema"]


def test_groups_the_order_does_not_name_follow_it_in_sorted_order():
    from cellquorum.stages.composition.compute import grouped_composition

    cell_type = pd.Series(["A"] * 6)
    group = pd.Series(["Zed", "Normal", "Alpha", "Normal", "Zed", "Alpha"])

    table = grouped_composition(cell_type, group, group_order=["Normal"])

    assert list(table.index) == ["Normal", "Alpha", "Zed"]


def test_a_group_in_the_order_that_has_no_cells_is_left_out_not_invented():
    from cellquorum.stages.composition.compute import grouped_composition

    table = grouped_composition(
        pd.Series(["A", "A"]), pd.Series(["Normal", "Normal"]), group_order=["Normal", "Ghost"]
    )

    assert list(table.index) == ["Normal"]


# ---- donor-paired test of each cell type's share ------------------------------------------


def _by_patient(shift_a: float, n: int = 8):
    """Control and case tables (patients x cell types, percent) with cell type A shifted."""
    donors = [f"P{i}" for i in range(1, n + 1)]
    rng = pd.Series(range(n))
    noise = (rng % 3 - 1).to_numpy() * 1.5  # small, mixed-sign, never zero for every donor
    control = pd.DataFrame({"A": 40.0 + noise, "B": 60.0 - noise}, index=donors)
    case = pd.DataFrame(
        {"A": 40.0 + shift_a + noise[::-1], "B": 60.0 - shift_a - noise[::-1]}, index=donors
    )
    return {"Normal": control, "Lymphedema": case}


def test_a_share_that_rises_in_every_donor_is_significant_in_the_paired_test():
    from cellquorum.stages.composition.compute import donor_paired_tests

    table = donor_paired_tests(_by_patient(shift_a=12.0), control="Normal", case="Lymphedema")

    row = table.set_index("cell_type").loc["A"]
    assert row["n_pairs"] == 8
    assert row["mean_diff_pct"] == pytest.approx(12.0, abs=0.5)
    assert row["p"] < 0.05 and row["padj"] < 0.05 and bool(row["significant"])


def test_a_share_with_no_consistent_direction_is_not_significant():
    from cellquorum.stages.composition.compute import donor_paired_tests

    table = donor_paired_tests(_by_patient(shift_a=0.0), control="Normal", case="Lymphedema")

    assert not table["significant"].any()


def test_only_donors_present_in_both_conditions_are_paired():
    from cellquorum.stages.composition.compute import donor_paired_tests

    tables = _by_patient(shift_a=12.0, n=8)
    tables["Lymphedema"] = tables["Lymphedema"].drop(index=["P1", "P2"])

    table = donor_paired_tests(tables, control="Normal", case="Lymphedema")

    assert set(table["n_pairs"]) == {6}


def test_too_few_pairs_gives_no_p_value_rather_than_a_misleading_one():
    from cellquorum.stages.composition.compute import donor_paired_tests

    tables = {k: v.iloc[:3] for k, v in _by_patient(shift_a=12.0).items()}

    table = donor_paired_tests(tables, control="Normal", case="Lymphedema", min_pairs=4)

    assert table["p"].isna().all() and table["padj"].isna().all()
    assert not table["significant"].any()
    assert set(table["n_pairs"]) == {3}


def test_the_difference_is_case_minus_control():
    from cellquorum.stages.composition.compute import donor_paired_tests

    table = donor_paired_tests(_by_patient(shift_a=12.0), control="Normal", case="Lymphedema")

    diffs = table.set_index("cell_type")["mean_diff_pct"]
    assert diffs["A"] > 0 > diffs["B"]
