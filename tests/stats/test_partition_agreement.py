"""Agreement between two clusterings, and the direction the agreement runs in."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.partition_agreement import (
    AGREEMENT_COLUMNS,
    COVERAGE_COLUMNS,
    CROSSTAB_COLUMNS,
    align_partitions,
    cluster_group_support,
    label_composition,
    partition_agreement,
    partition_crosstab,
)


def _cells(n: int) -> list[str]:
    return [f"cell{i:03d}" for i in range(n)]


def _coarse_and_refined():
    """A three-label partition and an eight-cluster strict refinement of it.

    This is the shape the question usually arrives in: a coarse subtype call and a
    higher-resolution clustering, and what the reader needs to know is whether the second
    is the first at finer grain or a different cut of the data.
    """
    index = _cells(120)
    coarse = pd.Series(["A"] * 60 + ["B"] * 40 + ["C"] * 20, index=index)
    # Each coarse label splits into disjoint fine clusters; no fine cluster straddles.
    fine = pd.Series(
        ["a1"] * 30 + ["a2"] * 20 + ["a3"] * 10 + ["b1"] * 25 + ["b2"] * 15 + ["c1"] * 20,
        index=index,
    )
    return coarse, fine


def test_a_refinement_is_asymmetric_and_the_asymmetry_is_the_answer():
    """ARI alone cannot separate "different cut" from "same cut, finer"; purity can."""
    coarse, fine = _coarse_and_refined()
    aligned, _ = align_partitions({"coarse": coarse, "fine": fine})
    row = partition_agreement(aligned).iloc[0]

    assert list(partition_agreement(aligned).columns) == AGREEMENT_COLUMNS
    # Every fine cluster sits wholly inside one coarse label.
    assert row["purity_a_in_b"] < 1.0  # coarse -> fine: a coarse label spans several
    assert row["purity_b_in_a"] == pytest.approx(1.0)
    assert row["n_clusters_a"] == 3
    assert row["n_clusters_b"] == 6
    # And the symmetric index is middling despite the perfect nesting, which is the point.
    assert 0.0 < row["ari"] < 0.8


def test_identical_partitions_agree_perfectly_in_both_directions():
    coarse, _ = _coarse_and_refined()
    aligned, _ = align_partitions({"one": coarse, "two": coarse.copy()})
    row = partition_agreement(aligned).iloc[0]
    assert row["ari"] == pytest.approx(1.0)
    assert row["ami"] == pytest.approx(1.0)
    assert row["purity_a_in_b"] == pytest.approx(1.0)
    assert row["purity_b_in_a"] == pytest.approx(1.0)


def test_a_shuffled_partition_has_no_agreement_but_still_has_purity():
    """Purity is not an agreement index and must not be read as one.

    A random partition into two halves still has purity ~0.5 against a two-label truth,
    because every cluster has *some* modal counterpart. The purities are only interpretable
    beside the indices, which is why both are returned in one row.
    """
    index = _cells(200)
    truth = pd.Series((["A"] * 100) + (["B"] * 100), index=index)
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.permutation(truth.to_numpy()), index=index)
    row = partition_agreement(align_partitions({"truth": truth, "noise": noise})[0]).iloc[0]
    assert abs(row["ari"]) < 0.1
    assert 0.4 < row["purity_a_in_b"] < 0.7


def test_the_intersection_is_reported_not_silently_taken():
    """Two partitions from different objects overlap without matching; say by how much."""
    a = pd.Series(["A"] * 50 + ["B"] * 50, index=_cells(100))
    b = pd.Series(["x"] * 40 + ["y"] * 40, index=_cells(120)[40:])
    aligned, coverage = align_partitions({"a": a, "b": b})

    assert list(coverage.columns) == COVERAGE_COLUMNS
    assert len(aligned) == 60  # cell040..cell099
    cov = coverage.set_index("partition")
    assert cov.loc["a", "n_labelled"] == 100
    assert cov.loc["a", "n_only_here"] == 40
    assert cov.loc["b", "n_only_here"] == 20
    assert cov.loc["a", "n_shared"] == 60
    # The index is over the intersection, not over either partition's own cells.
    assert partition_agreement(aligned).iloc[0]["n_cells"] == 60


def test_unlabelled_cells_are_not_a_cluster():
    """Both partitions failing to label the same cell is not the two of them agreeing."""
    index = _cells(100)
    a = pd.Series(["A"] * 50 + [None] * 50, index=index)
    b = pd.Series(["x"] * 50 + [np.nan] * 50, index=index)
    aligned, coverage = align_partitions({"a": a, "b": b})

    assert coverage.set_index("partition").loc["a", "n_unlabelled_in_shared"] == 50
    assert coverage.set_index("partition").loc["a", "n_clusters"] == 1
    row = partition_agreement(aligned).iloc[0]
    assert row["n_cells"] == 50
    assert row["n_clusters_a"] == 1 and row["n_clusters_b"] == 1


def test_the_string_nan_a_csv_round_trip_leaves_behind_is_also_unlabelled():
    """A partition written to CSV and read back brings its NaNs back as text."""
    index = _cells(20)
    a = pd.Series(["A"] * 10 + ["nan"] * 10, index=index)
    b = pd.Series(["x"] * 10 + [""] * 10, index=index)
    _, coverage = align_partitions({"a": a, "b": b})
    assert coverage.set_index("partition").loc["a", "n_clusters"] == 1
    assert coverage.set_index("partition").loc["b", "n_clusters"] == 1


def test_partitions_that_share_no_cells_raise_rather_than_return_nan():
    """An empty intersection is a barcode-convention mismatch, not a weak agreement."""
    a = pd.Series(["A"] * 10, index=[f"x{i}" for i in range(10)])
    b = pd.Series(["B"] * 10, index=[f"y{i}" for i in range(10)])
    with pytest.raises(ValueError, match="share no cell ids"):
        align_partitions({"a": a, "b": b})


def test_one_partition_is_not_an_agreement():
    with pytest.raises(ValueError, match="at least two"):
        align_partitions({"only": pd.Series(["A"], index=["cell000"])})


def test_three_partitions_give_all_three_pairs():
    coarse, fine = _coarse_and_refined()
    aligned, _ = align_partitions({"coarse": coarse, "fine": fine, "flat": coarse.copy()})
    table = partition_agreement(aligned)
    assert len(table) == 3
    assert set(zip(table["partition_a"], table["partition_b"], strict=True)) == {
        ("coarse", "fine"),
        ("coarse", "flat"),
        ("fine", "flat"),
    }


def test_the_crosstab_carries_both_fractions_because_they_answer_different_questions():
    coarse, fine = _coarse_and_refined()
    aligned, _ = align_partitions({"coarse": coarse, "fine": fine})
    table = partition_crosstab(aligned, "coarse", "fine")

    assert list(table.columns) == CROSSTAB_COLUMNS
    # Only the non-empty cells: a strict refinement has one row per fine cluster.
    assert len(table) == 6
    # A fine cluster is wholly inside its coarse label, but is only part of it.
    a1 = table[table["cluster_b"] == "a1"].iloc[0]
    assert a1["frac_of_b"] == pytest.approx(1.0)
    assert a1["frac_of_a"] == pytest.approx(30 / 60)
    # Each source cluster's dominant destination reads first.
    first_per_a = table.drop_duplicates("cluster_a")
    assert list(first_per_a["cluster_b"]) == ["a1", "b1", "c1"]


def test_a_cluster_can_be_one_donors_cells_wearing_a_clusters_name():
    """Cell count says nothing about replication; the supporting-donor count does."""
    labels = pd.Series(["big"] * 2000 + ["small"] * 200)
    donors = pd.Series(
        ["D1"] * 1000
        + ["D2"] * 1000
        + [f"D{i}" for i in range(1, 10) for _ in range(20)]
        + ["D1"] * 20
    )
    table = cluster_group_support(labels, donors, min_cells_per_group=10).set_index("cluster")

    assert table.loc["big", "n_cells"] == 2000
    assert table.loc["big", "n_groups_supporting"] == 2
    assert table.loc["small", "n_cells"] == 200
    assert table.loc["small", "n_groups_supporting"] == 9
    assert table.loc["big", "max_group_frac"] == pytest.approx(0.5)


def test_groups_below_the_cell_floor_are_counted_separately_not_dropped():
    """A donor contributing three cells is real and is not a unit a model can fit."""
    labels = pd.Series(["c"] * 33)
    donors = pd.Series(["D1"] * 30 + ["D2", "D3", "D4"])
    row = cluster_group_support(labels, donors, min_cells_per_group=10).iloc[0]
    assert row["n_groups"] == 4
    assert row["n_groups_supporting"] == 1
    assert row["groups_supporting"] == "D1"


def test_composition_reports_the_cluster_makeup_and_the_abundance_reading():
    labels = pd.Series(["A"] * 30 + ["B"] * 10)
    condition = pd.Series(["Case"] * 20 + ["Control"] * 10 + ["Case"] * 10)
    table = label_composition(labels, condition, order=["Case", "Control"]).set_index("cluster")

    assert table.loc["A", "n_Case"] == 20
    assert table.loc["A", "frac_within_cluster_Case"] == pytest.approx(20 / 30)
    # 30 Case cells overall, 20 of them in A.
    assert table.loc["A", "frac_of_condition_Case"] == pytest.approx(20 / 30)
    assert table.loc["B", "frac_of_condition_Control"] == pytest.approx(0.0)


def test_a_condition_absent_from_one_arm_still_gets_its_columns():
    """Two arms' tables have to stay column-compatible to be read side by side."""
    labels = pd.Series(["A"] * 10)
    condition = pd.Series(["Case"] * 10)
    table = label_composition(labels, condition, order=["Case", "Control"])
    assert table["n_Control"].tolist() == [0]
    assert table["frac_of_condition_Control"].tolist() == [0.0]
