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
