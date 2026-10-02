"""Tests for factorial design-matrix estimability checks.

These cover the general multi-factor / interaction design layer that sits under
DE/DA: building the model matrix from arbitrary factor columns and detecting the
non-estimable structures (rank deficiency, confounded factors, empty factorial
cells) that a two-level case/control check cannot catch.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.config.design import (
    analyze_design,
    build_design_matrix,
    validate_design_matrix,
)
from cellquorum.core.exceptions import CellQuorumConfigError


def _balanced_2x2() -> pd.DataFrame:
    """A fully crossed, replicated 2x2 factorial (genotype x treatment)."""
    rows = []
    for geno in ("wt", "ko"):
        for treat in ("veh", "drug"):
            for rep in range(2):  # two replicates per cell
                rows.append({"genotype": geno, "treatment": treat, "rep": rep})
    return pd.DataFrame(rows)


def test_build_design_matrix_additive_has_intercept_and_dummies():
    meta = _balanced_2x2()

    dm = build_design_matrix(meta, factors=["genotype", "treatment"])

    # Intercept + one dummy per factor (drop-first coding on a 2-level factor).
    assert "Intercept" in dm.columns
    assert dm.shape[0] == len(meta)
    assert dm.shape[1] == 3  # intercept + genotype[ko] + treatment[veh|drug]
    # Intercept is all ones.
    assert (dm["Intercept"] == 1).all()


def test_analyze_design_balanced_factorial_is_full_rank():
    meta = _balanced_2x2()

    report = analyze_design(meta, factors=["genotype", "treatment"])

    assert report.full_rank is True
    assert report.rank == report.n_columns
    assert report.confounded_pairs == []
    assert report.empty_cells == []


def test_analyze_design_interaction_is_estimable_when_all_cells_filled():
    meta = _balanced_2x2()

    report = analyze_design(
        meta, factors=["genotype", "treatment"], interactions=[("genotype", "treatment")]
    )

    # Intercept + geno + treat + geno:treat = 4 estimable columns, all filled.
    assert report.n_columns == 4
    assert report.full_rank is True
    assert report.empty_cells == []


def test_analyze_design_flags_confounded_nested_factor():
    # batch is perfectly nested in condition: b1/b2 only appear in "case",
    # b3/b4 only in "control". condition and batch cannot be separated.
    meta = pd.DataFrame(
        {
            "condition": ["case", "case", "control", "control"],
            "batch": ["b1", "b2", "b3", "b4"],
        }
    )

    report = analyze_design(meta, factors=["condition", "batch"])

    assert report.full_rank is False
    assert ("batch", "condition") in report.confounded_pairs or (
        "condition",
        "batch",
    ) in report.confounded_pairs


def test_analyze_design_reports_empty_factorial_cell():
    # ko+drug combination is entirely absent -> the interaction is not estimable
    # and the missing cell must be reported (never a silent gap).
    meta = pd.DataFrame(
        {
            "genotype": ["wt", "wt", "ko", "ko"],
            "treatment": ["veh", "drug", "veh", "veh"],
        }
    )

    report = analyze_design(
        meta, factors=["genotype", "treatment"], interactions=[("genotype", "treatment")]
    )

    assert report.empty_cells  # non-empty: at least the (ko, drug) cell
    combos = {tuple(sorted(cell.items())) for cell in report.empty_cells}
    assert (("genotype", "ko"), ("treatment", "drug")) in combos


def test_validate_design_matrix_raises_on_rank_deficiency():
    meta = pd.DataFrame(
        {
            "condition": ["case", "case", "control", "control"],
            "batch": ["b1", "b2", "b3", "b4"],
        }
    )

    with pytest.raises(CellQuorumConfigError, match="not estimable|rank"):
        validate_design_matrix(meta, factors=["condition", "batch"])


def test_validate_design_matrix_raises_on_missing_column():
    meta = pd.DataFrame({"condition": ["case", "control"]})

    with pytest.raises(CellQuorumConfigError, match="nonexistent"):
        validate_design_matrix(meta, factors=["condition", "nonexistent"])


def test_validate_design_matrix_returns_report_on_clean_design():
    meta = _balanced_2x2()

    report = validate_design_matrix(meta, factors=["genotype", "treatment"])

    assert report.full_rank is True
