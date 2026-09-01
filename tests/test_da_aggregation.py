"""Tests for differential abundance aggregation helpers."""

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.comparative.differential_abundance.aggregation import (
    aggregate_celltype_counts,
    build_cell_distribution_summary,
    build_composition_proportions,
)


def _adata():
    """Create a minimal test AnnData with 2 donors x 2 conditions x 3 cell type observations."""
    obs = pd.DataFrame(
        {
            "patient_id": (["d1"] * 6 + ["d2"] * 6),
            "condition": (["Normal"] * 3 + ["LE"] * 3) * 2,
            "cell_type": (["Tcell", "Fib", "Fib"] * 4),
        }
    )
    a = ad.AnnData(X=np.zeros((12, 4)), obs=obs)
    return a


def test_aggregate_celltype_counts_shape_and_totals():
    """Test basic aggregation shape, cell counts, and condition recording."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    # 2 donors x 2 conditions = up to 4 samples; 2 cell types.
    assert set(res.counts.columns) == {"Tcell", "Fib"}
    assert res.counts.values.sum() == 12
    # Each sample's condition is recorded.
    assert set(res.sample_meta["condition"]).issubset({"Normal", "LE"})


def test_aggregate_celltype_counts_row_sums():
    """Test that row sums match cell counts per sample."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    # Each sample's row sum should match the number of cells in that sample
    # d1, Normal: 3 cells; d1, LE: 3 cells; d2, Normal: 3 cells; d2, LE: 3 cells
    expected_totals = {
        "d1_Normal": 3,
        "d1_LE": 3,
        "d2_Normal": 3,
        "d2_LE": 3,
    }
    for sample_id, expected_count in expected_totals.items():
        if sample_id in res.counts.index:
            assert res.counts.loc[sample_id].sum() == expected_count


def test_aggregate_celltype_counts_integer_values():
    """Test that counts are integer-valued."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    assert res.counts.values.dtype == np.int64 or res.counts.values.dtype == int


def test_aggregate_celltype_counts_meta_alignment():
    """Test that sample_meta is properly aligned with counts index."""
    a = _adata()
    res = aggregate_celltype_counts(
        a, donor_col="patient_id", condition_col="condition", cell_type_col="cell_type"
    )
    # sample_meta should have the same index as counts
    assert (res.counts.index == res.sample_meta.index).all()
    # sample_meta should have donor_col and condition_col
    assert "patient_id" in res.sample_meta.columns
    assert "condition" in res.sample_meta.columns


def _summary_inputs():
    """Sample x cell-type counts + condition series with a clean pooled composition.

    LE (case) pools to A=12, B=4 (total 16) -> A 75%, B 25%.
    N  (control) pools to A=4, B=12 (total 16) -> A 25%, B 75%.
    An extra 'Other' condition sample must be ignored entirely.
    """
    counts = pd.DataFrame(
        {
            "TypeB": [2, 2, 6, 6, 99],
            "TypeA": [6, 6, 2, 2, 99],
        },
        index=["d1_LE", "d2_LE", "d1_N", "d2_N", "d3_Other"],
    )
    conditions = pd.Series(
        ["LE", "LE", "N", "N", "Other"],
        index=["d1_LE", "d2_LE", "d1_N", "d2_N", "d3_Other"],
    )
    return counts, conditions


def test_build_cell_distribution_summary_pooled_counts_and_relative():
    """Absolute = pooled counts per condition; relative = within-condition percent."""
    counts, conditions = _summary_inputs()
    test_results = pd.DataFrame(
        {"cell_type": ["TypeA", "TypeB"], "pvalue": [0.01, 0.03], "fdr": [0.02, 0.04]}
    )

    summary = build_cell_distribution_summary(
        counts, conditions, case="LE", control="N", test_results=test_results
    )

    # Rows are alphabetical by cell type; TypeC is absent.
    assert list(summary["cell_type"]) == ["TypeA", "TypeB"]

    row_a = summary.set_index("cell_type").loc["TypeA"]
    assert row_a["case_absolute"] == 12
    assert row_a["control_absolute"] == 4
    assert row_a["case_relative_pct"] == 75.0
    assert row_a["control_relative_pct"] == 25.0
    # p / FDR are attached to the case group only.
    assert row_a["case_pvalue"] == 0.01
    assert row_a["case_adj_pvalue"] == 0.02

    row_b = summary.set_index("cell_type").loc["TypeB"]
    assert row_b["case_absolute"] == 4
    assert row_b["control_absolute"] == 12
    assert row_b["case_relative_pct"] == 25.0
    assert row_b["control_relative_pct"] == 75.0


def test_build_cell_distribution_summary_ignores_other_conditions():
    """Samples whose condition is neither case nor control never contribute."""
    counts, conditions = _summary_inputs()

    summary = build_cell_distribution_summary(counts, conditions, case="LE", control="N")

    # The d3_Other sample's 99/99 counts must not leak into either arm.
    total_absolute = summary["case_absolute"].sum() + summary["control_absolute"].sum()
    assert total_absolute == 32  # 16 case + 16 control, Other excluded


def test_build_cell_distribution_summary_without_test_is_nan():
    """No test_results -> p/adjp columns exist but are NaN (stable schema)."""
    counts, conditions = _summary_inputs()

    summary = build_cell_distribution_summary(counts, conditions, case="LE", control="N")

    assert "case_pvalue" in summary.columns
    assert "case_adj_pvalue" in summary.columns
    assert summary["case_pvalue"].isna().all()
    assert summary["case_adj_pvalue"].isna().all()


def test_build_cell_distribution_summary_relative_sums_to_100():
    """Within each condition, relative percentages sum to 100."""
    counts, conditions = _summary_inputs()

    summary = build_cell_distribution_summary(counts, conditions, case="LE", control="N")

    assert round(summary["case_relative_pct"].sum(), 6) == 100.0
    assert round(summary["control_relative_pct"].sum(), 6) == 100.0


def _composition_inputs():
    """Per-sample counts + condition/donor series for composition proportions.

    Two case (LE) samples and two control (N) samples, each of total 8 cells:
      d1_LE/d2_LE -> TypeA 6, TypeB 2  (A 0.75, B 0.25)
      d1_N /d2_N  -> TypeA 2, TypeB 6  (A 0.25, B 0.75)
    An extra 'Other' condition sample (huge counts) must be dropped entirely.
    """
    counts = pd.DataFrame(
        {
            "TypeB": [2, 2, 6, 6, 99],
            "TypeA": [6, 6, 2, 2, 99],
        },
        index=["d1_LE", "d2_LE", "d1_N", "d2_N", "d3_Other"],
    )
    conditions = pd.Series(
        ["LE", "LE", "N", "N", "Other"],
        index=["d1_LE", "d2_LE", "d1_N", "d2_N", "d3_Other"],
    )
    donors = pd.Series(
        ["d1", "d2", "d1", "d2", "d3"],
        index=["d1_LE", "d2_LE", "d1_N", "d2_N", "d3_Other"],
    )
    return counts, conditions, donors


def test_build_composition_proportions_schema_and_rows():
    """Tidy long format: one row per (in-scope sample, cell type) with expected columns."""
    counts, conditions, donors = _composition_inputs()

    comp = build_composition_proportions(counts, conditions, donors, case="LE", control="N")

    assert list(comp.columns) == [
        "sample",
        "donor",
        "condition",
        "cell_type",
        "count",
        "proportion",
    ]
    # 4 in-scope samples x 2 cell types = 8 rows; the 'Other' sample is dropped.
    assert len(comp) == 8
    assert set(comp["condition"]) == {"LE", "N"}
    assert "d3_Other" not in set(comp["sample"])


def test_build_composition_proportions_within_sample_values():
    """Proportion is within-sample count/total and matches counts exactly."""
    counts, conditions, donors = _composition_inputs()

    comp = build_composition_proportions(counts, conditions, donors, case="LE", control="N")

    row = comp[(comp["sample"] == "d1_LE") & (comp["cell_type"] == "TypeA")].iloc[0]
    assert row["count"] == 6
    assert row["proportion"] == 0.75
    assert row["donor"] == "d1"

    row_n = comp[(comp["sample"] == "d1_N") & (comp["cell_type"] == "TypeA")].iloc[0]
    assert row_n["proportion"] == 0.25


def test_build_composition_proportions_sums_to_one_per_sample():
    """Within every sample, cell-type proportions sum to 1."""
    counts, conditions, donors = _composition_inputs()

    comp = build_composition_proportions(counts, conditions, donors, case="LE", control="N")

    per_sample = comp.groupby("sample")["proportion"].sum()
    assert np.allclose(per_sample.to_numpy(), 1.0)


def test_build_composition_proportions_control_first_ordering():
    """Rows are ordered control arm first, then case, then donor, then cell type."""
    counts, conditions, donors = _composition_inputs()

    comp = build_composition_proportions(counts, conditions, donors, case="LE", control="N")

    # The control ('N') block precedes the case ('LE') block.
    first_case_pos = comp.index[comp["condition"] == "LE"][0]
    last_control_pos = comp.index[comp["condition"] == "N"][-1]
    assert last_control_pos < first_case_pos
    # Cell types are alphabetical within a sample.
    d1_n = comp[comp["sample"] == "d1_N"]
    assert list(d1_n["cell_type"]) == ["TypeA", "TypeB"]
