"""Tests for differential-abundance composition figure helpers (synthetic fixtures)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.comparative.differential_abundance.composition_figure import (
    composition_cell_type_order,
    plot_condition_composition,
    plot_per_patient_composition,
    pooled_condition_proportions,
)


def _proportions() -> pd.DataFrame:
    """Tidy composition proportions for 2 control + 2 case samples, 3 cell types.

    Pooled totals: TypeA is the most abundant overall, then TypeB, then TypeC.
    Each sample sums to 1.0 across its cell types.
    """
    rows = []
    per_sample = {
        # sample: (donor, condition, {cell_type: count})
        "d1_N": ("d1", "N", {"TypeA": 60, "TypeB": 30, "TypeC": 10}),
        "d2_N": ("d2", "N", {"TypeA": 60, "TypeB": 30, "TypeC": 10}),
        "d1_LE": ("d1", "LE", {"TypeA": 30, "TypeB": 60, "TypeC": 10}),
        "d2_LE": ("d2", "LE", {"TypeA": 30, "TypeB": 60, "TypeC": 10}),
    }
    for sample, (donor, cond, counts) in per_sample.items():
        total = sum(counts.values())
        for ct, count in counts.items():
            rows.append(
                {
                    "sample": sample,
                    "donor": donor,
                    "condition": cond,
                    "cell_type": ct,
                    "count": count,
                    "proportion": count / total,
                }
            )
    return pd.DataFrame(rows)


def test_composition_cell_type_order_is_abundance_desc():
    """Cell types are ordered by pooled abundance, descending."""
    order = composition_cell_type_order(_proportions())
    assert order == ["TypeA", "TypeB", "TypeC"]


def test_composition_cell_type_order_tiebreak_alphabetical():
    """Equal-abundance cell types fall back to alphabetical order."""
    df = pd.DataFrame(
        {
            "sample": ["s1", "s1"],
            "donor": ["d1", "d1"],
            "condition": ["N", "N"],
            "cell_type": ["Zeta", "Alpha"],
            "count": [10, 10],
            "proportion": [0.5, 0.5],
        }
    )
    assert composition_cell_type_order(df) == ["Alpha", "Zeta"]


def test_pooled_condition_proportions_sums_to_one_and_ordered():
    """Pooled fractions sum to 1 within each condition; rows control-then-case."""
    frac = pooled_condition_proportions(_proportions(), case="LE", control="N")

    assert list(frac.index) == ["N", "LE"]
    assert np.allclose(frac.sum(axis=1).to_numpy(), 1.0)
    # Control (N) pooled: TypeA 120/200 = 0.6, TypeB 60/200 = 0.3, TypeC 20/200 = 0.1
    assert np.isclose(frac.loc["N", "TypeA"], 0.6)
    assert np.isclose(frac.loc["N", "TypeB"], 0.3)
    # Case (LE) pooled: TypeA 60/200 = 0.3, TypeB 120/200 = 0.6
    assert np.isclose(frac.loc["LE", "TypeA"], 0.3)
    assert np.isclose(frac.loc["LE", "TypeB"], 0.6)


def test_plot_condition_composition_returns_stacked_figure():
    """Condition figure is a Figure with one bar per (condition x cell type)."""
    fig = plot_condition_composition(_proportions(), case="LE", control="N")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 2 conditions x 3 cell types = 6 stacked bar rectangles.
    assert len(ax.patches) == 6
    # y-axis is a 0-100 percentage scale.
    assert ax.get_ylim() == (0.0, 100.0)


def test_plot_per_patient_composition_returns_stacked_figure():
    """Per-patient figure is a Figure with one bar per (sample x cell type)."""
    fig = plot_per_patient_composition(_proportions(), case="LE", control="N")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 4 samples x 3 cell types = 12 stacked bar rectangles.
    assert len(ax.patches) == 12
    # One x tick per sample.
    assert len(ax.get_xticks()) == 4
