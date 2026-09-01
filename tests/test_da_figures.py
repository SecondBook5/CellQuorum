"""Tests for differential-abundance figure data-prep helpers (synthetic fixtures)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.comparative.differential_abundance.da_figures import (
    milo_beeswarm_order,
    plot_milo_beeswarm,
    plot_sccoda_composition,
    prepare_milo_beeswarm,
    sccoda_composition_order,
    sccoda_single_reference,
)


def _milo_results() -> pd.DataFrame:
    """Milo output shape: one row per neighborhood, annotated by majority cell type.

    Row 5 has no majority cell type (unannotated) and must be dropped — it cannot
    be placed on the categorical y-axis of a beeswarm.
    """
    return pd.DataFrame(
        {
            "nhood": [1, 2, 3, 4, 5],
            "logFC": [2.0, 1.5, -2.0, 0.1, 1.0],
            "PValue": [0.001, 0.01, 0.001, 0.5, 0.2],
            "SpatialFDR": [0.02, 0.20, 0.03, 0.60, 0.30],
            "nhood_size": [50, 40, 45, 30, 20],
            "majority_celltype": ["TypeA", "TypeA", "TypeB", "TypeB", np.nan],
            "celltype_fraction": [0.9, 0.8, 0.95, 0.7, np.nan],
        }
    )


def test_prepare_milo_beeswarm_drops_unannotated_neighborhoods():
    """Neighborhoods without a majority cell type are removed (can't be placed)."""
    prepared = prepare_milo_beeswarm(_milo_results())

    assert len(prepared) == 4
    assert set(prepared["majority_celltype"]) == {"TypeA", "TypeB"}
    assert prepared["majority_celltype"].notna().all()


def test_prepare_milo_beeswarm_flags_significance_at_threshold():
    """A `significant` column marks neighborhoods below the SpatialFDR cutoff."""
    prepared = prepare_milo_beeswarm(_milo_results(), spatial_fdr=0.1)

    assert "significant" in prepared.columns
    by_nhood = prepared.set_index("nhood")["significant"]
    # 0.02 and 0.03 are below 0.1; 0.20 and 0.60 are not.
    assert bool(by_nhood[1]) is True
    assert bool(by_nhood[2]) is False
    assert bool(by_nhood[3]) is True
    assert bool(by_nhood[4]) is False


def test_prepare_milo_beeswarm_threshold_is_configurable():
    """Loosening the SpatialFDR cutoff turns more neighborhoods significant."""
    prepared = prepare_milo_beeswarm(_milo_results(), spatial_fdr=0.25)

    by_nhood = prepared.set_index("nhood")["significant"]
    # At 0.25, the 0.20 neighborhood now counts as significant too.
    assert bool(by_nhood[2]) is True
    assert bool(by_nhood[4]) is False


def _sccoda_results() -> pd.DataFrame:
    """scCODA output shape: two reference blocks (auto + explicit) stacked."""
    return pd.DataFrame(
        {
            "cell_type": ["TypeA", "TypeB", "TypeA", "TypeB"],
            "log2_fold_change": [1.5, -0.2, 1.4, -0.25],
            "inclusion_probability": [0.9, 0.3, 0.88, 0.28],
            "credible_effect": [True, False, True, False],
            "reference": ["auto", "auto", "TypeC", "TypeC"],
        }
    )


def test_sccoda_single_reference_selects_requested_block():
    """Only rows for the requested reference are returned."""
    auto = sccoda_single_reference(_sccoda_results(), reference="auto")
    assert len(auto) == 2
    assert set(auto["reference"]) == {"auto"}

    explicit = sccoda_single_reference(_sccoda_results(), reference="TypeC")
    assert len(explicit) == 2
    assert set(explicit["reference"]) == {"TypeC"}


def test_sccoda_single_reference_falls_back_when_missing():
    """A reference that isn't present falls back to the first block available."""
    picked = sccoda_single_reference(_sccoda_results(), reference="does_not_exist")
    assert len(picked) == 2
    # 'auto' is the first reference value in the frame.
    assert set(picked["reference"]) == {"auto"}


def test_sccoda_single_reference_tolerates_missing_column():
    """A frame with no `reference` column is returned unchanged."""
    df = pd.DataFrame(
        {
            "cell_type": ["TypeA", "TypeB"],
            "log2_fold_change": [1.5, -0.2],
            "inclusion_probability": [0.9, 0.3],
            "credible_effect": [True, False],
        }
    )
    picked = sccoda_single_reference(df, reference="auto")
    assert len(picked) == 2
    assert list(picked["cell_type"]) == ["TypeA", "TypeB"]


def test_milo_beeswarm_order_is_median_logfc_desc():
    """Rows are ordered by per-cell-type median logFC, descending (case-enriched top)."""
    prepared = prepare_milo_beeswarm(_milo_results())
    # TypeA median(2.0, 1.5) = 1.75; TypeB median(-2.0, 0.1) = -0.95.
    assert milo_beeswarm_order(prepared) == ["TypeA", "TypeB"]


def test_plot_milo_beeswarm_returns_figure_with_ordered_rows():
    """The beeswarm is a Figure with one y-tick per cell type in median-logFC order."""
    fig = plot_milo_beeswarm(_milo_results(), case="LE", control="N", spatial_fdr=0.1)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["TypeA", "TypeB"]


def _composition_props() -> pd.DataFrame:
    """Tidy per-sample proportions: 2 control (N) + 2 case (LE) samples, 3 cell types.

    Case (LE) mean proportions: TypeC 0.1 < TypeA 0.3 < TypeB 0.6.
    """
    rows = []
    per_sample = {
        "d1_N": ("d1", "N", {"TypeA": 0.6, "TypeB": 0.3, "TypeC": 0.1}),
        "d2_N": ("d2", "N", {"TypeA": 0.6, "TypeB": 0.3, "TypeC": 0.1}),
        "d1_LE": ("d1", "LE", {"TypeA": 0.3, "TypeB": 0.6, "TypeC": 0.1}),
        "d2_LE": ("d2", "LE", {"TypeA": 0.3, "TypeB": 0.6, "TypeC": 0.1}),
    }
    for sample, (donor, cond, props) in per_sample.items():
        for ct, prop in props.items():
            rows.append(
                {
                    "sample": sample,
                    "donor": donor,
                    "condition": cond,
                    "cell_type": ct,
                    "count": int(prop * 100),
                    "proportion": prop,
                }
            )
    return pd.DataFrame(rows)


def _sccoda_single() -> pd.DataFrame:
    """A single-reference scCODA result over the same three cell types."""
    return pd.DataFrame(
        {
            "cell_type": ["TypeA", "TypeB", "TypeC"],
            "log2_fold_change": [-1.2, 0.9, 0.05],
            "inclusion_probability": [0.85, 0.92, 0.2],
            "credible_effect": [True, True, False],
            "reference": ["auto", "auto", "auto"],
        }
    )


def test_sccoda_composition_order_is_case_proportion_ascending():
    """Cell types are ordered by case mean proportion, ascending (smallest at bottom)."""
    order = sccoda_composition_order(_sccoda_single(), _composition_props(), case="LE")
    assert order == ["TypeC", "TypeA", "TypeB"]


def test_plot_sccoda_composition_returns_two_panel_figure():
    """scCODA figure has two panels; the credibility panel has one bar per cell type."""
    fig = plot_sccoda_composition(_sccoda_single(), _composition_props(), case="LE", control="N")
    assert isinstance(fig, Figure)
    # Panel A (dumbbell) + Panel B (inclusion bar) = at least two axes.
    assert len(fig.axes) >= 2
    # Panel B holds one horizontal bar per cell type (3).
    bar_panel = fig.axes[1]
    assert len(bar_panel.containers[0]) == 3
