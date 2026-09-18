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


def test_plot_milo_beeswarm_cluster_labels_rename_yticks_only():
    """Bare cell-type ids on the y-axis are meaningless without cross-referencing --

    cluster_labels renames them for display; row assignment/order (still keyed
    off the raw id internally) is unaffected.
    """
    fig = plot_milo_beeswarm(
        _milo_results(),
        case="LE",
        control="N",
        spatial_fdr=0.1,
        cluster_labels={"TypeA": "Papillary", "TypeB": "Reticular"},
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["Papillary", "Reticular"]


def _milo_results_integer_celltypes() -> pd.DataFrame:
    """Real Milo output shape from a numeric cluster label (fibroblast_subcluster
    0-8): majority_celltype comes out int64, NOT string -- unlike every fixture
    above. Real bug, confirmed live: milo_beeswarm_order stringifies the group
    labels for axis ticks (str(ct)), but the main plotting loop still filters
    `prepared["majority_celltype"] == ct` against those STRING values, so the
    int64 column never matches anything. Every scatter call plots zero points,
    matplotlib autoscales the x-axis to near-nothing around 0, and the "n/N"
    text annotations -- placed at the REAL data max, computed separately from
    the unfiltered table -- land wildly outside that collapsed axis. Saving
    with bbox_inches="tight" then expands the canvas to reach them: a real run
    produced a 116,970 x 1,613px PNG, all its actual content crushed into a
    sliver at the left edge.
    """
    return pd.DataFrame(
        {
            "nhood": list(range(1, 7)),
            "logFC": [2.0, 1.5, -2.0, 0.1, 1.0, -1.2],
            "PValue": [0.001, 0.01, 0.001, 0.5, 0.2, 0.03],
            "SpatialFDR": [0.02, 0.20, 0.03, 0.60, 0.30, 0.04],
            "nhood_size": [50, 40, 45, 30, 20, 35],
            "majority_celltype": [0, 0, 1, 1, 2, 2],
            "celltype_fraction": [0.9, 0.8, 0.95, 0.7, 0.85, 0.88],
        }
    )


def test_plot_milo_beeswarm_actually_plots_points_for_integer_celltypes():
    """The real bug: with an int64 majority_celltype column (Milo's real
    output shape for a numeric cluster label), every group must still end up
    with its points on the canvas -- not silently filtered to zero by a
    string/int comparison mismatch.
    """
    fig = plot_milo_beeswarm(
        _milo_results_integer_celltypes(), case="LE", control="N", spatial_fdr=0.1
    )
    ax = fig.axes[0]
    scatter_collections = [c for c in ax.collections if c.get_offsets().size > 0]
    total_points = sum(len(c.get_offsets()) for c in scatter_collections)
    assert total_points == 6  # every neighborhood row must be plotted


def test_plot_milo_beeswarm_xlim_reflects_the_real_data_range():
    """A direct pin on the axis-blowout symptom: the x-axis must scale to the
    actual logFC data, not collapse to near-zero because nothing was plotted.
    """
    fig = plot_milo_beeswarm(
        _milo_results_integer_celltypes(), case="LE", control="N", spatial_fdr=0.1
    )
    ax = fig.axes[0]
    xmin, xmax = ax.get_xlim()
    assert xmax - xmin > 1.0  # not the ~0.1-wide collapsed range from the bug


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


def test_plot_sccoda_composition_cluster_labels_rename_yticks_only():
    fig = plot_sccoda_composition(
        _sccoda_single(),
        _composition_props(),
        case="LE",
        control="N",
        cluster_labels={"TypeA": "Papillary", "TypeB": "Reticular", "TypeC": "Perivascular"},
    )
    ax_a = fig.axes[0]
    labels = [t.get_text() for t in ax_a.get_yticklabels()]
    # sccoda_composition_order is TypeC, TypeA, TypeB (ascending case proportion).
    assert labels == ["Perivascular", "Papillary", "Reticular"]
