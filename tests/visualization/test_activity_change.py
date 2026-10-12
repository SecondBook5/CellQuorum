"""Activity change by group: the donor-aware, depth-adjusted change per source, as a heatmap."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

from cellquorum.visualization.activity_change import plot_activity_change  # noqa: E402

LABEL = "Change in activity at equal depth (Lymphedema - Normal)"


def _table() -> pd.DataFrame:
    rows = []
    spec = {
        "TF1": {"Structural": (2.0, 0.001, 8), "Inflamed": (0.5, 0.4, 7), "CV": (0.1, 0.9, 2)},
        "TF2": {"Structural": (-1.0, 0.2, 8), "Inflamed": (-2.5, 0.01, 7), "CV": (0.3, 0.8, 2)},
        "TF3": {"Structural": (0.2, 0.7, 8), "Inflamed": (0.1, 0.9, 7), "CV": (0.0, 1.0, 2)},
        "TF4": {"Structural": (3.0, 0.5, 8), "Inflamed": (-3.0, 0.5, 7), "CV": (0.0, 1.0, 2)},
    }
    for source, groups in spec.items():
        for group, (delta, fdr, n) in groups.items():
            rows.append(
                {"source": source, "group": group, "delta": delta, "fdr": fdr, "n_pairs": n}
            )
    return pd.DataFrame(rows)


def test_the_colourbar_names_the_change_and_the_comparison():
    fig = plot_activity_change(_table(), groups=["Structural", "CV", "Inflamed"], value_label=LABEL)

    assert any(a.get_ylabel() == LABEL for a in fig.axes)


def test_columns_follow_the_order_given_not_alphabetical():
    fig = plot_activity_change(_table(), groups=["Structural", "CV", "Inflamed"], value_label=LABEL)

    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert ticks == ["Structural", "CV", "Inflamed"]


def test_a_significant_cell_is_outlined_and_the_others_are_not():
    fig = plot_activity_change(_table(), groups=["Structural", "CV", "Inflamed"], value_label=LABEL)

    outlines = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert len(outlines) == 2  # TF1 in Structural and TF2 in Inflamed, FDR < 0.05


def test_a_group_with_too_few_donor_pairs_is_marked_not_coloured():
    fig = plot_activity_change(_table(), groups=["Structural", "CV", "Inflamed"], value_label=LABEL)

    words = [t.get_text() for t in fig.axes[0].texts]
    assert words.count("n<3") == 4  # CV has 2 pairs for every source


def test_a_significant_source_is_never_dropped_to_fit_the_row_cap():
    fig = plot_activity_change(
        _table(), groups=["Structural", "CV", "Inflamed"], value_label=LABEL, max_rows=2
    )

    rows = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert {"TF1", "TF2"} <= rows  # significant, kept although TF4 has the largest changes


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="fdr"):
        plot_activity_change(_table().drop(columns="fdr"), groups=["Structural"], value_label=LABEL)


def test_a_group_without_rows_is_refused_by_name():
    with pytest.raises(ValueError, match="Missing"):
        plot_activity_change(_table(), groups=["Structural", "Missing"], value_label=LABEL)
