"""Activity change by group: the donor-aware, depth-adjusted change per source, as a heatmap."""

from __future__ import annotations

import matplotlib
import numpy as np
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


def test_activity_grid_displays_donor_supported_changes_in_requested_order():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    fig = plot_activity_change(_table(), groups=["Structural", "CV", "Inflamed"], value_label=LABEL)

    try:
        assert any(a.get_ylabel() == LABEL for a in fig.axes)

        ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert ticks == ["Structural", "CV", "Inflamed"]

        outlines = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
        assert len(outlines) == 2  # TF1 in Structural and TF2 in Inflamed, FDR < 0.05

        words = [t.get_text() for t in fig.axes[0].texts]
        assert words.count("n<3") == 4  # CV has 2 pairs for every source

        ax = fig.axes[0]
        image = ax.images[0]
        cmap, norm = image.get_cmap(), image.norm
        data = np.ma.getdata(image.get_array())
        labels = [t for t in ax.texts if t.get_text() != "n<3"]
        assert labels
        for text in labels:
            column, row = int(round(text.get_position()[0])), int(round(text.get_position()[1]))
            r, g, b = cmap(norm(data[row, column]))[:3]
            background = 0.2126 * r + 0.7152 * g + 0.0722 * b
            r2, g2, b2 = to_rgb(text.get_color())
            foreground = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2
            assert abs(foreground - background) > 0.45, (text.get_text(), foreground, background)
    finally:
        plt.close(fig)


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


def test_pinned_sources_are_always_shown_even_when_their_change_is_small():
    fig = plot_activity_change(
        _table(),
        groups=["Structural", "CV", "Inflamed"],
        value_label=LABEL,
        max_rows=2,
        pin=["TF3"],
    )

    rows = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert "TF3" in rows  # weakest source of all, shown because it was asked for
