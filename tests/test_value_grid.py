"""The value grid's three visual states, and the one that used to erase its own numbers.

A grid carries a low value, a value that failed a threshold, and no value at all, and the
tests here pin each to a distinguishable mark rather than to a particular colour. The cell
text gets the most attention because its bug was invisible in review: text colour used to be
chosen from where the value sat on the scale, which is right for a diverging map, where both
ends are dark, and wrong for a sequential one, where the low end is near-white and "extremes
take light text" paints white on white. A number that is not there does not look wrong, it
looks like an empty cell, so the rule is asserted directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellquorum.visualization.grids import _text_on, value_grid
from cellquorum.visualization.measured_layout import ABSENT, INK, measure_labels_in


def _grid(values: pd.DataFrame, **kwargs):
    fig, ax = value_grid(
        values,
        cmap=kwargs.pop("cmap", "Blues"),
        vmin=kwargs.pop("vmin", 0.0),
        vmax=kwargs.pop("vmax", 1.0),
        colorbar_label="share",
        notes=["one note"],
        row_label="rows",
        col_label="cols",
        **kwargs,
    )
    return fig, ax


# --------------------------------------------------------------------------- #
# cell text stays legible
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("cmap", "vmin", "vmax", "value"),
    [
        # Sequential: the low end is the near-white end, and it needs dark text.
        ("Blues", 0.0, 1.0, 0.0),
        ("Blues", 0.0, 1.0, 0.02),
        ("Greys", 0.0, 1.0, 0.05),
        # Diverging: the low end is dark, and needs light text.
        ("RdBu_r", -3.0, 3.0, -3.0),
    ],
)
def test_text_colour_follows_the_painted_cell_not_the_value_rank(cmap, vmin, vmax, value):
    fig, ax = _grid(
        pd.DataFrame([[value]], index=["r"], columns=["c"]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annotate=pd.DataFrame([["17"]], index=["r"], columns=["c"]),
    )
    (text,) = [t for t in ax.texts if t.get_text() == "17"]
    painted = plt.get_cmap(cmap)((value - vmin) / (vmax - vmin))
    assert text.get_color() == _text_on(painted)
    plt.close(fig)


def test_a_pale_cell_and_a_dark_cell_get_different_text_colours():
    values = pd.DataFrame([[0.02, 0.98]], index=["r"], columns=["pale", "dark"])
    fig, ax = _grid(
        values,
        annotate=pd.DataFrame([["4", "783"]], index=["r"], columns=["pale", "dark"]),
    )
    colours = {t.get_text(): t.get_color() for t in ax.texts if t.get_text() in {"4", "783"}}
    assert colours["4"] == INK
    assert colours["783"] == "white"
    plt.close(fig)


def test_an_absent_cell_keeps_dark_text_because_its_fill_is_the_neutral_grey():
    fig, ax = _grid(
        pd.DataFrame([[np.nan]], index=["r"], columns=["c"]),
        annotate=pd.DataFrame([["n/a"]], index=["r"], columns=["c"]),
    )
    (text,) = [t for t in ax.texts if t.get_text() == "n/a"]
    assert text.get_color() == INK
    assert _text_on(matplotlib.colors.to_rgba(ABSENT)) == INK
    plt.close(fig)


def test_blank_and_missing_annotations_write_no_text():
    values = pd.DataFrame([[0.5, 0.5, 0.5]], index=["r"], columns=["a", "b", "c"])
    fig, ax = _grid(
        values,
        annotate=pd.DataFrame([["", None, "9"]], index=["r"], columns=["a", "b", "c"]),
    )
    assert [t.get_text() for t in ax.texts] == ["9"]
    plt.close(fig)


# --------------------------------------------------------------------------- #
# absent, and below-threshold
# --------------------------------------------------------------------------- #


def test_a_missing_value_is_painted_neutral_rather_than_at_an_end_of_the_scale():
    values = pd.DataFrame([[np.nan, 1.0]], index=["r"], columns=["absent", "high"])
    fig, ax = _grid(values)
    (mesh,) = ax.images
    fill = mesh.cmap(np.ma.masked_invalid(values.to_numpy(dtype=float)))[0, 0]
    assert tuple(np.round(fill[:3], 6)) == tuple(np.round(matplotlib.colors.to_rgba(ABSENT)[:3], 6))
    plt.close(fig)


def test_below_threshold_cells_are_outlined_and_the_others_are_not():
    values = pd.DataFrame([[0.2, 0.8]], index=["r"], columns=["weak", "strong"])
    below = pd.DataFrame([[True, False]], index=["r"], columns=["weak", "strong"])
    fig, ax = _grid(values, below=below)
    outlines = [p for p in ax.patches if not p.get_fill()]
    assert len(outlines) == 1
    assert outlines[0].get_linestyle() != "solid"
    plt.close(fig)


def _outline_centre(patch) -> tuple[float, float]:
    """Where a below-threshold outline sits, in the cell coordinates of the grid."""
    x, y = patch.get_xy()
    return x + patch.get_width() / 2, y + patch.get_height() / 2


def test_the_outline_marks_the_cell_the_flag_names():
    values = pd.DataFrame([[0.2, 0.8], [0.4, 0.6]], index=["r0", "r1"], columns=["c0", "c1"])
    below = pd.DataFrame([[False, False], [False, True]], index=["r0", "r1"], columns=["c0", "c1"])
    fig, ax = _grid(values, below=below)
    (outline,) = [p for p in ax.patches if not p.get_fill()]
    # Cell (row 1, col 1) is centred on (1, 1) in the image's own coordinates.
    assert _outline_centre(outline) == pytest.approx((1.0, 1.0))
    plt.close(fig)


def test_a_below_frame_in_another_order_is_realigned_not_taken_positionally():
    values = pd.DataFrame([[0.2, 0.8], [0.4, 0.6]], index=["r0", "r1"], columns=["c0", "c1"])
    below = pd.DataFrame([[True, False], [False, False]], index=["r1", "r0"], columns=["c0", "c1"])
    fig, ax = _grid(values, below=below)
    (outline,) = [p for p in ax.patches if not p.get_fill()]
    # The flag belongs to r1/c0, which is row 1 of ``values`` however ``below`` was ordered.
    assert _outline_centre(outline) == pytest.approx((0.0, 1.0))
    plt.close(fig)


def test_two_adjacent_outlines_do_not_touch():
    """On a grid where most cells are marked, an outline on the boundary is shared geometry.

    A dash sitting exactly between two marked cells cannot be attributed to either of them, so
    the mark stops carrying information on precisely the grids that use it most. The outlines
    are inset, and the invariant is that neighbouring ones leave a visible gap.
    """
    values = pd.DataFrame([[0.2, 0.8]], index=["r"], columns=["c0", "c1"])
    below = pd.DataFrame([[True, True]], index=["r"], columns=["c0", "c1"])
    fig, ax = _grid(values, below=below)
    left, right = sorted((p for p in ax.patches if not p.get_fill()), key=lambda p: p.get_xy()[0])
    assert left.get_width() < 1.0
    assert left.get_xy()[0] + left.get_width() < right.get_xy()[0]
    plt.close(fig)


# --------------------------------------------------------------------------- #
# shape and labels
# --------------------------------------------------------------------------- #


def test_a_tall_narrow_grid_stays_tall_and_narrow():
    tall = pd.DataFrame(np.linspace(0, 1, 40).reshape(20, 2))
    wide = pd.DataFrame(np.linspace(0, 1, 40).reshape(2, 20))
    fig_tall, _ = _grid(tall)
    fig_wide, _ = _grid(wide)
    assert fig_tall.get_figheight() > fig_wide.get_figheight()
    assert fig_wide.get_figwidth() > fig_tall.get_figwidth()
    plt.close(fig_tall)
    plt.close(fig_wide)


def test_a_colourbar_label_longer_than_its_bar_is_wrapped_to_the_bar():
    """
    Pin the bug: the bar is as tall as the data area, so on a landscape grid it is an inch
    tall while the label naming the quantity is three inches long. matplotlib does not wrap
    it — it draws it centred and running off both ends of the figure, over the notes at one
    end and off the page at the other, and the colour is left unexplained.
    """
    values = pd.DataFrame(np.linspace(0, 1, 45).reshape(3, 15))
    long_label = "collectri activity, lymphedema minus normal (mixed-model coefficient)"
    fig, _ = value_grid(
        values,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        colorbar_label=long_label,
        notes=["one note"],
        row_label="rows",
        col_label="cols",
    )
    try:
        (bar_axes,) = [a for a in fig.axes if a.get_ylabel().startswith("collectri")]
        drawn = bar_axes.get_ylabel()
        assert "\n" in drawn
        assert drawn.split() == long_label.split()
        # Per line, against the bar's height: a rotated label longer than its bar is drawn
        # off the end of it.
        bar_h_in = bar_axes.get_position().height * fig.get_figheight()
        assert all(w <= bar_h_in for w in measure_labels_in(drawn.split("\n"), fontsize=8))
    finally:
        plt.close(fig)


def test_an_axis_label_longer_than_its_axis_is_wrapped_and_stays_on_the_figure():
    """The row label runs *along* the rows, so on a three-row landscape grid "how the score
    was computed" is longer than the panel is tall and matplotlib centres it off the top."""
    values = pd.DataFrame(np.linspace(0, 1, 27).reshape(3, 9))
    fig, ax = value_grid(
        values,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        colorbar_label="effect",
        notes=["one note"],
        row_label="how the score was computed",
        col_label="collectri source · subtype",
    )
    try:
        assert "\n" in ax.get_ylabel()
        assert ax.get_ylabel().split() == "how the score was computed".split()
        # The guarantee is per line: no line of the row label is longer than the panel is
        # tall, which is what keeps a rotated label inside the figure.
        data_h_in = ax.get_position().height * fig.get_figheight()
        widths = measure_labels_in(ax.get_ylabel().split("\n"), fontsize=8.5)
        assert all(width <= data_h_in for width in widths)
    finally:
        plt.close(fig)


def test_a_short_colourbar_label_is_left_on_one_line():
    values = pd.DataFrame(np.linspace(0, 1, 45).reshape(3, 15))
    fig, _ = _grid(values)
    try:
        (bar_axes,) = [a for a in fig.axes if a.get_ylabel() == "share"]
        assert "\n" not in bar_axes.get_ylabel()
    finally:
        plt.close(fig)


def test_the_index_and_columns_are_the_tick_labels_in_the_order_given():
    values = pd.DataFrame([[0.1, 0.2]], index=["only row"], columns=["second", "first"])
    fig, ax = _grid(values)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["only row"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["second", "first"]
    assert ax.get_xlabel() == "cols"
    assert ax.get_ylabel() == "rows"
    plt.close(fig)
