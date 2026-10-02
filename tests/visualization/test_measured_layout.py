"""Inch-measured layout: does the space reserved match the space the text takes?

Three modules draw panels whose text lives outside the axes, and all three used to
carry their own copy of this arithmetic with the same three bugs. These tests pin the
arithmetic rather than any one figure, so a fix here is a fix for every panel: a gutter
measured instead of guessed, an axes that is square on the page before ``aspect="equal"``
gets a chance to shrink it, and notes that wrap to the figure instead of stretching it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from cellquorum.visualization.measured_layout import (
    CHAR_IN,
    NOTE_LINE_IN,
    grid_canvas,
    measure_labels_in,
    row_panel_canvas,
    square_matrix_canvas,
    stacked_panel_canvas,
    widest_label_in,
    wrap_label,
    wrap_notes,
    write_notes,
    xy_panel_canvas,
)

LONG_NOTE = (
    "One-sided hypergeometric test against 18,412 testable genes, Benjamini-Hochberg "
    "across the 55 pairs; * FDR < 0.05, ** < 0.01, *** < 0.001."
)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def test_a_longer_label_measures_wider_and_an_empty_list_measures_nothing():
    widths = measure_labels_in(["i", "mmmmmmmmmm"])
    assert widths[1] > widths[0] > 0.0
    assert measure_labels_in([]) == []
    assert widest_label_in([]) == 0.0


def test_the_measurement_beats_a_character_count_on_narrow_glyphs():
    """The bug this replaces: a constant inches-per-character over-reserves 40% of the
    width on a panel of narrow names and clips the labels on a panel of wide ones."""
    narrow = "iiiiiiiiiiiiiiii"
    assert measure_labels_in([narrow])[0] < CHAR_IN * len(narrow)


def test_the_measurement_scales_with_the_font_it_will_be_drawn_at():
    small, large = widest_label_in(["program name"], 6.0), widest_label_in(["program name"], 12.0)
    assert large > small * 1.5


# --------------------------------------------------------------------------- #
# wrapping
# --------------------------------------------------------------------------- #


def test_a_note_that_fits_is_returned_untouched():
    assert wrap_notes(["short"], width_in=6.0) == ["short"]


def test_a_note_that_does_not_fit_is_broken_into_lines_that_do():
    lines = wrap_notes([LONG_NOTE], width_in=2.5)
    assert len(lines) > 1
    assert all(width <= 2.5 for width in measure_labels_in(lines, fontsize=7.5))
    # Nothing is dropped in the process: the words are the same words.
    assert " ".join(lines).split() == LONG_NOTE.split()


def test_wrapping_is_per_note_so_one_long_caveat_does_not_break_up_the_others():
    lines = wrap_notes(["short one", LONG_NOTE, "short two"], width_in=2.5)
    assert lines[0] == "short one"
    assert lines[-1] == "short two"


def test_a_narrower_figure_wraps_the_same_note_onto_more_lines():
    assert len(wrap_notes([LONG_NOTE], width_in=2.5)) > len(wrap_notes([LONG_NOTE], width_in=5.0))


def test_an_unwrappably_narrow_figure_lets_the_note_run_wide_rather_than_stacking_words():
    """A column of one word per line is less legible than a note that overhangs."""
    lines = wrap_notes([LONG_NOTE], width_in=0.2)
    assert max(len(line.split()) for line in lines) > 1


def test_continuation_lines_are_indented_so_one_note_does_not_read_as_two():
    lines = wrap_notes([LONG_NOTE], width_in=2.5)
    assert not lines[0].startswith(" ")
    assert all(line.startswith("  ") for line in lines[1:])


def test_a_label_is_wrapped_to_lines_that_all_fit_however_narrow_the_room_is():
    """The guarantee ``wrap_notes`` deliberately does not make. A note may overhang the
    figure rather than break to one word per line; a colourbar label may not, because
    overhanging its bar means drawn off the page."""
    label = "collectri activity, lymphedema minus normal (mixed-model coefficient)"
    for width_in in (0.85, 1.4, 2.5):
        lines = wrap_label(label, width_in=width_in, fontsize=8.0)
        assert all(w <= width_in for w in measure_labels_in(lines, fontsize=8.0))
        assert " ".join(lines).split() == label.split()


def test_a_label_that_fits_is_returned_untouched_and_never_hanging_indented():
    assert wrap_label("share", width_in=6.0) == ["share"]
    lines = wrap_label("a colourbar label long enough to wrap twice over", width_in=1.0)
    assert len(lines) > 1
    # A centred rotated label with an indented second line reads as a misalignment.
    assert all(line == line.strip() for line in lines)


def test_a_single_unbreakable_word_is_returned_rather_than_split_mid_word():
    (line,) = wrap_label("phosphatidylinositol-3-kinase", width_in=0.2)
    assert line == "phosphatidylinositol-3-kinase"


def test_a_blank_note_is_kept_as_a_blank_line_rather_than_dropped():
    """Callers use one to separate two groups of notes; silently removing it would
    reflow the block and leave the reserved room a line too tall."""
    assert wrap_notes(["a", "", "b"], width_in=6.0) == ["a", "", "b"]


# --------------------------------------------------------------------------- #
# the square canvas
# --------------------------------------------------------------------------- #


def test_the_axes_is_square_in_inches_not_only_in_data_units():
    """``aspect="equal"`` is imposed at draw time. An axes that is not already square
    gets shrunk and re-centred then, leaving a dead band that ``bbox_inches="tight"``
    cannot reclaim because the layout genuinely occupies it."""
    fig, ax, _ = square_matrix_canvas(n=6, label_in=widest_label_in(["apical junction"]))
    try:
        box = ax.get_position()
        assert box.width * fig.get_figwidth() == pytest.approx(
            box.height * fig.get_figheight(), abs=0.01
        )
    finally:
        plt.close(fig)


def test_the_row_label_gutter_is_the_measured_label_width_plus_the_tick_pad():
    label_in = widest_label_in(["a_very_long_program_name_indeed"])
    fig, ax, _ = square_matrix_canvas(n=4, label_in=label_in)
    try:
        gutter = ax.get_position().x0 * fig.get_figwidth()
        assert label_in < gutter < label_in + 0.3
    finally:
        plt.close(fig)


def test_dropping_the_colourbar_returns_its_width_to_the_page():
    with_bar = square_matrix_canvas(n=4, label_in=0.5)
    without = square_matrix_canvas(n=4, label_in=0.5, colorbar=False)
    try:
        assert with_bar[2] is not None
        assert without[2] is None
        assert without[0].get_figwidth() < with_bar[0].get_figwidth()
    finally:
        plt.close(with_bar[0])
        plt.close(without[0])


def test_a_long_note_grows_the_data_area_rather_than_a_dead_band():
    """Every panel here is saved with ``bbox_inches="tight"``, so a note wider than the
    figure widens the figure. If the extra width is not spent on the panel it is spent
    on nothing, and the figure's aspect ratio becomes a function of its word count."""
    plain = square_matrix_canvas(n=3, label_in=0.5)
    noted = square_matrix_canvas(n=3, label_in=0.5, notes=[LONG_NOTE])
    try:
        plain_data = plain[1].get_position().width * plain[0].get_figwidth()
        noted_data = noted[1].get_position().width * noted[0].get_figwidth()
        assert noted_data > plain_data
    finally:
        plt.close(plain[0])
        plt.close(noted[0])


def test_the_growth_is_capped_so_a_wordy_caveat_cannot_inflate_a_small_matrix():
    fig, ax, _ = square_matrix_canvas(n=3, label_in=0.5, notes=[LONG_NOTE * 4], max_data_in=3.0)
    try:
        assert ax.get_position().width * fig.get_figwidth() == pytest.approx(3.0, abs=0.01)
    finally:
        plt.close(fig)


def test_the_room_below_the_axes_counts_wrapped_lines_not_notes():
    """Reserving one line for a note that renders as three draws the last two over the
    column labels."""
    one = square_matrix_canvas(n=3, label_in=0.5, notes=["short"])
    wrapped = square_matrix_canvas(n=3, label_in=0.5, notes=[LONG_NOTE * 3])
    try:
        below_one = one[1].get_position().y0 * one[0].get_figheight()
        below_wrapped = wrapped[1].get_position().y0 * wrapped[0].get_figheight()
        assert below_wrapped > below_one + NOTE_LINE_IN
    finally:
        plt.close(one[0])
        plt.close(wrapped[0])


# --------------------------------------------------------------------------- #
# writing the notes
# --------------------------------------------------------------------------- #


def test_the_notes_are_stacked_in_reading_order_below_the_axes():
    fig, ax, _ = square_matrix_canvas(n=3, label_in=0.5, notes=["first", "second", "third"])
    try:
        write_notes(fig, ["first", "second", "third"])
        by_text = {note.get_text(): note.get_position()[1] for note in fig.texts}
        assert by_text["first"] > by_text["second"] > by_text["third"]
        top = max(by_text.values()) * fig.get_figheight()
        assert top <= ax.get_position().y0 * fig.get_figheight() + 0.01
    finally:
        plt.close(fig)


def test_a_written_note_stays_inside_the_figure_it_is_written_on():
    fig, _ax, _ = square_matrix_canvas(n=3, label_in=0.5, notes=[LONG_NOTE])
    try:
        write_notes(fig, [LONG_NOTE])
        renderer = fig.canvas.get_renderer()
        limit = fig.get_figwidth() * fig.dpi
        assert all(note.get_window_extent(renderer).x1 <= limit + 1.0 for note in fig.texts)
    finally:
        plt.close(fig)


def test_the_pitch_is_inches_so_the_notes_do_not_crowd_on_a_short_figure():
    """Placed in figure fractions, the same three notes crowd together on a short
    figure and drift apart on a tall one."""
    short = plt.figure(figsize=(6.0, 3.0))
    tall = plt.figure(figsize=(6.0, 9.0))
    try:
        for fig in (short, tall):
            write_notes(fig, ["a", "b"])
        gap_in = [
            abs(fig.texts[0].get_position()[1] - fig.texts[1].get_position()[1])
            * fig.get_figheight()
            for fig in (short, tall)
        ]
        assert gap_in[0] == pytest.approx(gap_in[1], abs=1e-9)
        assert gap_in[0] == pytest.approx(NOTE_LINE_IN, abs=1e-9)
    finally:
        plt.close(short)
        plt.close(tall)


# --------------------------------------------------------------------------- #
# the row-per-item panel
# --------------------------------------------------------------------------- #


def test_the_row_panel_reserves_the_measured_gutter_and_nothing_more():
    label_in = widest_label_in(["Integrin-focal adhesion/Actomyosin contractility"])
    fig, ax = row_panel_canvas(n_rows=6, label_in=label_in, data_in=3.6)
    try:
        # The axes starts where the labels end, plus the stated pad — not at a fraction
        # of the figure, which is what moves when the label set changes.
        assert ax.get_position().x0 * fig.get_figwidth() == pytest.approx(label_in + 0.18)
        assert ax.get_position().width * fig.get_figwidth() == pytest.approx(3.6)
    finally:
        plt.close(fig)


def test_the_row_panel_data_width_is_the_same_however_long_the_labels_are():
    """The cross-figure invariant: a coefficient of 0.6 has to be the same length in
    every panel, or the comparison the panels exist for is not available."""
    narrow, wide = (
        row_panel_canvas(n_rows=4, label_in=widest_label_in(labels), data_in=3.6)
        for labels in (["A/B"], ["A very long program name/Another long program name"])
    )
    try:
        widths = [
            fig.axes[0].get_position().width * fig.get_figwidth() for fig, _ in (narrow, wide)
        ]
        assert widths[0] == pytest.approx(widths[1]) == pytest.approx(3.6)
        # The figure grew instead, which is what a wider gutter should cost.
        assert wide[0].get_figwidth() > narrow[0].get_figwidth()
    finally:
        plt.close(narrow[0])
        plt.close(wide[0])


def test_the_row_panel_grows_by_one_row_height_per_row():
    small, large = (row_panel_canvas(n_rows=n, label_in=0.5, data_in=3.0) for n in (12, 20))
    try:
        heights = [
            fig.axes[0].get_position().height * fig.get_figheight() for fig, _ in (small, large)
        ]
        assert heights[1] - heights[0] == pytest.approx(8 * 0.24)
    finally:
        plt.close(small[0])
        plt.close(large[0])


def test_a_short_row_panel_is_not_a_strip():
    fig, ax = row_panel_canvas(n_rows=2, label_in=0.5, data_in=3.0)
    try:
        assert ax.get_position().height * fig.get_figheight() == pytest.approx(1.6)
    finally:
        plt.close(fig)


def test_the_row_panel_leaves_the_notes_room_below_the_x_axis_label():
    """The bug this pins: folding the x-axis room into the axes height leaves the notes
    stacked from the figure's bottom edge with the axis printed through them."""
    fig, ax = row_panel_canvas(n_rows=8, label_in=0.5, data_in=3.0, notes=[LONG_NOTE, "second"])
    try:
        write_notes(fig, [LONG_NOTE, "second"])
        renderer = fig.canvas.get_renderer()
        top_of_notes = max(note.get_window_extent(renderer).y1 for note in fig.texts) / fig.dpi
        # The axis label and ticks live in the 0.58 in below the axes; the notes are below
        # that, so their top must clear it.
        assert top_of_notes <= ax.get_position().y0 * fig.get_figheight() - 0.4
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# the two-continuous-axis panel
# --------------------------------------------------------------------------- #


def test_the_xy_panel_is_square_in_inches_unless_told_otherwise():
    """An axis-against-comparable-axis panel has to be square on the page, or a point's
    distance from the diagonal is read off two different scales."""
    fig, ax = xy_panel_canvas(data_in=3.4)
    try:
        box = ax.get_position()
        assert (
            box.width * fig.get_figwidth()
            == pytest.approx(box.height * fig.get_figheight())
            == pytest.approx(3.4)
        )
    finally:
        plt.close(fig)


def test_the_xy_panel_reserves_the_stated_axis_room_and_nothing_more():
    fig, ax = xy_panel_canvas(data_in=3.0, yaxis_in=0.7)
    try:
        assert ax.get_position().x0 * fig.get_figwidth() == pytest.approx(0.7)
    finally:
        plt.close(fig)


def test_the_xy_panel_data_width_survives_a_wordy_caveat():
    """The same cross-figure invariant as the row panel: a decade of p-value is one length."""
    plain, wordy = (
        xy_panel_canvas(data_in=3.0, notes=notes) for notes in ((), [LONG_NOTE, LONG_NOTE])
    )
    try:
        widths = [
            fig.axes[0].get_position().width * fig.get_figwidth() for fig, _ in (plain, wordy)
        ]
        assert widths[0] == pytest.approx(widths[1]) == pytest.approx(3.0)
        # The caveat bought height, not width.
        assert wordy[0].get_figheight() > plain[0].get_figheight()
        assert wordy[0].get_figwidth() == pytest.approx(plain[0].get_figwidth())
    finally:
        plt.close(plain[0])
        plt.close(wordy[0])


def test_the_xy_panel_does_not_print_its_x_axis_label_through_the_notes():
    fig, ax = xy_panel_canvas(data_in=3.0, notes=[LONG_NOTE, "second"])
    try:
        write_notes(fig, [LONG_NOTE, "second"])
        renderer = fig.canvas.get_renderer()
        top_of_notes = max(note.get_window_extent(renderer).y1 for note in fig.texts) / fig.dpi
        assert top_of_notes <= ax.get_position().y0 * fig.get_figheight() - 0.4
    finally:
        plt.close(fig)


def test_a_non_square_xy_panel_is_allowed_but_must_be_asked_for():
    fig, ax = xy_panel_canvas(data_in=4.0, height_in=2.0)
    try:
        box = ax.get_position()
        assert box.width * fig.get_figwidth() == pytest.approx(4.0)
        assert box.height * fig.get_figheight() == pytest.approx(2.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(("data_in", "height_in"), [(0.0, None), (-1.0, None), (3.0, 0.0)])
def test_an_xy_panel_with_no_area_is_refused(data_in, height_in):
    """Silently returning a zero-height axes produces a figure with no panel in it."""
    with pytest.raises(ValueError):
        xy_panel_canvas(data_in=data_in, height_in=height_in)


# --------------------------------------------------------------------------- #
# the rectangular grid panel
# --------------------------------------------------------------------------- #


def test_the_grid_keeps_its_cells_square_in_inches_not_only_in_data_units():
    """A 22-by-12 grid of 3:1 cells reads as though the columns matter more than the rows."""
    fig, ax, _ = grid_canvas(n_rows=22, n_cols=12, row_label_in=0.6, col_label_in=0.8)
    try:
        box = ax.get_position()
        cell_w = box.width * fig.get_figwidth() / 12
        cell_h = box.height * fig.get_figheight() / 22
        assert cell_h == pytest.approx(cell_w, rel=0.02)
    finally:
        plt.close(fig)


def test_a_narrow_tall_grid_is_not_widened_into_a_skyscraper():
    """
    Pin the bug: the ``min_data_in`` floor is on the *width*, and the height is derived from
    the cell size, so on a few-column grid the floor pays for its width in height. Observed:
    a 15-row 3-column footprint grid asked for 0.34 in cells, was widened to the 1.8 in
    floor, and came out 9 in tall; the single-column version of the same panel got 1.8 in
    cells and was 27 in tall — a figure no page can hold and no reader can scan.
    """
    for n_cols in (1, 3):
        fig, ax, _ = grid_canvas(
            n_rows=15, n_cols=n_cols, row_label_in=0.6, col_label_in=0.6, cell_in=0.34
        )
        try:
            cell_in = ax.get_position().width * fig.get_figwidth() / n_cols
            assert cell_in <= 0.46 + 1e-9
            assert ax.get_position().height * fig.get_figheight() < 7.0
        finally:
            plt.close(fig)


def test_the_width_floor_still_saves_a_two_column_grid_from_being_a_strip():
    """The cap is a ceiling on the cure, not a repeal of it: at the default cell size two
    columns are still widened, just not without limit."""
    fig, ax, _ = grid_canvas(n_rows=6, n_cols=2, row_label_in=0.5, col_label_in=0.5)
    try:
        assert ax.get_position().width * fig.get_figwidth() > 2 * 0.30
    finally:
        plt.close(fig)


def test_a_short_wide_grid_keeps_its_cells_square_rather_than_stretching_its_rows():
    """The floor was applied to the height as well, which raised a three-row grid to 1.8 in
    without touching its width — cells 0.35 in wide and 0.60 in tall, in a helper whose one
    promise is that they are square."""
    fig, ax, _ = grid_canvas(n_rows=3, n_cols=15, row_label_in=0.6, col_label_in=0.6)
    try:
        box = ax.get_position()
        cell_w = box.width * fig.get_figwidth() / 15
        cell_h = box.height * fig.get_figheight() / 3
        assert cell_h == pytest.approx(cell_w, rel=0.02)
    finally:
        plt.close(fig)


def test_the_grid_reserves_the_measured_gutters_and_nothing_more_without_axis_labels():
    fig, ax, _ = grid_canvas(n_rows=6, n_cols=6, row_label_in=0.75, col_label_in=0.5)
    try:
        assert ax.get_position().x0 * fig.get_figwidth() == pytest.approx(0.75 + 0.14)
    finally:
        plt.close(fig)


def test_an_axis_label_on_a_grid_is_not_printed_through_the_notes():
    """
    Pin the bug: ``axis_label_in`` defaulting to zero is right, ignoring it is not.

    The room below a grid is measured from the *tick* strings, so a caller that also sets
    ``set_xlabel`` gets it drawn in the footnote block. Observed: a sender grid printed
    "Candidate sender" across a caveat about what its colour means.
    """
    notes = [LONG_NOTE, "second"]
    fig, ax, _ = grid_canvas(
        n_rows=8, n_cols=6, row_label_in=0.6, col_label_in=0.9, notes=notes, axis_label_in=0.24
    )
    try:
        ax.set_xlabel("Candidate sender", fontsize=8.5)
        write_notes(fig, notes)
        renderer = fig.canvas.get_renderer()
        label = ax.xaxis.get_label().get_window_extent(renderer)
        for note in fig.texts:
            assert not label.overlaps(note.get_window_extent(renderer))
    finally:
        plt.close(fig)


def test_asking_for_axis_label_room_moves_both_gutters_not_only_the_bottom():
    """A ``set_ylabel`` clips against the row labels for the same reason a ``set_xlabel``
    overprints the notes, so one argument has to buy both."""
    plain = grid_canvas(n_rows=6, n_cols=6, row_label_in=0.5, col_label_in=0.5)
    labelled = grid_canvas(
        n_rows=6, n_cols=6, row_label_in=0.5, col_label_in=0.5, axis_label_in=0.24
    )
    try:
        left_gain = (
            labelled[1].get_position().x0 * labelled[0].get_figwidth()
            - plain[1].get_position().x0 * plain[0].get_figwidth()
        )
        bottom_gain = (
            labelled[1].get_position().y0 * labelled[0].get_figheight()
            - plain[1].get_position().y0 * plain[0].get_figheight()
        )
        assert left_gain == pytest.approx(0.24)
        assert bottom_gain == pytest.approx(0.24)
    finally:
        plt.close(plain[0])
        plt.close(labelled[0])


# --------------------------------------------------------------------------- #
# the stacked-track panel
# --------------------------------------------------------------------------- #


def test_the_stacked_tracks_get_the_heights_they_asked_for_in_inches():
    """A height *ratio* applied to a figure sized for its labels gives a different split
    on every figure; the point of stating inches is that the tall track stays tall."""
    fig, axes = stacked_panel_canvas(heights_in=(1.55, 0.24, 0.62), label_in=0.6, data_in=4.0)
    try:
        heights = [ax.get_position().height * fig.get_figheight() for ax in axes]
        assert heights == pytest.approx([1.55, 0.24, 0.62])
    finally:
        plt.close(fig)


def test_the_stacked_tracks_are_returned_top_first_and_do_not_overlap():
    fig, axes = stacked_panel_canvas(heights_in=(1.5, 0.3, 0.6), label_in=0.6, data_in=4.0)
    try:
        boxes = [ax.get_position() for ax in axes]
        assert boxes[0].y0 > boxes[1].y0 > boxes[2].y0
        for upper, lower in zip(boxes, boxes[1:], strict=False):
            assert lower.y1 <= upper.y0
    finally:
        plt.close(fig)


def test_the_stacked_tracks_share_one_x_extent():
    fig, axes = stacked_panel_canvas(heights_in=(1.5, 0.3), label_in=0.9, data_in=4.0)
    try:
        spans = {(round(ax.get_position().x0, 9), round(ax.get_position().x1, 9)) for ax in axes}
        assert len(spans) == 1
        assert axes[0].get_position().width * fig.get_figwidth() == pytest.approx(4.0)
    finally:
        plt.close(fig)


def test_a_stack_with_no_tracks_or_a_zero_height_track_is_refused():
    with pytest.raises(ValueError, match="at least one height"):
        stacked_panel_canvas(heights_in=(), label_in=0.5, data_in=3.0)
    with pytest.raises(ValueError, match="must be positive"):
        stacked_panel_canvas(heights_in=(1.5, 0.0), label_in=0.5, data_in=3.0)
