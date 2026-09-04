"""Program-correlation figures: does the panel say what the frame says?

The failure these figures exist to prevent is a heatmap that is read as evidence when
its n is wrong, its coefficient is really the condition contrast, or its two programs
share genes. So the tests do not check that a figure was produced — they check that each
of those three facts reaches the page, and that the marks disagree with each other when
the numbers do.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.program_correlation import program_correlation_tests
from cellquorum.visualization.program_correlation import (
    program_correlation_heatmap,
    program_correlation_notes,
    program_correlation_slopes,
)

RNG = np.random.default_rng(1337)


def _tests_frame(**overrides) -> pd.DataFrame:
    """A minimal but complete frame, so a test can perturb one column at a time."""
    base = pd.DataFrame(
        {
            "program_a": ["endomt", "endomt", "junctions"],
            "program_b": ["junctions", "flow", "flow"],
            "unit": ["donor"] * 3,
            "n_units": [9] * 3,
            "method": ["spearman"] * 3,
            "r": [0.83, -0.41, 0.12],
            "p_value": [0.0001, 0.27, 0.75],
            "fdr": [0.0004, 0.40, 0.75],
            "r_adjusted": [0.78, -0.15, 0.05],
            "p_adjusted": [0.004, 0.70, 0.90],
            "fdr_adjusted": [0.012, 0.90, 0.90],
            "shared_genes": [7, 0, 0],
            "shares_genes": [True, False, False],
            "reason": ["", "", ""],
        }
    )
    for column, values in overrides.items():
        base[column] = values
    return base


def _figure_text(fig) -> str:
    """Every string on the figure — footnotes, cell marks, labels — as one blob.

    Line breaks are flattened to spaces because the footnotes are wrapped to the
    figure's width, so a sentence the notes yield whole reaches the page in pieces.
    """
    pieces = [text.get_text() for text in fig.texts]
    for ax in fig.axes:
        pieces.extend(text.get_text() for text in ax.texts)
        pieces.append(ax.get_xlabel())
        pieces.append(ax.get_ylabel())
        pieces.append(ax.get_title())
        pieces.extend(label.get_text() for label in ax.get_yticklabels())
        pieces.extend(label.get_text() for label in ax.get_xticklabels())
        legend = ax.get_legend()
        if legend is not None:
            pieces.extend(text.get_text() for text in legend.get_texts())
    return " ".join("\n".join(pieces).split())


def _matrix_axes(fig):
    """The heatmap panel: the one drawn with imshow, not the colourbar."""
    for ax in fig.axes:
        if ax.images and ax.get_yticklabels():
            return ax
    raise AssertionError("no axes looked like the correlation matrix")


# --------------------------------------------------------------------------- #
# the notes carry the three facts a coefficient cannot carry
# --------------------------------------------------------------------------- #


def test_the_unit_and_its_count_reach_the_notes():
    notes = program_correlation_notes(_tests_frame())
    assert "9 donors" in notes[0]
    assert "Spearman" in notes[0]


def test_the_unit_reads_as_a_noun_not_as_the_obs_column_name():
    """The unit arrives as a column key. "12 sample_ids" is an identifier where the
    sentence wants a thing counted; the naming convention is not part of the noun."""
    notes = program_correlation_notes(_tests_frame(unit=["sample_id"] * 3, n_units=[12] * 3))
    assert "12 samples" in notes[0]
    assert "sample_id" not in notes[0]


def test_the_notes_state_the_test_is_two_sided():
    """Two-sided, unlike the one-sided overlap test: opposite movement is a finding."""
    assert "two-sided" in program_correlation_notes(_tests_frame())[0]


def test_a_row_level_unit_is_called_out_as_pseudoreplication():
    frame = _tests_frame(unit=["row"] * 3, n_units=[2144] * 3)
    joined = "\n".join(program_correlation_notes(frame))
    assert "anticonservative" in joined
    assert "independent" in joined


def test_a_sample_level_unit_gets_no_pseudoreplication_warning():
    joined = "\n".join(program_correlation_notes(_tests_frame()))
    assert "anticonservative" not in joined


def test_shared_gene_counts_are_named_in_the_notes_not_just_marked():
    joined = "\n".join(program_correlation_notes(_tests_frame()))
    assert "endomt/junctions (7)" in joined


def test_the_notes_name_pairs_the_way_the_axes_label_them():
    """One vocabulary per figure. A footnote reading ``endomt_lec/mesenchymal gain`` under
    an axis reading ``EndoMT (LEC)`` makes the reader translate to find the pair the mark
    refers to, which is the whole job the footnote was there to do."""
    labels = {"endomt": "EndoMT (LEC)", "junctions": "Core junctions"}
    joined = "\n".join(program_correlation_notes(_tests_frame(), program_labels=labels))
    assert "EndoMT (LEC)/Core junctions (7)" in joined
    assert "endomt/junctions" not in joined


def test_a_figures_notes_use_the_same_labels_as_its_axes():
    labels = {"endomt": "EndoMT (LEC)", "junctions": "Core junctions", "flow": "Protective flow"}
    for figure in (
        program_correlation_heatmap(_tests_frame(), program_labels=labels),
        program_correlation_slopes(_tests_frame(), program_labels=labels),
    ):
        text = _figure_text(figure)
        assert "EndoMT (LEC)/Core junctions" in text
        assert "endomt" not in text
        plt.close(figure)


def test_unsupplied_gene_lists_are_reported_as_unknown_not_as_zero():
    frame = _tests_frame(shared_genes=[-1, -1, -1], shares_genes=[False] * 3)
    joined = "\n".join(program_correlation_notes(frame))
    assert "unknown" in joined
    assert "by construction" in joined


def test_the_notes_name_what_was_removed_rather_than_asserting_it_was_the_condition():
    """The condition is not the only common cause a study has to remove — depth raises
    every score at once. A panel that says "condition removed" when depth was also
    removed is describing an analysis that was not run."""
    frame = _tests_frame(
        adjusted_for=["condition, log1p_total_counts"] * 3,
        fdr=[0.001, 0.40, 0.75],
        fdr_adjusted=[0.30, 0.90, 0.90],
    )
    joined = "\n".join(program_correlation_notes(frame))
    assert "condition and log1p total counts" in joined


def test_a_frame_with_no_adjustment_record_still_reads_as_the_condition():
    """Back-compatibility with a table written before the covariates existed: the
    condition is what such a frame's upper triangle removed."""
    fig = program_correlation_heatmap(_tests_frame())
    try:
        assert "with the condition removed" in _figure_text(fig)
    finally:
        plt.close(fig)


def test_an_adjustment_over_fewer_units_than_the_measurement_is_disclosed():
    """Two coefficients computed over different numbers of units is the thing a reader
    would never guess from a matrix, so it cannot stay in the CSV alone."""
    frame = _tests_frame(n_units=[18] * 3, n_units_adjusted=[16, 16, 17])
    joined = "\n".join(program_correlation_notes(frame))
    assert "16" in joined
    assert "18" in joined


def test_an_adjustment_over_every_unit_is_not_footnoted_as_a_shortfall():
    frame = _tests_frame(n_units=[18] * 3, n_units_adjusted=[18] * 3)
    assert "missing a value" not in "\n".join(program_correlation_notes(frame))


def test_a_pair_that_loses_its_call_under_adjustment_is_named():
    """The whole correction: significant pooled, not significant within the arms."""
    frame = _tests_frame(fdr=[0.001, 0.40, 0.75], fdr_adjusted=[0.30, 0.90, 0.90])
    joined = "\n".join(program_correlation_notes(frame))
    assert "common cause" in joined
    assert "endomt/junctions" in joined


def test_a_pair_that_keeps_its_call_is_not_named_as_having_lost_it():
    joined = "\n".join(program_correlation_notes(_tests_frame()))
    assert "common cause" not in joined


def test_a_missing_adjustment_is_not_a_lost_call():
    """No condition supplied is not the same claim as "the condition explains it"."""
    frame = _tests_frame(
        r_adjusted=[np.nan] * 3, p_adjusted=[np.nan] * 3, fdr_adjusted=[np.nan] * 3
    )
    assert "common cause" not in "\n".join(program_correlation_notes(frame))


def test_an_untested_pair_is_named_with_its_reason():
    frame = _tests_frame(
        p_value=[0.0001, 0.27, np.nan],
        fdr=[0.0004, 0.40, np.nan],
        reason=["", "", "3 donor(s) is too few to test"],
    )
    joined = "\n".join(program_correlation_notes(frame))
    assert "not tested" in joined
    assert "too few to test" in joined


def test_the_family_size_counts_tested_pairs_not_all_pairs():
    frame = _tests_frame(p_value=[0.0001, 0.27, np.nan], fdr=[0.0004, 0.40, np.nan])
    assert "2 of 3 testable pair(s)" in program_correlation_notes(frame)[0]


# --------------------------------------------------------------------------- #
# the heatmap
# --------------------------------------------------------------------------- #


def test_the_two_triangles_carry_different_quantities():
    fig = program_correlation_heatmap(_tests_frame())
    try:
        matrix = _matrix_axes(fig).images[0].get_array()
        order = [label.get_text() for label in _matrix_axes(fig).get_yticklabels()]
        i, j = order.index("endomt"), order.index("junctions")
        assert matrix[max(i, j), min(i, j)] == pytest.approx(0.83)  # measured, below
        assert matrix[min(i, j), max(i, j)] == pytest.approx(0.78)  # adjusted, above
    finally:
        plt.close(fig)


def test_the_diagonal_is_left_empty_rather_than_drawn_at_one():
    """A set correlates with itself at 1 and would own the end of the colour scale."""
    fig = program_correlation_heatmap(_tests_frame())
    try:
        matrix = _matrix_axes(fig).images[0].get_array()
        assert all(matrix.mask[k, k] for k in range(matrix.shape[0]))
    finally:
        plt.close(fig)


def test_the_colour_scale_is_the_full_correlation_range_not_the_observed_one():
    """Autoscaling to the data makes r = 0.3 look like r = 1 on a panel with no strong
    pair, and makes two arms of the same study incomparable."""
    fig = program_correlation_heatmap(_tests_frame(r=[0.30, 0.10, 0.05]))
    try:
        image = _matrix_axes(fig).images[0]
        assert (image.norm.vmin, image.norm.vmax) == (-1.0, 1.0)
    finally:
        plt.close(fig)


def test_a_shared_gene_pair_is_marked_in_its_cell():
    fig = program_correlation_heatmap(_tests_frame())
    try:
        assert "‡" in _figure_text(fig)
    finally:
        plt.close(fig)


def test_a_null_cell_is_left_blank_rather_than_labelled_ns():
    """Fifty-five cells reading "ns" hide the handful that carry a result."""
    fig = program_correlation_heatmap(_tests_frame())
    try:
        cell_marks = [text.get_text() for text in _matrix_axes(fig).texts]
        assert "ns" not in cell_marks
        # Two significant cells out of six occupied ones, and nothing drawn in the four
        # null cells at all.
        assert sorted(cell_marks) == ["*", "***‡"]
    finally:
        plt.close(fig)


def test_an_untested_cell_is_marked_distinctly_from_a_null_one():
    frame = _tests_frame(
        p_value=[0.0001, 0.27, np.nan],
        fdr=[0.0004, 0.40, np.nan],
        reason=["", "", "3 donor(s) is too few to test"],
    )
    fig = program_correlation_heatmap(frame)
    try:
        assert "?" in [text.get_text() for text in _matrix_axes(fig).texts]
    finally:
        plt.close(fig)


def test_without_an_adjustment_the_upper_triangle_is_empty_and_says_why():
    frame = _tests_frame(
        r_adjusted=[np.nan] * 3, p_adjusted=[np.nan] * 3, fdr_adjusted=[np.nan] * 3
    )
    fig = program_correlation_heatmap(frame)
    try:
        matrix = _matrix_axes(fig).images[0].get_array()
        upper = [matrix.mask[i, j] for i in range(3) for j in range(3) if j > i]
        assert all(upper), "the lower triangle was mirrored instead of left empty"
        assert "no condition was supplied" in _figure_text(fig)
    finally:
        plt.close(fig)


def test_the_cell_label_flips_to_white_on_a_dark_cell():
    """Ink on a saturated cell of a diverging map is unreadable in print."""
    fig = program_correlation_heatmap(_tests_frame())
    try:
        by_text = {text.get_text(): text for text in _matrix_axes(fig).texts}
        assert by_text["***‡"].get_color() == "white"  # |r| = 0.83
    finally:
        plt.close(fig)


def test_the_axes_are_square_on_the_page_so_no_dead_band_is_left():
    """``aspect="equal"`` is applied after layout; an axes that is not already square
    gets shrunk and re-centred, leaving a band of empty figure under the labels."""
    fig = program_correlation_heatmap(_tests_frame())
    try:
        box = _matrix_axes(fig).get_position()
        width_in = box.width * fig.get_figwidth()
        height_in = box.height * fig.get_figheight()
        assert width_in == pytest.approx(height_in, abs=0.02)
    finally:
        plt.close(fig)


def test_the_row_labels_get_a_gutter_wide_enough_for_them():
    frame = _tests_frame(
        program_a=["a_very_long_program_name_indeed"] * 2 + ["short"],
        program_b=["short", "middling_name", "middling_name"],
    )
    fig = program_correlation_heatmap(frame)
    try:
        renderer = fig.canvas.get_renderer()
        matrix = _matrix_axes(fig)
        left_edge = matrix.get_position().x0 * fig.get_figwidth() * fig.dpi
        for label in matrix.get_yticklabels():
            assert label.get_window_extent(renderer).x0 >= -1.0
            assert label.get_window_extent(renderer).x1 <= left_edge + 1.0
    finally:
        plt.close(fig)


def test_the_program_order_can_be_pinned_to_match_the_panels_beside_it():
    order = ["flow", "junctions", "endomt"]
    fig = program_correlation_heatmap(_tests_frame(), program_order=order)
    try:
        assert [label.get_text() for label in _matrix_axes(fig).get_yticklabels()] == order
    finally:
        plt.close(fig)


def test_display_labels_replace_the_program_names_everywhere():
    fig = program_correlation_heatmap(_tests_frame(), program_labels={"endomt": "EndoMT (LEC)"})
    try:
        text = _figure_text(fig)
        assert "EndoMT (LEC)" in text
    finally:
        plt.close(fig)


def test_a_frame_without_the_defining_columns_is_refused():
    with pytest.raises(ValueError, match="program_correlation_tests"):
        program_correlation_heatmap(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_an_empty_frame_is_refused_rather_than_drawn_blank():
    with pytest.raises(ValueError, match="no pair"):
        program_correlation_heatmap(_tests_frame().iloc[:0])


def test_a_long_footnote_wraps_instead_of_stretching_the_figure():
    """Notes are saved with ``bbox_inches="tight"``, so a wide note is not clipped —
    it widens the saved figure and leaves the panel in a band of empty page. The
    figure's aspect ratio must not be a function of how many words the caveats needed.
    """
    fig = program_correlation_heatmap(_tests_frame())
    try:
        renderer = fig.canvas.get_renderer()
        limit = fig.get_figwidth() * fig.dpi
        for note in fig.texts:
            assert note.get_window_extent(renderer).x1 <= limit + 1.0, note.get_text()
    finally:
        plt.close(fig)


def test_the_room_reserved_for_the_notes_is_the_room_they_take():
    """Counting notes rather than wrapped lines reserves one line for a three-line
    caveat, and the bottom note is then drawn over the column labels."""
    fig = program_correlation_heatmap(_tests_frame())
    try:
        renderer = fig.canvas.get_renderer()
        tops = [note.get_window_extent(renderer).y1 for note in fig.texts]
        axes_bottom = _matrix_axes(fig).get_position().y0 * fig.get_figheight() * fig.dpi
        # Every note line sits below the panel, including the topmost one.
        assert max(tops) <= axes_bottom + 1.0
    finally:
        plt.close(fig)


def test_turning_the_footnotes_off_keeps_the_one_that_reads_the_panel():
    """``footnotes=False`` drops the statistical caveats, which a caption may repeat.
    It cannot drop the sentence saying which triangle is which — without it the two
    halves of the matrix are indistinguishable, and the figure means nothing.
    """
    fig = program_correlation_heatmap(_tests_frame(), footnotes=False)
    try:
        text = _figure_text(fig)
        assert "Below the diagonal" in text
        assert "Benjamini" not in text
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# the slope figure
# --------------------------------------------------------------------------- #


def test_every_pair_gets_a_row_labelled_with_both_programs():
    fig = program_correlation_slopes(_tests_frame())
    try:
        labels = [
            label.get_text()
            for ax in fig.axes
            for label in ax.get_yticklabels()
            if label.get_text()
        ]
        assert any("endomt/junctions" in label for label in labels)
        assert len(labels) == 3
    finally:
        plt.close(fig)


def test_pairs_are_ordered_by_absolute_coefficient_so_the_finding_is_at_the_top():
    fig = program_correlation_slopes(_tests_frame())
    try:
        labels = [label.get_text() for ax in fig.axes for label in ax.get_yticklabels()]
        assert "endomt/junctions" in labels[0]  # |0.83|
        assert "junctions/flow" in labels[-1]  # |0.12|
    finally:
        plt.close(fig)


def test_the_x_axis_spans_the_whole_correlation_range():
    fig = program_correlation_slopes(_tests_frame(r=[0.3, 0.1, 0.05]))
    try:
        ax = next(ax for ax in fig.axes if ax.get_xlabel())
        assert ax.get_xlim()[0] <= -1.0
        assert ax.get_xlim()[1] >= 1.0
    finally:
        plt.close(fig)


def test_a_pair_with_no_adjustment_gets_one_dot_and_no_segment():
    """Drawing a segment to nowhere would imply the coefficient collapsed to zero."""
    frame = _tests_frame(
        r_adjusted=[0.78, np.nan, np.nan],
        p_adjusted=[0.004, np.nan, np.nan],
        fdr_adjusted=[0.012, np.nan, np.nan],
    )
    fig = program_correlation_slopes(frame)
    try:
        ax = next(ax for ax in fig.axes if ax.get_xlabel())
        # Horizontal two-point lines only: the zero rule is also a two-point Line2D.
        segments = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(segments) == 1
    finally:
        plt.close(fig)


def test_truncation_is_footnoted_rather_than_silent():
    fig = program_correlation_slopes(_tests_frame(), max_pairs=2)
    try:
        text = _figure_text(fig)
        assert "1 further pair(s)" in text
        labels = [
            label.get_text()
            for ax in fig.axes
            for label in ax.get_yticklabels()
            if label.get_text()
        ]
        assert len(labels) == 2
    finally:
        plt.close(fig)


def test_the_notes_clear_the_x_axis_instead_of_being_printed_through_it():
    """Folding the axis-label room into the axes height rather than reserving it below
    stacks the notes from the figure's bottom edge and prints the tick labels over them.
    """
    fig = program_correlation_slopes(_tests_frame())
    try:
        # The axis label's position is solved from the tick labels at draw time, so
        # before a draw it still reports its unplaced default.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = next(ax for ax in fig.axes if ax.get_xlabel())
        axis_bottom = min(
            [ax.xaxis.get_label().get_window_extent(renderer).y0]
            + [label.get_window_extent(renderer).y0 for label in ax.get_xticklabels()]
        )
        assert fig.texts, "no notes were drawn"
        assert max(note.get_window_extent(renderer).y1 for note in fig.texts) <= axis_bottom + 1.0
    finally:
        plt.close(fig)


def test_the_legend_names_which_dot_is_which():
    fig = program_correlation_slopes(_tests_frame())
    try:
        text = _figure_text(fig)
        assert "As measured" in text
        assert "Condition removed" in text
    finally:
        plt.close(fig)


def test_the_legend_names_the_covariates_too_when_the_frame_records_them():
    fig = program_correlation_slopes(
        _tests_frame(adjusted_for=["condition, log1p_total_counts"] * 3)
    )
    try:
        assert "Condition and log1p total counts removed" in _figure_text(fig)
    finally:
        plt.close(fig)


def test_a_frame_with_no_finite_coefficient_is_refused():
    frame = _tests_frame(r=[np.nan] * 3)
    with pytest.raises(ValueError, match="nothing to draw"):
        program_correlation_slopes(frame)


# --------------------------------------------------------------------------- #
# the figures and the primitive agree
# --------------------------------------------------------------------------- #


def test_both_figures_draw_the_frame_the_primitive_actually_writes():
    """The contract test: no column the figures read is absent from the real output."""
    scores = pd.DataFrame(RNG.normal(size=(240, 4)), columns=["w", "x", "y", "z"])
    metadata = pd.DataFrame(
        {
            "donor": np.repeat([f"D{i}" for i in range(8)], 30),
            "condition": np.repeat(["Normal"] * 4 + ["Disease"] * 4, 30),
        }
    )
    table = program_correlation_tests(
        scores,
        metadata,
        sample_col="donor",
        condition_col="condition",
        program_genes={"w": ["A", "B"], "x": ["B", "C"], "y": ["D"], "z": ["E"]},
    )
    heatmap = program_correlation_heatmap(table)
    slopes = program_correlation_slopes(table)
    try:
        assert "8 donors" in _figure_text(heatmap)
        assert "8 donors" in _figure_text(slopes)
    finally:
        plt.close(heatmap)
        plt.close(slopes)
