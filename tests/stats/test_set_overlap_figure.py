"""The overlap figures: do they carry the numbers, and the caveats, unchanged?

An overlap figure has one job a table does not — showing shape — and one way to fail
that a table cannot: it can look confident about a pair that was never tested, or rank
pairs by a colour that does not track the evidence. So what is tested here is that
every count drawn is a count from the frame, that an untested pair is visibly untested,
and that the figure and its notes cannot disagree with the table they came from.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cellquorum.stats.gene_set_overlap import exclusive_combinations, set_overlap_tests
from cellquorum.visualization.set_overlap import (
    set_overlap_matrix,
    set_overlap_notes,
    set_overlap_upset,
)

SETS = {
    "apical_junction": ["g1", "g2", "g3", "g4", "g5"],
    "emt": ["g3", "g4", "g5", "g6", "g7"],
    "ros": ["g50", "g51", "g52"],
}
UNIVERSE = [f"g{i}" for i in range(1, 101)]


def _texts(fig) -> str:
    """Every string on the figure, with line breaks flattened to spaces.

    Footnotes are wrapped to the figure's width, so a note the table yields as one
    sentence reaches the page as two or three lines. Flattening keeps the assertions
    about *what* is said rather than about where the wrap happened to fall.
    """
    joined = "\n".join(artist.get_text() for artist in fig.findobj(match=plt.Text))
    return " ".join(joined.split())


# --------------------------------------------------------------------------- #
# the UpSet plot
# --------------------------------------------------------------------------- #


def test_the_upset_draws_one_column_per_occupied_combination():
    fig = set_overlap_upset(SETS)
    try:
        # 4 occupied combinations of the 7 possible, so 4 bars.
        bars = [patch for patch in fig.axes[0].patches if patch.get_height() > 0]
        assert len(bars) == 4
        heights = sorted(patch.get_height() for patch in bars)
        # apical_junction only 2, emt only 2, ros only 3, shared 3.
        assert heights == [2.0, 2.0, 3.0, 3.0]
    finally:
        plt.close(fig)


def test_the_counts_drawn_above_the_bars_are_the_counts_in_the_table():
    table = exclusive_combinations(SETS)
    fig = set_overlap_upset(SETS)
    try:
        text = _texts(fig)
        for size in table["size"]:
            assert str(int(size)) in text
    finally:
        plt.close(fig)


def _matrix_axes(fig):
    """The UpSet's dot-matrix panel: the only one drawn with scatter, not bars.

    Not "the first axes with y-ticklabels" — the bar panel has numeric ones — and not
    "the axes with no x ticks", because the matrix shares its x-axis with the bars and
    so clearing one clears both.
    """
    for ax in fig.axes:
        if ax.collections:
            return ax
    raise AssertionError("no axes looked like the membership matrix")


def _sidebar_axes(fig):
    return next(ax for ax in fig.axes if ax.get_xlabel().startswith("Set size"))


def _row_labels(fig) -> list[str]:
    return [label.get_text() for label in _matrix_axes(fig).get_yticklabels()]


def test_every_set_gets_a_row_labelled_for_reading():
    fig = set_overlap_upset(SETS, set_labels={"emt": "EMT"})
    try:
        labels = _row_labels(fig)
        assert "EMT" in labels
        assert "apical junction" in labels  # underscores become spaces
        assert "ros" in labels
    finally:
        plt.close(fig)


def test_rows_lead_with_the_least_exclusive_set_because_that_is_the_finding():
    # `emt` and `apical_junction` each keep 2 of 5; `ros` shares nothing.
    fig = set_overlap_upset(SETS)
    try:
        assert _row_labels(fig)[-1] == "ros"
    finally:
        plt.close(fig)

    fig = set_overlap_upset(SETS, sort_sets="size")
    try:
        # Largest first, and `ros` (3 genes) is smallest, so it lands last either
        # way here — what changes is that the wholly-shared set no longer leads.
        assert _row_labels(fig)[0] in {"apical junction", "emt"}
    finally:
        plt.close(fig)


def test_a_set_wholly_contained_in_the_others_leads_and_is_marked():
    """The failure this prevents: a restatement presented as a separate readout."""
    sets = {"big": ["g1", "g2", "g3", "g4"], "subset": ["g2", "g3"], "other": ["g7", "g8"]}
    fig = set_overlap_upset(sets)
    try:
        labels = _row_labels(fig)
        assert labels[0] == "subset ‡"  # least exclusive first, and marked
        text = _texts(fig)
        assert "every member is also in another set" in text
        assert "no column of its own" in text
    finally:
        plt.close(fig)


def test_a_panel_of_independent_sets_earns_no_mark():
    fig = set_overlap_upset({"a": ["g1", "g2"], "b": ["g3", "g4"]})
    try:
        assert "‡" not in _texts(fig)
    finally:
        plt.close(fig)


def test_an_unknown_row_order_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="sort_sets must be one of"):
        set_overlap_upset(SETS, sort_sets="alphabetical")


def test_the_row_labels_get_a_gutter_wide_enough_for_them():
    """A label drawn into the sidebar's axes is painted over in its middle.

    The symptom is a figure whose row labels show only their first and last few
    characters, which reads as a rendering bug rather than as a layout one.
    """
    long_name = "protective_flow_KLF2_KLF4_and_friends"
    sets = {long_name: ["g1", "g2"], "b": ["g3"]}
    fig = set_overlap_upset(sets)
    try:
        matrix, sidebar = _matrix_axes(fig), _sidebar_axes(fig)
        renderer = fig.canvas.get_renderer()
        label = matrix.get_yticklabels()[0]
        left_edge = label.get_window_extent(renderer).x0
        assert (
            left_edge > sidebar.get_window_extent().x1
        ), "the row label starts inside the sidebar axes, which will paint over it"
    finally:
        plt.close(fig)


def test_the_sidebar_carries_no_tick_marks_of_its_own():
    """The house axis style resets tick length, so hiding a tick before it is styled
    puts the tick back — a row of orphan dashes floating left of the sidebar."""
    fig = set_overlap_upset(SETS)
    try:
        sidebar = _sidebar_axes(fig)
        assert all(tick.tick1line.get_markersize() == 0 for tick in sidebar.yaxis.get_major_ticks())
    finally:
        plt.close(fig)


def test_a_truncated_upset_says_how_much_it_is_not_showing():
    """Silent truncation would make a shared panel look like an independent one."""
    fig = set_overlap_upset(SETS, max_combinations=2)
    try:
        text = _texts(fig)
        assert "2 further combination" in text
        # The two dropped columns held 3 + 2 genes, whichever order they sorted in.
        assert "combination(s) holding 5 genes" in text
    finally:
        plt.close(fig)


def test_the_universe_and_what_fell_outside_it_are_stated_on_the_figure():
    sets = {"a": ["g1", "g2", "Mm.Actb"], "b": ["g2", "g3"]}
    fig = set_overlap_upset(sets, universe=UNIVERSE)
    try:
        text = _texts(fig)
        assert "100 genes in the universe" in text
        assert "a 1" in text
        assert "gene-naming mismatch" in text
    finally:
        plt.close(fig)


def test_a_min_size_that_removes_everything_raises_rather_than_drawing_nothing():
    with pytest.raises(ValueError, match="nothing to draw"):
        set_overlap_upset(SETS, min_size=99)


# --------------------------------------------------------------------------- #
# the similarity matrix
# --------------------------------------------------------------------------- #


def test_the_matrix_colours_the_lower_triangle_and_leaves_the_diagonal_empty():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    fig = set_overlap_matrix(overlaps)
    try:
        image = fig.axes[0].images[0]
        data = image.get_array()
        assert data.shape == (3, 3)
        # A set overlaps itself perfectly; drawing that would own the colour scale.
        assert all(np.ma.is_masked(data[i, i]) for i in range(3))
        # Upper triangle is the same pair written twice, so only one is drawn.
        assert np.ma.is_masked(data[0, 1])
        assert not np.ma.is_masked(data[1, 0])
        assert data[1, 0] == pytest.approx(3 / 7)  # apical_junction/emt Jaccard
    finally:
        plt.close(fig)


def test_a_significant_pair_is_starred_and_a_chance_pair_is_bare():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    fig = set_overlap_matrix(overlaps)
    try:
        # One starred cell: the real overlap. The two disjoint pairs sit at FDR 1.
        stars = [
            artist.get_text() for artist in fig.axes[0].texts if set(artist.get_text()) == {"*"}
        ]
        # ** not ***: the marks read the FDR, and p = 6.0e-4 over three pairs is
        # 1.8e-3. A figure marked from the uncorrected p-value would say ***.
        assert stars == ["**"]
        assert float(overlaps["fdr"].min()) == pytest.approx(1.798e-3, rel=1e-2)
    finally:
        plt.close(fig)


def test_each_drawn_cell_prints_the_count_of_shared_elements():
    """The count is checkable in a way a Jaccard is not, and it survives a flat scale."""
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    fig = set_overlap_matrix(overlaps)
    try:
        printed = [artist.get_text() for artist in fig.axes[0].texts]
        assert printed.count("3") == 1  # apical_junction/emt share 3 genes
        # A pair sharing nothing gets no number — a printed 0 would compete with the
        # counts that matter, and the pale cell already says it.
        assert "0" not in printed
    finally:
        plt.close(fig)


def test_an_untested_pair_is_marked_and_never_left_looking_like_a_null():
    """A blank cell reads as "no overlap"; an unattempted test must not read that way."""
    sets = {"a": ["g1", "g2"], "b": ["Hs.NOTAGENE"], "c": ["g2", "g3"]}
    overlaps = set_overlap_tests(sets, universe=UNIVERSE)
    fig = set_overlap_matrix(overlaps)
    try:
        marks = [artist.get_text() for artist in fig.axes[0].texts]
        assert marks.count("?") == 2  # b against a, and b against c
        text = _texts(fig)
        assert "Marked ? and not tested" in text
        assert "check the gene naming" in text
    finally:
        plt.close(fig)


def test_the_matrix_can_colour_by_any_column_and_says_which_in_the_colourbar():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    fig = set_overlap_matrix(overlaps, value="fold_enrichment")
    try:
        assert "Fold enrichment" in _texts(fig)
    finally:
        plt.close(fig)


def test_colouring_by_a_column_that_is_not_there_names_the_ones_that_are():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    with pytest.raises(ValueError, match="not a column"):
        set_overlap_matrix(overlaps, value="similarity")


def test_an_empty_table_raises():
    import pandas as pd

    from cellquorum.stats.gene_set_overlap import OVERLAP_COLUMNS

    with pytest.raises(ValueError, match="empty"):
        set_overlap_matrix(pd.DataFrame(columns=list(OVERLAP_COLUMNS)))


# --------------------------------------------------------------------------- #
# the notes are the table's, not the drawing's
# --------------------------------------------------------------------------- #


def test_the_notes_state_the_universe_because_the_p_values_depend_on_it():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    joined = "\n".join(set_overlap_notes(overlaps))
    assert "100 testable genes" in joined
    assert "Benjamini" in joined


def test_notes_refuse_to_quote_one_universe_when_the_pairs_used_two():
    """Concatenating two runs' tables is the way this happens, and it invalidates the FDR."""
    import pandas as pd

    wide = set_overlap_tests(SETS, universe=UNIVERSE)
    narrow = set_overlap_tests(SETS, universe=[f"g{i}" for i in range(1, 11)])
    joined = "\n".join(set_overlap_notes(pd.concat([wide, narrow], ignore_index=True)))
    assert "different universes" in joined
    assert "not over one family" in joined


def test_dropped_members_are_footnoted_once_not_per_pair():
    sets = {"a": ["g1", "g2", "Mm.Actb"], "b": ["g2", "g3"], "c": ["g4", "Mm.Vcl"]}
    notes = set_overlap_notes(set_overlap_tests(sets, universe=UNIVERSE))
    outside = [note for note in notes if "outside the universe" in note]
    assert len(outside) == 1


def test_the_notes_a_panel_draws_are_the_notes_the_table_yields():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    notes = set_overlap_notes(overlaps)
    fig = set_overlap_matrix(overlaps)
    try:
        drawn = _texts(fig)
        for note in notes:
            assert note in drawn
    finally:
        plt.close(fig)


def test_a_panel_drawn_into_a_given_axes_leaves_the_notes_to_its_caller():
    overlaps = set_overlap_tests(SETS, universe=UNIVERSE)
    fig, ax = plt.subplots()
    try:
        set_overlap_matrix(overlaps, ax=ax, footnotes=False, title="LEC")
        assert "hypergeometric" not in _texts(fig)
        assert ax.get_title(loc="left") == "LEC"
        # ...and the caller can still get them, verbatim.
        assert any("hypergeometric" in note for note in set_overlap_notes(overlaps))
    finally:
        plt.close(fig)


def test_the_figure_survives_a_csv_round_trip(tmp_path):
    """Figures are drawn from the written CSV, where "" has become NaN."""
    import pandas as pd

    sets = {"a": ["g1", "g2"], "b": ["Hs.NOTAGENE"], "c": ["g2", "g3"]}
    path = tmp_path / "overlaps.csv"
    set_overlap_tests(sets, universe=UNIVERSE).to_csv(path, index=False)
    fig = set_overlap_matrix(pd.read_csv(path))
    try:
        assert "Marked ? and not tested" in _texts(fig)
    finally:
        plt.close(fig)
