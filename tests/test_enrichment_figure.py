"""GSEA figures: does the panel say what the table says?

A figure of a GSEA result is only worth drawing if it can be checked against the numbers
beside it, so these tests are mostly about agreement rather than about pixels: the shaded
leading edge holds the number of genes the table reports, the dot area is the leading-edge
size at a scale that does not move between figures, a floored p-value is marked as floored,
and a truncated panel says so.

The fixtures are small and hand-checkable for the same reason the overlap tests' are: the
arithmetic here is countable, and a test that cannot be verified by counting is not testing
the arithmetic.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellquorum.visualization.enrichment import (
    gsea_arm_comparison,
    gsea_dotplot,
    gsea_notes,
    gsea_running_es,
)

PERMUTATIONS = 1000
FLOOR = 1.0 / (PERMUTATIONS + 1)


def _table() -> pd.DataFrame:
    """Four pathways: two up, two down; one at the p floor; one not significant."""
    return pd.DataFrame(
        {
            "source": ["UP_STRONG", "UP_WEAK", "DOWN_NS", "DOWN_FLOORED"],
            "score": [1.9, 1.2, -1.1, -2.4],
            "es": [0.44, 0.31, -0.28, -0.66],
            "pvalue": [0.002, 0.03, 0.4, FLOOR],
            "padj": [0.004, 0.04, 0.5, 0.0002],
            "significant": [True, True, False, True],
            "p_at_resolution_limit": [False, False, False, True],
            "p_resolution_limit": [FLOOR] * 4,
            "permutations": [PERMUTATIONS] * 4,
            "collection": ["hallmark"] * 4,
            "set_size": [120, 40, 60, 171],
            "leading_edge_size": [30, 8, 12, 65],
            "leading_edge": ["A;B", "C", "D", "E;F;G"],
        }
    )


def _walk(*, n: int = 400, hits: tuple[int, ...] = (5, 9, 14, 20, 31), up: bool = True):
    """One synthetic walk whose peak is placed where the hits are, so the ES is knowable."""
    rank = np.arange(1, n + 1)
    hit = np.isin(rank, hits)
    n_h = int(hit.sum())
    metric = np.linspace(3.0, -3.0, n)
    p_hit = np.where(hit, np.abs(metric) / np.abs(metric[hit]).sum(), 0.0).cumsum()
    p_miss = np.where(~hit, 1.0 / (n - n_h), 0.0).cumsum()
    running = p_hit - p_miss
    if not up:
        rank = rank[::-1]
        running = -running
        order = np.argsort(rank)
        rank, running, hit, metric = rank[order], running[order], hit[order], -metric[::-1][order]
    return pd.DataFrame(
        {
            "source": "UP_STRONG" if up else "DOWN_FLOORED",
            "rank": rank,
            "running_es": running,
            "hit": hit.astype(int),
            "metric": metric,
        }
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rows(ax) -> list[str]:
    return [text.get_text() for text in ax.get_yticklabels()]


def _marks(ax) -> list[str]:
    return [child.get_text() for child in ax.texts]


def _notes(fig) -> str:
    """Every note on the figure as one whitespace-normalised string.

    Normalised because :func:`write_notes` wraps to the figure width and gives the
    continuation lines a hanging indent, so a sentence a test looks for is split by
    however many spaces the wrap happened to insert.
    """
    return " ".join(" ".join(text.get_text() for text in fig.texts).split())


# --------------------------------------------------------------------------- #
# the dotplot: order, direction, and what the dot means
# --------------------------------------------------------------------------- #


def test_the_rows_read_by_signed_score_so_the_two_directions_separate():
    """Not by p-value: at a high permutation count the strongest pathways share one
    floored p, and ordering on it would order the headline result arbitrarily."""
    ax = gsea_dotplot(_table()).axes[0]
    assert _rows(ax) == ["UP STRONG", "UP WEAK", "DOWN NS", "DOWN FLOORED"]


def test_a_dot_is_filled_when_significant_and_open_when_not():
    ax = gsea_dotplot(_table()).axes[0]
    faces = {
        float(c.get_offsets()[0][1]): tuple(round(v, 3) for v in c.get_facecolor()[0][:3])
        for c in ax.collections
        if len(c.get_offsets()) == 1
    }
    # Row 2 is DOWN_NS, drawn open — a white face with a coloured edge.
    assert faces[2.0] == (1.0, 1.0, 1.0)
    assert all(face != (1.0, 1.0, 1.0) for row, face in faces.items() if row != 2.0)


def test_the_dot_area_is_the_leading_edge_size_at_a_scale_that_does_not_move():
    """The invariant: a 30-gene edge is the same dot in every figure this draws, so a
    pathway does not look like a bigger finding in the panel that held weaker ones."""
    table = _table()
    full = gsea_dotplot(table).axes[0]
    subset = gsea_dotplot(table.iloc[:2]).axes[0]

    def area(ax, row: float) -> float:
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if len(offsets) == 1 and float(offsets[0][1]) == row:
                return float(collection.get_sizes()[0])
        raise AssertionError(f"no dot on row {row}")

    # UP_STRONG is row 0 in both panels, and its 30 genes must draw the same dot.
    assert area(full, 0.0) == pytest.approx(area(subset, 0.0))
    # And the area is proportional to the count, not to its square root: area is what the
    # eye integrates, so 30 genes against 8 must be 30/8 of the ink.
    assert area(full, 0.0) / area(full, 1.0) == pytest.approx(30 / 8)


def test_a_bigger_leading_edge_draws_a_bigger_dot():
    ax = gsea_dotplot(_table()).axes[0]
    by_row = {}
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets) == 1:
            by_row[float(offsets[0][1])] = float(collection.get_sizes()[0])
    # 65 genes (row 3) > 30 (row 0) > 12 (row 2) > 8 (row 1).
    assert by_row[3.0] > by_row[0.0] > by_row[2.0] > by_row[1.0]


def test_up_and_down_are_drawn_in_different_colours_and_the_axis_says_which_is_which():
    fig = gsea_dotplot(_table(), case_label="Lymphedema", control_label="Normal")
    ax = fig.axes[0]
    assert "Lymphedema" in ax.get_xlabel()
    assert "Normal" in ax.get_xlabel()
    colors = {}
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets) == 1:
            colors[float(offsets[0][1])] = tuple(collection.get_edgecolor()[0][:3])
    assert colors[0.0] == colors[1.0]  # both up
    assert colors[2.0] == colors[3.0]  # both down
    assert colors[0.0] != colors[3.0]


def test_the_x_axis_is_symmetric_so_a_positive_and_negative_score_are_comparable():
    ax = gsea_dotplot(_table()).axes[0]
    low, high = ax.get_xlim()
    assert low == pytest.approx(-high)
    assert high > 2.4  # the largest |score| fits inside


# --------------------------------------------------------------------------- #
# significance, and the difference between measured and floored
# --------------------------------------------------------------------------- #


def test_a_floored_p_value_is_marked_beside_its_stars_and_explained_in_the_notes():
    fig = gsea_dotplot(_table())
    assert any(mark.endswith("†") for mark in _marks(fig.axes[0]))
    notes = " ".join(gsea_notes(_table()))
    assert "resolution limit" in notes
    assert "1,000 permutations" in notes
    assert "bound, not a measurement" in notes


def test_a_table_with_no_floored_row_says_nothing_about_the_limit():
    table = _table()
    table["p_at_resolution_limit"] = False
    assert not [note for note in gsea_notes(table) if "resolution limit" in note]


def test_a_boolean_column_that_came_back_off_disk_as_text_is_still_read_correctly():
    """``bool("False")`` is ``True``, so a CSV round-trip would otherwise mark every row."""
    table = _table()
    table["p_at_resolution_limit"] = ["False", "False", "False", "True"]
    assert "1 pathway(s)" in " ".join(gsea_notes(table))


def test_a_non_significant_row_gets_no_star_rather_than_the_word_ns():
    """``ns`` printed beside every null row turns the gutter into a column of noise."""
    marks = _marks(gsea_dotplot(_table()).axes[0])
    assert not any(mark.startswith("ns") for mark in marks)
    assert sum(mark.count("*") > 0 for mark in marks) == 3


# --------------------------------------------------------------------------- #
# selection, truncation, and the refusals
# --------------------------------------------------------------------------- #


def test_truncation_keeps_the_strongest_and_is_always_footnoted():
    fig = gsea_dotplot(_table(), max_sets=2)
    # |score|: 2.4 and 1.9 are the two strongest, and they point opposite ways — so a
    # panel truncated by strength must still be ordered by sign.
    assert _rows(fig.axes[0]) == ["UP STRONG", "DOWN FLOORED"]
    assert "2 further pathway(s)" in " ".join(gsea_notes(_table(), dropped=2))


def test_significant_only_drops_the_null_rows_before_selecting():
    ax = gsea_dotplot(_table(), significant_only=True).axes[0]
    assert "DOWN NS" not in _rows(ax)
    assert len(_rows(ax)) == 3


def test_display_names_are_used_where_they_are_given_and_underscores_elsewhere():
    ax = gsea_dotplot(_table(), set_labels={"UP_STRONG": "Actomyosin (custom)"}).axes[0]
    assert _rows(ax)[0] == "Actomyosin (custom)"
    assert _rows(ax)[1] == "UP WEAK"


def test_a_table_with_no_finite_score_is_refused_rather_than_drawn_empty():
    table = _table()
    table["score"] = np.nan
    with pytest.raises(ValueError, match="no pathway has a finite score"):
        gsea_dotplot(table)


def test_significant_only_on_an_all_null_table_is_refused_with_the_same_message():
    table = _table()
    table["padj"] = 0.9
    with pytest.raises(ValueError, match="no pathway has a finite score"):
        gsea_dotplot(table, significant_only=True)


def test_a_table_missing_the_optional_columns_still_draws():
    """A run from before the leading edge was reported must not crash the figure."""
    table = _table()[["source", "score", "padj"]]
    ax = gsea_dotplot(table).axes[0]
    assert len(_rows(ax)) == 4


# --------------------------------------------------------------------------- #
# the running walk: the one view in which the score is checkable
# --------------------------------------------------------------------------- #


def test_the_walk_marks_the_peak_and_the_peak_is_the_es_the_notes_report():
    walk = _walk()
    fig = gsea_running_es(walk, "UP_STRONG")
    peak = float(walk["running_es"].max())
    assert any(f"{peak:+.3f}" in text.get_text() for text in fig.axes[0].texts)


def test_the_shaded_region_holds_the_hits_left_of_the_peak_for_an_enriched_set():
    walk = _walk(hits=(5, 9, 14, 20, 31))
    notes = _notes(gsea_running_es(walk, "UP_STRONG"))
    # The peak of a walk with every hit at the top is at the last of them, rank 31, and
    # all five hits are inside it.
    assert "5 of 5 pathway genes" in notes
    assert "rank 31" in notes


def test_a_depleted_set_shades_the_other_end_of_the_ranking():
    notes = _notes(gsea_running_es(_walk(up=False), "DOWN_FLOORED"))
    assert "to the end of the ranking" in notes
    assert "from the start of the ranking" not in notes


def test_the_walk_reports_the_table_s_score_and_fdr_when_the_table_is_given():
    notes = _notes(gsea_running_es(_walk(), "UP_STRONG", table=_table()))
    assert "NES = +1.90" in notes
    assert "FDR = 0.004" in notes
    assert "1,000 permutations" in notes


def test_the_walk_names_the_first_leading_edge_genes_so_the_figure_stands_alone():
    notes = _notes(gsea_running_es(_walk(), "UP_STRONG", table=_table()))
    assert "Leading edge: A, B" in notes


def test_a_disagreement_between_the_table_and_the_drawn_walk_is_named_not_hidden():
    """The two are computed differently — closed form in the stage, walked here — so they
    must agree. A difference means the figure and the table came from different runs."""
    table = _table()
    table.loc[table["source"] == "UP_STRONG", "leading_edge_size"] = 99
    notes = _notes(gsea_running_es(_walk(), "UP_STRONG", table=table))
    assert "reported a leading edge of 99" in notes
    assert "different runs" in notes


def test_the_three_tracks_share_one_x_range_and_only_the_bottom_one_is_labelled():
    fig = gsea_running_es(_walk(), "UP_STRONG")
    walk_ax, hits_ax, metric_ax = fig.axes[:3]
    assert walk_ax.get_xlim() == hits_ax.get_xlim() == metric_ax.get_xlim()
    assert not [label for label in walk_ax.get_xticklabels() if label.get_text()]
    assert [label for label in metric_ax.get_xticklabels() if label.get_text()]
    assert "400" in metric_ax.get_xlabel()  # the ranked universe is named


def test_asking_for_a_pathway_the_walk_does_not_hold_says_what_it_does_hold():
    with pytest.raises(ValueError, match="no running-ES walk for 'ABSENT'"):
        gsea_running_es(_walk(), "ABSENT")


def test_the_walk_can_be_drawn_without_its_table():
    notes = _notes(gsea_running_es(_walk(), "UP_STRONG"))
    assert "NES" in notes  # the ES/NES distinction is stated either way
    assert "FDR" not in notes


# --------------------------------------------------------------------------- #
# row labels: a pathway name is not a label anyone chose
# --------------------------------------------------------------------------- #


def _long_table() -> pd.DataFrame:
    """Reactome-shaped names: a shared prefix on every row and one absurdly long one."""
    table = _table()
    table["source"] = [
        "REACTOME_LAMININ_INTERACTIONS",
        "REACTOME_RHO_GTPASE_CYCLE",
        "REACTOME_SIGNALING_BY_WNT",
        "REACTOME_ACTIVATION_OF_THE_MRNA_UPON_BINDING_OF_THE_CAP_BINDING_COMPLEX_AND_EIFS",
    ]
    return table


def test_a_prefix_on_every_row_is_dropped_and_the_drop_is_stated():
    """Nine characters of gutter repeated down the figure, saying what the caption says."""
    fig = gsea_dotplot(_long_table())
    assert _rows(fig.axes[0])[0] == "LAMININ INTERACTIONS"
    assert "shared 'REACTOME' prefix removed" in _notes(fig)


def test_a_prefix_only_some_rows_share_is_kept():
    table = _long_table()
    table.loc[0, "source"] = "HALLMARK_MITOTIC_SPINDLE"
    assert "HALLMARK MITOTIC SPINDLE" in _rows(gsea_dotplot(table).axes[0])


def test_one_unbounded_name_does_not_set_the_width_of_the_whole_figure():
    """The gutter is the widest label, so without a bound the 79-character name decides how
    much room is left for the data area every other row is read in."""
    bounded = gsea_dotplot(_long_table())
    unbounded = gsea_dotplot(_long_table(), max_label_chars=None)
    assert bounded.get_figwidth() < unbounded.get_figwidth()
    assert all(len(label) <= 48 for label in _rows(bounded.axes[0]))
    assert any(label.endswith("…") for label in _rows(bounded.axes[0]))
    assert "truncated with an ellipsis" in _notes(bounded)


def test_a_truncated_name_says_where_the_full_one_is():
    """A truncated name is no longer something a reader can look up, so the note has to say
    the table still has it."""
    assert "the table carries them in full" in _notes(gsea_dotplot(_long_table()))


def test_names_that_all_fit_are_not_footnoted_about_truncation():
    assert "truncated" not in _notes(gsea_dotplot(_table()))


def test_display_names_are_left_alone_by_both_steps():
    """A caller who passed a name chose it; neither the prefix rule nor the length bound is
    entitled to edit it."""
    chosen = "A deliberately chosen and rather long display name for this pathway"
    fig = gsea_dotplot(_long_table(), set_labels=dict.fromkeys(_long_table()["source"], chosen))
    assert set(_rows(fig.axes[0])) == {chosen}
    assert "prefix removed" not in _notes(fig)


def test_the_size_key_labels_clear_the_markers_they_label():
    """The largest marker is the point of the key and overflows a default handle box, which
    prints the reader's gene count on top of the dot it belongs to."""
    fig = gsea_dotplot(_table())
    fig.canvas.draw()  # legend children get their real offsets only at draw time
    legend = fig.axes[0].get_legend()
    renderer = fig.canvas.get_renderer()
    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True):
        marker = handle.get_window_extent(renderer)
        assert text.get_window_extent(renderer).x0 >= marker.x1 - 1.0


# --------------------------------------------------------------------------- #
# two arms: the figure a specificity claim needs
# --------------------------------------------------------------------------- #


def _arms() -> dict[str, pd.DataFrame]:
    """Two arms over five pathways, with one of each interesting case.

    ``FLIPPED`` is significant both ways (the strongest dissociation), ``ONLY_A`` is
    significant in A and null in B, ``AGREED`` is the same in both, ``UNTESTED`` is absent
    from B entirely, and ``NULL_BOTH`` is significant in neither.
    """
    a = pd.DataFrame(
        {
            "source": ["FLIPPED", "ONLY_A", "AGREED", "UNTESTED", "NULL_BOTH"],
            "score": [2.0, 1.5, -2.2, 1.6, 0.4],
            "padj": [0.001, 0.01, 0.001, 0.002, 0.8],
        }
    )
    b = pd.DataFrame(
        {
            "source": ["FLIPPED", "ONLY_A", "AGREED", "NULL_BOTH"],
            "score": [-1.8, 1.4, -2.1, 0.3],
            "padj": [0.01, 0.6, 0.001, 0.9],
        }
    )
    return {"A": a, "B": b}


def _dots(ax) -> dict[float, list[tuple[float, tuple[float, ...]]]]:
    """Every drawn dot as ``row -> [(x, facecolor_rgb)]``."""
    by_row: dict[float, list[tuple[float, tuple[float, ...]]]] = {}
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets) != 1:
            continue
        x, row = float(offsets[0][0]), float(offsets[0][1])
        face = tuple(round(v, 3) for v in collection.get_facecolor()[0][:3])
        by_row.setdefault(row, []).append((x, face))
    return by_row


def test_the_rows_are_ordered_by_how_far_the_arms_disagree():
    """Not by either arm's score: the strongest pathways are the ones both arms agree on,
    which is what a specificity panel is not about."""
    ax = gsea_arm_comparison(_arms()).axes[0]
    # FLIPPED spans 2.0 to -1.8 (3.8), ONLY_A spans 1.5 to 1.4 (0.1), AGREED -2.2 to -2.1
    # (0.1), UNTESTED has one arm so spans 0. NULL_BOTH is not significant anywhere.
    assert _rows(ax)[0] == "FLIPPED"
    assert "NULL BOTH" not in _rows(ax)
    assert len(_rows(ax)) == 4


def test_each_arm_gets_its_own_dot_and_a_bar_joins_them():
    ax = gsea_arm_comparison(_arms()).axes[0]
    row = _rows(ax).index("FLIPPED")
    xs = sorted(x for x, _ in _dots(ax)[float(row)])
    assert xs == pytest.approx([-1.8, 2.0])
    # The connector spans exactly the two dots, and is a line rather than a marker.
    spans = [
        (line.get_xdata()[0], line.get_xdata()[1])
        for line in ax.lines
        if len(line.get_xdata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1] == row
    ]
    assert any(
        min(span) == pytest.approx(-1.8) and max(span) == pytest.approx(2.0) for span in spans
    )


def test_a_tested_null_arm_is_an_open_dot_not_a_missing_one():
    """The reader needs to see the effect was estimated in that arm and how big it was."""
    ax = gsea_arm_comparison(_arms()).axes[0]
    row = float(_rows(ax).index("ONLY A"))
    faces = {round(x, 3): face for x, face in _dots(ax)[row]}
    assert faces[1.4] == (1.0, 1.0, 1.0)  # B: tested, ns → open
    assert faces[1.5] != (1.0, 1.0, 1.0)  # A: significant → filled


def test_an_untested_pathway_draws_one_dot_and_the_note_refuses_to_call_it_a_null():
    fig = gsea_arm_comparison(_arms())
    ax = fig.axes[0]
    assert len(_dots(ax)[float(_rows(ax).index("UNTESTED"))]) == 1
    notes = _notes(fig)
    assert "No dot for B on UNTESTED" in notes
    assert "never tested in that arm, which is not the same as tested and null" in notes


def test_the_two_arms_are_named_in_a_key_and_coloured_differently():
    fig = gsea_arm_comparison(_arms())
    legend = fig.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["A", "B"]
    ax = fig.axes[0]
    row = float(_rows(ax).index("FLIPPED"))
    faces = {face for _, face in _dots(ax)[row]}
    assert len(faces) == 2  # both filled, and not the same colour


def test_an_explicit_source_list_is_drawn_in_the_order_given_whatever_its_spread():
    """For the block a manuscript argues about, rather than the top of a ranking."""
    fig = gsea_arm_comparison(_arms(), sources=["AGREED", "NULL_BOTH", "FLIPPED"])
    assert _rows(fig.axes[0]) == ["AGREED", "NULL BOTH", "FLIPPED"]
    # And the spread-ordering note is withdrawn, because the rows are not sorted by spread.
    assert "chosen set of pathways, drawn in the order given" in _notes(fig)
    assert "largest first" not in _notes(fig)


def test_truncation_of_the_automatic_selection_is_footnoted():
    fig = gsea_arm_comparison(_arms(), max_sets=2)
    assert len(_rows(fig.axes[0])) == 2
    assert "2 further pathway(s) significant in at least one arm are not drawn" in _notes(fig)


def test_the_axis_names_which_direction_is_which_arm_of_the_contrast():
    fig = gsea_arm_comparison(_arms(), case_label="Lymphedema", control_label="Normal")
    assert "Lymphedema" in fig.axes[0].get_xlabel()
    assert "up in Lymphedema" in _notes(fig)


def test_one_arm_is_refused_because_a_comparison_needs_two():
    with pytest.raises(ValueError, match="at least two tables, got 1"):
        gsea_arm_comparison({"A": _arms()["A"]})


def test_a_pathway_no_arm_holds_is_refused_by_name():
    with pytest.raises(ValueError, match="no arm has a row for ABSENT"):
        gsea_arm_comparison(_arms(), sources=["FLIPPED", "ABSENT"])


def test_a_pair_of_all_null_tables_is_refused_rather_than_drawn_empty():
    arms = _arms()
    for table in arms.values():
        table["padj"] = 0.9
    with pytest.raises(ValueError, match="no pathway is significant"):
        gsea_arm_comparison(arms)


def test_the_x_axis_is_symmetric_and_holds_both_arms_extremes():
    ax = gsea_arm_comparison(_arms()).axes[0]
    low, high = ax.get_xlim()
    assert low == pytest.approx(-high)
    assert high > 2.2


def test_the_panel_names_whatever_signed_score_it_was_handed():
    """The same shape of question — two arms, one signed score per row — is asked of a
    leading edge's log fold changes, and an axis reading "enrichment score" over fold
    changes is a mislabelled figure rather than an unlabelled one."""
    fig = gsea_arm_comparison(
        _arms(),
        score_label="log2 fold change",
        row_noun="gene",
        case_label="Lymphedema",
        control_label="Normal",
    )
    assert fig.axes[0].get_xlabel() == "log2 fold change  (→ Lymphedema / ← Normal)"
    assert "One row per gene" in _notes(fig)
    assert "strongest genes are usually the ones both arms agree on" in _notes(fig)
    assert "pathway" not in _notes(fig)


def test_the_default_quantity_is_still_the_enrichment_score():
    fig = gsea_arm_comparison(_arms())
    assert fig.axes[0].get_xlabel().startswith("Normalised enrichment score")
    assert "One row per pathway" in _notes(fig)


def test_a_supplied_order_is_stated_when_the_caller_names_it():
    """The note has to be true of the figure: "not a ranking" was wrong the first time a
    caller passed a list ordered by fold change in one arm."""
    fig = gsea_arm_comparison(
        _arms(),
        sources=["AGREED", "FLIPPED"],
        order_label="|log2 fold change| in A, largest first",
    )
    assert "ordered by |log2 fold change| in A, largest first" in _notes(fig)
    assert "drawn in the order given" not in _notes(fig)


def test_an_order_label_is_ignored_when_the_selection_was_automatic():
    """The panel is ordered by spread in that case, and the note must say so rather than
    repeat a label that does not describe the rows."""
    fig = gsea_arm_comparison(_arms(), order_label="something else")
    assert "something else" not in _notes(fig)
    assert "spread between the arms, largest first" in _notes(fig)
