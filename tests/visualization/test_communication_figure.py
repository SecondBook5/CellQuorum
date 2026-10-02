"""Ligand-activity figures: the ranking, the sender grid, the ligand-target grid.

The tests read what was drawn rather than that something was drawn. Three failure modes are
worth pinning: a ranking that answers "where is TGFB1" by leaving it out, a grid that reads a
missing measurement as zero, and a footnote that describes an ordering the figure does not
have.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.communication import (  # noqa: E402
    ligand_activity_arm_comparison,
    ligand_activity_ranking,
    ligand_target_grid,
    sender_attribution_grid,
)


def _activities(n: int = 60) -> pd.DataFrame:
    """A pool with a clear top three and a long tail, so percentiles are meaningful."""
    rng = np.random.default_rng(0)
    names = ["TGFB1", "IL1B", "TNF"] + [f"LIG{i}" for i in range(n - 3)]
    scores = np.concatenate([[0.30, 0.24, 0.21], rng.uniform(0.0, 0.10, size=n - 3)])
    return pd.DataFrame({"test_ligand": names, "aupr_corrected": scores})


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test here builds figures; matplotlib keeps them until closed."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _notes(fig) -> str:
    """The footnotes as one whitespace-normalised string.

    :func:`write_notes` wraps and hanging-indents, so a phrase can straddle a line break;
    matching against the raw text would make an assertion depend on the figure's width.
    """
    return " ".join(" ".join(t.get_text().split()) for t in fig.texts)


def _labels(fig) -> list[str]:
    ax = fig.axes[0]
    return [t.get_text() for t in ax.get_yticklabels()]


# --- ligand_activity_ranking -------------------------------------------------------------


def test_rows_are_the_top_n_best_first_with_their_rank() -> None:
    fig = ligand_activity_ranking(_activities(), top_n=3)
    assert _labels(fig) == ["TGFB1  (1)", "IL1B  (2)", "TNF  (3)"]


def test_the_pool_percentiles_are_over_every_ligand_not_the_drawn_rows() -> None:
    """Drawing p50 of the top three would make every panel's reference line meaningless."""
    activities = _activities()
    fig = ligand_activity_ranking(activities, top_n=3)
    # A reference line is two points at one x; a lollipop marker is one point, and a
    # lollipop stem is two points at two x values.
    lines = [
        line.get_xdata()[0]
        for line in fig.axes[0].get_lines()
        if len(line.get_xdata()) == 2 and len(set(np.round(line.get_xdata(), 9))) == 1
    ]
    expected = float(np.percentile(activities["aupr_corrected"], 50))
    assert any(abs(x - expected) < 1e-9 for x in lines)
    # The median of the top three would be IL1B's 0.24, which must not be a drawn line.
    assert not any(abs(x - 0.24) < 1e-9 for x in lines)


def test_the_notes_say_how_many_of_how_many_were_drawn() -> None:
    fig = ligand_activity_ranking(_activities(60), top_n=5)
    assert "top 5 of 60 ligands tested" in _notes(fig)


def test_the_notes_refuse_to_call_the_percentiles_a_threshold() -> None:
    notes = _notes(ligand_activity_ranking(_activities(), top_n=5))
    assert "no p-value" in notes
    assert "not a significance threshold" in notes


def test_a_named_ligand_outside_the_top_rows_is_still_drawn_and_footnoted() -> None:
    """A ranking that omits the ligand the manuscript is about answers by omission."""
    activities = _activities()
    activities.loc[activities["test_ligand"] == "TGFB1", "aupr_corrected"] = 0.005
    fig = ligand_activity_ranking(activities, top_n=3, highlight=["TGFB1"])
    assert any(label.startswith("TGFB1") for label in _labels(fig))
    assert "asked for" in _notes(fig)


def test_a_named_ligand_absent_from_the_pool_is_not_reported_as_a_low_score() -> None:
    fig = ligand_activity_ranking(_activities(), top_n=3, highlight=["GHOST"])
    assert not any(label.startswith("GHOST") for label in _labels(fig))
    assert "not in the tested ligand pool" in _notes(fig)
    assert "which is not a low score" in _notes(fig)


def test_a_highlighted_row_is_a_different_colour_from_the_rest() -> None:
    fig = ligand_activity_ranking(_activities(), top_n=3, highlight=["IL1B"])
    colors = {
        round(float(line.get_ydata()[0]), 3): line.get_color()
        for line in fig.axes[0].get_lines()
        if len(set(np.round(line.get_ydata(), 9))) == 1 and len(line.get_xdata()) == 2
    }
    # Rows are drawn top-down, so IL1B (rank 2 of 3) sits at y=1 and TGFB1 at y=2.
    assert colors[1.0] != colors[2.0]


def test_the_axis_names_the_receiver_whose_response_was_scored() -> None:
    fig = ligand_activity_ranking(_activities(), top_n=3, receiver_label="LEC")
    assert "LEC" in fig.axes[0].get_xlabel()
    assert "not evidence that any particular cell type sent" in _notes(fig)


def test_the_axis_quantity_is_renamable() -> None:
    fig = ligand_activity_ranking(_activities(), top_n=3, score_label="Pearson activity")
    assert "Pearson activity" in fig.axes[0].get_xlabel()
    assert "AUPR" not in fig.axes[0].get_xlabel()


def test_ranking_refusals() -> None:
    with pytest.raises(ValueError, match="top_n"):
        ligand_activity_ranking(_activities(), top_n=0)
    with pytest.raises(ValueError, match="test_ligand"):
        ligand_activity_ranking(pd.DataFrame({"aupr_corrected": [0.1]}))
    with pytest.raises(ValueError, match="finite"):
        ligand_activity_ranking(pd.DataFrame({"test_ligand": ["A"], "aupr_corrected": [np.nan]}))


# --- sender_attribution_grid -------------------------------------------------------------


def _expression() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender": ["Fib", "Mac", "T", "Fib", "Mac", "T"],
            "ligand": ["TGFB1", "TGFB1", "TGFB1", "IL1B", "IL1B", "IL1B"],
            "fraction_expressing": [0.80, 0.30, 0.02, 0.05, 0.60, 0.40],
            "n_cells": [2000, 1500, 900, 2000, 1500, 900],
            "expressed": ["TRUE", "TRUE", "FALSE", "FALSE", "TRUE", "TRUE"],
        }
    )


def _grid(fig) -> np.ndarray:
    return fig.axes[0].get_images()[0].get_array()


def test_the_grid_holds_the_fractions_at_the_named_rows_and_columns() -> None:
    fig = sender_attribution_grid(_expression(), ligands=["TGFB1", "IL1B"], senders=["Fib", "Mac"])
    assert _grid(fig).tolist() == [[0.80, 0.30], [0.05, 0.60]]


def test_the_caller_order_is_honoured_and_footnoted() -> None:
    fig = sender_attribution_grid(
        _expression(),
        ligands=["IL1B", "TGFB1"],
        senders=["T", "Mac", "Fib"],
        order_label="ligand activity",
    )
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["IL1B", "TGFB1"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T", "Mac", "Fib"]
    assert "in the order given, by ligand activity" in _notes(fig)


def test_without_a_supplied_order_the_note_denies_it_is_the_activity_ranking() -> None:
    notes = _notes(sender_attribution_grid(_expression()))
    assert "not by ligand activity" in notes


def test_a_missing_pair_is_grey_not_zero() -> None:
    expression = _expression()
    expression = expression[~((expression["sender"] == "T") & (expression["ligand"] == "IL1B"))]
    fig = sender_attribution_grid(expression, ligands=["IL1B"], senders=["Fib", "Mac", "T"])
    assert np.ma.is_masked(_grid(fig)[0, 2])
    assert "which is not a fraction of zero" in _notes(fig)


def test_cells_below_the_threshold_are_outlined_and_counted() -> None:
    fig = sender_attribution_grid(
        _expression(), ligands=["TGFB1", "IL1B"], senders=["Fib", "Mac", "T"], expr_prop=0.10
    )
    outlined = [p for p in fig.axes[0].patches if not p.get_fill()]
    # TGFB1/T at 0.02 and IL1B/Fib at 0.05.
    assert len(outlined) == 2
    assert "below the 0.1 detection threshold" in _notes(fig)


def test_the_expressed_flag_is_used_when_no_threshold_is_given() -> None:
    """R writes booleans as text and ``bool("FALSE")`` is ``True``."""
    fig = sender_attribution_grid(
        _expression(), ligands=["TGFB1", "IL1B"], senders=["Fib", "Mac", "T"]
    )
    assert len([p for p in fig.axes[0].patches if not p.get_fill()]) == 2


def test_the_cells_per_sender_are_stated() -> None:
    notes = _notes(sender_attribution_grid(_expression(), senders=["Fib", "Mac"]))
    assert "Fib 2,000" in notes


def test_colour_is_declared_to_be_expression_not_activity() -> None:
    assert "Colour is expression, not activity" in _notes(sender_attribution_grid(_expression()))


def test_a_duplicated_pair_is_refused_rather_than_silently_resolved() -> None:
    expression = pd.concat([_expression(), _expression().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        sender_attribution_grid(expression)


def test_a_requested_sender_absent_from_the_table_is_footnoted() -> None:
    fig = sender_attribution_grid(_expression(), senders=["Fib", "GHOST"])
    assert "GHOST" in _notes(fig)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["Fib"]


def test_the_receiver_held_out_of_its_own_sender_pool_is_footnoted_not_just_missing() -> None:
    """
    A missing column and a column of zeros look the same on the page and are opposite claims.

    The engine excludes the receiver from its own sender pool by design, so a grid drawn from
    that table has no LEC column — and the surviving TGF-beta result in this study is
    autocrine, which is exactly the reading the omission would silently block.
    """
    notes = _notes(sender_attribution_grid(_expression(), receiver_label="LEC"))
    assert "LEC is the receiver and is not a column" in notes
    assert "autocrine" in notes
    assert "not a fraction of zero" in notes


def test_a_receiver_that_is_one_of_the_columns_gets_no_exclusion_note() -> None:
    """The note would be false: the column is there, so nothing was held out."""
    assert "is the receiver and is not a column" not in _notes(
        sender_attribution_grid(_expression(), receiver_label="Fib")
    )


def test_attribution_refusals() -> None:
    with pytest.raises(ValueError, match="sender"):
        sender_attribution_grid(pd.DataFrame({"ligand": ["A"], "fraction_expressing": [0.1]}))


# --- ligand_target_grid ------------------------------------------------------------------


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["TGFB1", "TGFB1", "TGFB1", "IL1B", "IL1B"],
            "target": ["FN1", "COL1A1", "MYL9", "FN1", "CXCL8"],
            "weight": [0.9, 0.7, 0.4, 0.3, 0.8],
        }
    )


def test_the_target_grid_holds_the_weights() -> None:
    fig = ligand_target_grid(_weights(), ligands=["TGFB1"], targets=["FN1", "COL1A1"])
    assert _grid(fig).tolist() == [[0.9, 0.7]]


def test_an_unlinked_pair_is_grey_and_the_note_says_so_is_not_zero() -> None:
    fig = ligand_target_grid(_weights(), ligands=["IL1B"], targets=["COL1A1", "CXCL8"])
    assert np.ma.is_masked(_grid(fig)[0, 0])
    assert "not a weight of zero" in _notes(fig)


def test_the_target_cap_is_footnoted_when_it_bites() -> None:
    fig = ligand_target_grid(_weights(), max_targets=2)
    assert "further target gene(s) are omitted" in _notes(fig)
    assert len(fig.axes[0].get_xticklabels()) == 2


def test_the_cap_is_silent_when_it_does_not_bite() -> None:
    assert "omitted" not in _notes(ligand_target_grid(_weights(), max_targets=40))


def test_curated_module_membership_is_appended_to_the_column_labels() -> None:
    fig = ligand_target_grid(
        _weights(),
        ligands=["TGFB1"],
        targets=["MYL9", "FN1"],
        target_groups={"MYL9": "Actomyosin"},
    )
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "MYL9 · Actomyosin" in labels
    assert "FN1" in labels
    assert "1 of 2 target genes carry their curated-module membership" in _notes(fig)


def test_the_note_calls_the_weights_a_prior_not_a_measurement() -> None:
    notes = _notes(ligand_target_grid(_weights()))
    assert "prior model, not a measured effect" in notes


def test_target_grid_refusals() -> None:
    with pytest.raises(ValueError, match="target"):
        ligand_target_grid(pd.DataFrame({"ligand": ["A"], "weight": [0.1]}))
    with pytest.raises(ValueError, match="finite"):
        ligand_target_grid(pd.DataFrame({"ligand": ["A"], "target": ["B"], "weight": [np.nan]}))


# --- ligand_activity_arm_comparison ------------------------------------------------------


def _two_arms() -> dict[str, pd.DataFrame]:
    """LEC ranks four ligands; BEC ranks three of them and never tested PTK7L."""
    lec = pd.DataFrame(
        {
            "test_ligand": ["TGFB1", "WNT5A", "IL1B", "PTK7L"],
            "aupr_corrected": [0.30, 0.22, 0.11, 0.19],
        }
    )
    bec = pd.DataFrame(
        {"test_ligand": ["TGFB1", "WNT5A", "IL1B"], "aupr_corrected": [0.28, 0.04, 0.10]}
    )
    return {"LEC": lec, "BEC": bec}


def test_the_comparison_orders_by_the_distance_between_the_arms() -> None:
    """A ligand both arms score highly is the tissue response, which is not the finding."""
    fig = ligand_activity_arm_comparison(_two_arms())
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    # PTK7L 0.19 (absent in BEC, so measured from zero), WNT5A 0.18, TGFB1 0.02, IL1B 0.01.
    # TGFB1 outscores WNT5A in both arms and still lands third, which is the whole point.
    assert labels == ["PTK7L †", "WNT5A", "TGFB1", "IL1B"]


def test_a_weak_single_arm_ligand_does_not_outrank_a_strong_two_arm_gap() -> None:
    """
    Verify absences are not a tier that swamps the comparison.

    Ordering every absent-in-some-arm ligand ahead of every two-arm gap sounds conservative and
    is not: a real pool had 106 of 450 single-arm, so twenty of twenty-three rows were an
    alphabetical slice of the absent tier and the comparison itself was pushed off the panel.
    """
    arms = _two_arms()
    arms["LEC"] = pd.concat(
        [arms["LEC"], pd.DataFrame({"test_ligand": ["FAINT"], "aupr_corrected": [0.01]})],
        ignore_index=True,
    )
    fig = ligand_activity_arm_comparison(arms, top_n=2)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    # FAINT is absent from BEC, but it only reached 0.01, so it ranks below WNT5A's 0.18 gap.
    assert labels == ["PTK7L †", "WNT5A"]


def test_a_ligand_never_a_candidate_in_one_arm_gets_one_dot_not_a_zero() -> None:
    fig = ligand_activity_arm_comparison(_two_arms())
    ax = fig.axes[0]
    # Rows are drawn top-down, so PTK7L (row 0 of 4) sits at y=3.
    dots = [
        line
        for line in ax.get_lines()
        if line.get_marker() == "o" and round(float(line.get_ydata()[0]), 6) == 3.0
    ]
    assert len(dots) == 1
    assert abs(float(dots[0].get_xdata()[0]) - 0.19) < 1e-9


def test_the_note_calls_an_absent_ligand_a_receptor_absence_not_a_low_score() -> None:
    notes = _notes(ligand_activity_arm_comparison(_two_arms()))
    assert "never a candidate in BEC" in notes
    assert "not a low score" in notes


def test_the_comparison_refuses_to_imply_the_gap_was_tested() -> None:
    notes = _notes(ligand_activity_arm_comparison(_two_arms()))
    assert "no gap between two dots is a tested difference" in notes


def test_the_two_arms_are_different_colours_and_named_in_the_legend() -> None:
    fig = ligand_activity_arm_comparison(_two_arms())
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["LEC", "BEC"]
    colors = {h.get_color() for h in legend.legend_handles}
    assert len(colors) == 2


def test_a_supplied_ligand_order_is_honoured_and_footnoted() -> None:
    fig = ligand_activity_arm_comparison(
        _two_arms(), ligands=["IL1B", "TGFB1"], order_label="the manuscript's order"
    )
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["IL1B", "TGFB1"]
    assert "in the order given, by the manuscript's order" in _notes(fig)


def test_a_highlighted_ligand_below_the_cut_is_still_drawn() -> None:
    fig = ligand_activity_arm_comparison(_two_arms(), top_n=1, highlight=["TGFB1"])
    assert "TGFB1" in [t.get_text() for t in fig.axes[0].get_yticklabels()]


def test_a_highlighted_ligand_in_no_arm_is_footnoted_not_invented() -> None:
    fig = ligand_activity_arm_comparison(_two_arms(), highlight=["GHOST"])
    assert "GHOST" not in " ".join(t.get_text() for t in fig.axes[0].get_yticklabels())
    assert "ranked in no receiver" in _notes(fig)


def test_the_notes_count_the_pool_across_both_arms() -> None:
    assert "4 rows drawn of 4 ligands ranked in at least one receiver" in _notes(
        ligand_activity_arm_comparison(_two_arms())
    )


def test_the_pool_count_is_not_read_as_the_number_ranked() -> None:
    """``22 of 450 ligands ranked`` reads as if 428 were never ranked; they all were."""
    notes = _notes(ligand_activity_arm_comparison(_two_arms(), top_n=2))
    assert "2 rows drawn of 4 ligands ranked in at least one receiver" in notes


def test_comparison_refusals() -> None:
    arms = _two_arms()
    with pytest.raises(ValueError, match="at least two arms"):
        ligand_activity_arm_comparison({"LEC": arms["LEC"]})
    with pytest.raises(ValueError, match="arm 'BEC' has no 'test_ligand'"):
        ligand_activity_arm_comparison({"LEC": arms["LEC"], "BEC": pd.DataFrame({"x": [1]})})
