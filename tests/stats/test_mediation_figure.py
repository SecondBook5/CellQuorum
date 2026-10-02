"""The mediation forest plot: does the drawing carry the table's caveats?

A figure is where caveats go to die — the table says "these two scores share genes"
and the plot shows a confident interval. So what is tested here is not that lines
appear, but that every guard the statistics recorded survives the transfer: the
circularity grade, the pairing flag, the withheld proportion, and a refused fit.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.causal_mediation import MEDIATION_COLUMNS, MEDIATION_TERMS
from cellquorum.visualization.mediation import (
    mediation_footnotes,
    mediation_forest,
    mediation_sensitivity,
)


def _row(**overrides) -> dict:
    base = {
        "group": "all",
        "mediator": "core_junctions",
        "outcome": "endomt",
        "term": "acme",
        "estimate": 0.5,
        "ci_low": 0.2,
        "ci_high": 0.8,
        "p_value": 0.01,
        "p_value_unclustered": 0.01,
        "clustering_changes_the_call": False,
        "n_samples": 18,
        "n_donors": 9,
        "n_case": 9,
        "n_control": 9,
        "shared_features": 0,
        "feature_overlap": 0.0,
        "circularity": "disjoint",
        "method": "donor_bootstrap",
        "reason": "",
    }
    base.update(overrides)
    return base


def _table(*mediators: dict) -> pd.DataFrame:
    """A minimal well-formed table: every term present for every mediator."""
    rows = []
    for spec in mediators:
        for term in MEDIATION_TERMS:
            rows.append(_row(term=term, **spec))
    return pd.DataFrame(rows, columns=list(MEDIATION_COLUMNS))


def _texts(fig) -> str:
    return "\n".join(artist.get_text() for artist in fig.findobj(match=plt.Text))


def test_a_plain_table_draws_one_band_per_mediator():
    table = _table({"mediator": "core_junctions"}, {"mediator": "pro_permeability"})
    fig = mediation_forest(table)
    try:
        ax = fig.axes[0]
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "core junctions",
            "pro permeability",
        ]
    finally:
        plt.close(fig)


def test_display_labels_replace_the_column_names():
    fig = mediation_forest(
        _table({"mediator": "core_junctions"}),
        program_labels={"core_junctions": "Core junctions"},
    )
    try:
        assert fig.axes[0].get_yticklabels()[0].get_text() == "Core junctions"
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# the caveats must reach the drawing
# --------------------------------------------------------------------------- #


def test_a_shared_gene_set_is_marked_on_the_row_and_footnoted():
    fig = mediation_forest(
        _table({"mediator": "integrin_focal_adhesion", "circularity": "overlapping"})
    )
    try:
        assert "†" in fig.axes[0].get_yticklabels()[0].get_text()
        assert "share some genes" in _texts(fig)
    finally:
        plt.close(fig)


def test_a_nested_gene_set_gets_the_stronger_footnote():
    fig = mediation_forest(_table({"mediator": "mesenchymal_gain", "circularity": "nested"}))
    try:
        assert "‡" in fig.axes[0].get_yticklabels()[0].get_text()
        assert "largely definitional" in _texts(fig)
    finally:
        plt.close(fig)


def test_an_unrecorded_pairing_flag_is_not_read_as_a_flagged_one():
    """``bool(float("nan"))`` is True, so a missing flag must not become a caveat."""
    table = _table({"mediator": "core_junctions"})
    table["clustering_changes_the_call"] = np.nan
    fig = mediation_forest(table)
    try:
        assert "treated as" not in _texts(fig)
    finally:
        plt.close(fig)


def test_a_term_that_depended_on_ignoring_the_pairing_is_footnoted():
    table = _table({"mediator": "core_junctions"})
    table.loc[table["term"] == "acme", "clustering_changes_the_call"] = True
    fig = mediation_forest(table)
    try:
        assert "treated as" in _texts(fig)
    finally:
        plt.close(fig)


def test_a_withheld_proportion_is_footnoted_rather_than_silently_absent():
    table = _table({"mediator": "core_junctions"})
    is_proportion = table["term"] == "proportion_mediated"
    table.loc[is_proportion, ["estimate", "ci_low", "ci_high", "p_value"]] = np.nan
    table.loc[is_proportion, "reason"] = "the total effect's interval crosses zero"
    fig = mediation_forest(table)
    try:
        assert "Proportion mediated is withheld" in _texts(fig)
    finally:
        plt.close(fig)


def test_a_reported_proportion_is_not_footnoted_as_withheld():
    """The failure mode this catches: a caveat on a figure that has nothing to caveat.

    A table read back from CSV carries ``reason`` as NaN rather than "", and
    ``str(nan)`` is a three-character string. Testing the reason alone therefore
    footnoted every figure — including the LEC panel, whose proportion mediated was
    reported at 62%.
    """
    table = _table({"mediator": "core_junctions"})
    table["reason"] = np.nan
    fig = mediation_forest(table)
    try:
        assert "withheld" not in _texts(fig)
    finally:
        plt.close(fig)


def test_the_caveats_survive_a_csv_round_trip(tmp_path):
    """Figures are drawn from the written CSV, not from the in-memory frame."""
    table = _table({"mediator": "core_junctions"}, {"mediator": "sparse_program"})
    sparse = table["mediator"] == "sparse_program"
    table.loc[sparse, ["estimate", "ci_low", "ci_high", "p_value"]] = np.nan
    table.loc[sparse, "reason"] = "only 3 donors contribute (floor is 6)"
    is_proportion = (table["term"] == "proportion_mediated") & ~sparse
    table.loc[is_proportion, ["estimate", "ci_low", "ci_high", "p_value"]] = np.nan
    table.loc[is_proportion, "reason"] = "the total effect's interval crosses zero"

    path = tmp_path / "mediation.csv"
    table.to_csv(path, index=False)
    fig = mediation_forest(pd.read_csv(path))
    try:
        text = _texts(fig)
        assert "3 donors contribute" in text
        assert "Proportion mediated is withheld" in text
    finally:
        plt.close(fig)


def test_a_refused_fit_says_so_where_its_intervals_would_have_been():
    table = _table({"mediator": "core_junctions"}, {"mediator": "sparse_program"})
    sparse = table["mediator"] == "sparse_program"
    table.loc[sparse, ["estimate", "ci_low", "ci_high", "p_value"]] = np.nan
    table.loc[sparse, "reason"] = "only 3 donors contribute (floor is 6)"
    fig = mediation_forest(table)
    try:
        text = _texts(fig)
        assert "not estimable" in text
        assert "3 donors contribute" in text
        # The surviving mediator is still drawn.
        assert len(fig.axes[0].get_yticklabels()) == 2
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# composing panels: a lineage beside its specificity control
# --------------------------------------------------------------------------- #


def test_panels_can_share_a_figure_and_each_keeps_its_own_numbers():
    lec = _table({"mediator": "core_junctions"})
    bec = _table(
        {"mediator": "core_junctions", "estimate": 0.001, "ci_low": -0.01, "ci_high": 0.01}
    )
    fig, axes = plt.subplots(1, 2, sharex=True)
    try:
        mediation_forest(lec, ax=axes[0], footnotes=False, title="LEC")
        mediation_forest(bec, ax=axes[1], footnotes=False, legend=False, title="BEC")
        assert axes[0].get_title(loc="left") == "LEC"
        assert axes[1].get_title(loc="left") == "BEC"
        # One key for the pair, not one per panel.
        assert axes[0].get_legend() is not None
        assert axes[1].get_legend() is None
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_a_panel_drawn_into_a_given_axes_does_not_write_footnotes_itself():
    table = _table({"mediator": "integrin_focal_adhesion", "circularity": "overlapping"})
    fig, ax = plt.subplots()
    try:
        mediation_forest(table, ax=ax, footnotes=False)
        assert "share some genes" not in _texts(fig)
        # ...but the caller can still get them, verbatim.
        assert any("share some genes" in note for note in mediation_footnotes(table))
    finally:
        plt.close(fig)


def test_footnotes_read_from_the_table_alone_match_what_the_figure_draws():
    table = _table({"mediator": "core_junctions"}, {"mediator": "sparse_program"})
    sparse = table["mediator"] == "sparse_program"
    table.loc[sparse, ["estimate", "ci_low", "ci_high", "p_value"]] = np.nan
    table.loc[sparse, "reason"] = "only 3 donors contribute (floor is 6)"
    table.loc[table["term"] == "total", "clustering_changes_the_call"] = True

    notes = mediation_footnotes(table, program_labels={"sparse_program": "Sparse program"})
    joined = "\n".join(notes)
    assert "Sparse program" in joined
    assert "3 donors contribute" in joined
    assert "treated as" in joined

    fig = mediation_forest(table, program_labels={"sparse_program": "Sparse program"})
    try:
        drawn = _texts(fig)
        for note in notes:
            assert note in drawn
    finally:
        plt.close(fig)


def test_a_clean_table_earns_no_footnotes():
    assert mediation_footnotes(_table({"mediator": "core_junctions"})) == []


# --------------------------------------------------------------------------- #
# the sensitivity panel: does the claim survive the obvious objections?
# --------------------------------------------------------------------------- #


def test_the_sensitivity_panel_draws_one_interval_per_analysis():
    primary = _table({"mediator": "core_junctions"}, {"mediator": "pro_permeability"})
    adjusted = _table(
        {"mediator": "core_junctions", "estimate": 0.1, "ci_low": -0.05, "ci_high": 0.25},
        {"mediator": "pro_permeability", "estimate": 0.4, "ci_low": 0.1, "ci_high": 0.7},
    )
    fig = mediation_sensitivity({"Primary": primary, "Depth-adjusted": adjusted})
    try:
        ax = fig.axes[0]
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "core junctions",
            "pro permeability",
        ]
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert legend_labels == ["Primary", "Depth-adjusted"]
        # Two mediators x two analyses, each an interval line plus a point marker.
        assert len(ax.lines) == 1 + 2 * 2 * 2  # the zero line, then the intervals
    finally:
        plt.close(fig)


def test_the_reference_analysis_sets_the_row_order_and_a_missing_mediator_leaves_a_gap():
    """A sensitivity run on a subset must not silently reorder or drop the rows."""
    primary = _table({"mediator": "core_junctions"}, {"mediator": "pro_permeability"})
    # Only the overlapping mediator was rescored, so this table holds one row.
    partial = _table({"mediator": "pro_permeability", "estimate": 0.03})
    fig = mediation_sensitivity({"Primary": primary, "Shared genes removed": partial})
    try:
        ax = fig.axes[0]
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "core junctions",
            "pro permeability",
        ]
        # 3 intervals drawn, not 4: core junctions has no rescored counterpart.
        assert len(ax.lines) == 1 + 3 * 2
        # The key follows the caller's order, not the order things happened to be
        # drawn in — the partial analysis skipped the topmost row.
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert legend_labels == ["Primary", "Shared genes removed"]
    finally:
        plt.close(fig)


def test_an_analysis_that_could_not_be_fitted_says_n_a_rather_than_leaving_blank_space():
    primary = _table({"mediator": "core_junctions"})
    refused = _table({"mediator": "core_junctions"})
    refused.loc[:, ["estimate", "ci_low", "ci_high", "p_value"]] = np.nan
    refused.loc[:, "reason"] = "only 3 donors contribute (floor is 6)"
    fig = mediation_sensitivity({"Primary": primary, "By subcluster": refused})
    try:
        text = _texts(fig)
        assert "n/a" in text
        # And the reason is attributed to the analysis it came from.
        assert "By subcluster: Not estimable" in text
    finally:
        plt.close(fig)


def test_the_sensitivity_panel_can_compare_totals_instead_of_indirect_effects():
    primary = _table({"mediator": "core_junctions"})
    fig = mediation_sensitivity({"LEC": primary}, term="total")
    try:
        assert "Total" in fig.axes[0].get_xlabel()
    finally:
        plt.close(fig)


def test_the_sensitivity_panel_refuses_an_empty_mapping():
    with pytest.raises(ValueError, match="no sensitivity analyses"):
        mediation_sensitivity({})


def test_the_sensitivity_panel_refuses_a_reference_with_no_rows_for_the_term():
    with pytest.raises(ValueError, match="no 'acme' rows"):
        mediation_sensitivity({"Primary": _table({"mediator": "core_junctions"})}, group="nope")


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_several_groups_without_a_choice_raises_rather_than_picking_one():
    # Publishing a subtype's mediation under the cohort's title is the failure mode.
    table = pd.concat(
        [_table({"mediator": "core_junctions"}).assign(group=g) for g in ("1", "2")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="pass group="):
        mediation_forest(table)


def test_naming_the_group_draws_just_that_group_and_says_so_in_the_title():
    table = pd.concat(
        [_table({"mediator": "core_junctions"}).assign(group=g) for g in ("1", "2")],
        ignore_index=True,
    )
    fig = mediation_forest(table, group="2")
    try:
        # Left-aligned title, so it is not the default `get_title()` slot.
        assert "2" in fig.axes[0].get_title(loc="left")
    finally:
        plt.close(fig)


def test_an_absent_group_raises_with_the_available_ones_named():
    with pytest.raises(ValueError, match="present groups"):
        mediation_forest(_table({"mediator": "core_junctions"}), group="nope")


def test_an_empty_table_raises():
    with pytest.raises(ValueError, match="empty"):
        mediation_forest(pd.DataFrame(columns=list(MEDIATION_COLUMNS)))
