"""Set algebra on gene sets, and the four ways it misleads.

Fixtures here are tiny and hand-checkable on purpose: an overlap test is arithmetic, so a
test that cannot be verified by counting is not testing the arithmetic. The interesting
cases are not "does it compute Jaccard" but "does a similarity get a p-value", "does the
universe change the answer", and "does an annotation mismatch surface".
"""

from __future__ import annotations

import numpy as np
import pytest

from cellquorum.stats.gene_set_overlap import (
    COMBINATION_COLUMNS,
    CROSS_COLUMNS,
    MEMBERSHIP_COLUMNS,
    OVERLAP_COLUMNS,
    SELECTION_COLUMNS,
    cross_membership,
    cross_overlap_tests,
    exclusive_combinations,
    selection_overlap,
    set_overlap_tests,
    set_sizes,
)

# Three sets over a 100-gene universe. A/B share 3 of 5, A/C share nothing.
SETS = {
    "A": ["g1", "g2", "g3", "g4", "g5"],
    "B": ["g3", "g4", "g5", "g6", "g7"],
    "C": ["g50", "g51", "g52"],
}
UNIVERSE = [f"g{i}" for i in range(1, 101)]


def _row(table, a, b):
    match = table[(table["set_a"] == a) & (table["set_b"] == b)]
    assert len(match) == 1, f"expected exactly one {a}/{b} row, got {len(match)}"
    return match.iloc[0]


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #


def test_every_unordered_pair_appears_once_and_no_set_is_compared_to_itself():
    table = set_overlap_tests(SETS, universe=UNIVERSE)
    assert list(table.columns) == list(OVERLAP_COLUMNS)
    assert len(table) == 3  # A/B, A/C, B/C — not 9, not 6
    assert not (table["set_a"] == table["set_b"]).any()


def test_the_counted_quantities_are_the_ones_you_would_count_by_hand():
    row = _row(set_overlap_tests(SETS, universe=UNIVERSE), "A", "B")
    assert row["size_a"] == 5
    assert row["size_b"] == 5
    assert row["intersection"] == 3
    assert row["jaccard"] == pytest.approx(3 / 7)  # union is g1..g7
    assert row["overlap_coefficient"] == pytest.approx(3 / 5)
    # 5 x 5 / 100 genes = 0.25 expected by chance, so 3 observed is 12-fold.
    assert row["expected"] == pytest.approx(0.25)
    assert row["fold_enrichment"] == pytest.approx(12.0)


def test_a_real_overlap_is_significant_and_a_disjoint_pair_is_not():
    table = set_overlap_tests(SETS, universe=UNIVERSE)
    # P(share >= 3) drawing 5 and 5 from 100, summed by hand over the three ways it
    # can happen. Spelled out rather than compared to a round threshold so the test
    # would catch a left-tail or two-sided p-value, not just an implausible one.
    total = 75287520  # C(100, 5)
    expected = (10 * 4465 + 5 * 95 + 1) / total
    assert _row(table, "A", "B")["p_value"] == pytest.approx(expected)
    # No shared genes: P(overlap >= 0) is exactly 1, never a small number.
    assert _row(table, "A", "C")["p_value"] == pytest.approx(1.0)
    assert _row(table, "A", "C")["intersection"] == 0


def test_fdr_corrects_over_the_pairs_and_only_the_pairs():
    table = set_overlap_tests(SETS, universe=UNIVERSE)
    # 3 pairs, so the smallest p-value is multiplied by at most 3.
    smallest = table["p_value"].min()
    assert table.loc[table["p_value"].idxmin(), "fdr"] == pytest.approx(
        min(smallest * 3, 1.0), rel=1e-6
    )


# --------------------------------------------------------------------------- #
# the universe is the whole ballgame
# --------------------------------------------------------------------------- #


def test_the_same_overlap_is_more_significant_in_a_smaller_universe():
    """The reason there is no default: the convenient universe is the flattering one."""
    wide = _row(set_overlap_tests(SETS, universe=UNIVERSE), "A", "B")
    narrow = _row(set_overlap_tests(SETS, universe=[f"g{i}" for i in range(1, 11)]), "A", "B")
    assert narrow["p_value"] > wide["p_value"]
    assert narrow["fold_enrichment"] < wide["fold_enrichment"]
    # Same counted overlap, different conclusion — which is why it must be stated.
    assert narrow["intersection"] == wide["intersection"] == 3


def test_the_universe_is_required():
    with pytest.raises(TypeError):
        set_overlap_tests(SETS)  # type: ignore[call-arg]


def test_an_empty_universe_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="universe is empty"):
        set_overlap_tests(SETS, universe=[])


def test_one_set_cannot_make_a_pairwise_table():
    with pytest.raises(ValueError, match="at least 2 sets"):
        set_overlap_tests({"A": ["g1"]}, universe=UNIVERSE)


# --------------------------------------------------------------------------- #
# genes outside the universe are a warning
# --------------------------------------------------------------------------- #


def test_members_outside_the_universe_are_dropped_from_the_draw_and_counted():
    sets = {"A": ["g1", "g2", "g3", "Mm.Actb", "Mm.Vcl"], "B": ["g3", "g4"]}
    row = _row(set_overlap_tests(sets, universe=UNIVERSE), "A", "B")
    # A gene that could not have been drawn must not enlarge the draw.
    assert row["size_a"] == 3
    assert row["dropped_a"] == 2
    assert row["dropped_b"] == 0


def test_a_set_with_nothing_in_the_universe_is_not_tested_and_says_why():
    sets = {"A": ["g1", "g2"], "B": ["Hs.NOTAGENE", "Hs.ALSONOT"]}
    row = _row(set_overlap_tests(sets, universe=UNIVERSE), "A", "B")
    assert np.isnan(row["p_value"])
    assert "same gene naming" in row["reason"]
    # The counted columns are still reported, so the mismatch is visible.
    assert row["size_b"] == 0
    assert row["dropped_b"] == 2


# --------------------------------------------------------------------------- #
# the exclusive-membership table
# --------------------------------------------------------------------------- #


def test_exclusive_combinations_partition_the_union_exactly_once():
    table = exclusive_combinations(SETS)
    assert list(table.columns) == list(COMBINATION_COLUMNS)
    union = set().union(*(set(v) for v in SETS.values()))
    assert table["size"].sum() == len(union)
    listed = [gene for row in table["elements"] for gene in row.split(", ")]
    assert sorted(listed) == sorted(union)


def test_the_exclusive_rows_hold_only_what_is_in_no_other_set():
    table = exclusive_combinations(SETS).set_index("combination")
    assert table.loc["A", "size"] == 2  # g1, g2
    assert table.loc["A", "elements"] == "g1, g2"
    assert table.loc["A & B", "size"] == 3  # g3, g4, g5
    assert table.loc["C", "size"] == 3


def test_unoccupied_combinations_are_omitted_rather_than_listed_as_zero():
    table = exclusive_combinations(SETS)
    # 3 sets have 7 possible combinations; only 4 are occupied.
    assert len(table) == 4
    assert "A & C" not in set(table["combination"])
    assert (table["size"] > 0).all()


def test_rows_read_exclusive_first_then_largest_first():
    table = exclusive_combinations(SETS)
    assert list(table["n_sets"]) == sorted(table["n_sets"])
    singletons = table[table["n_sets"] == 1]
    assert list(singletons["size"]) == sorted(singletons["size"], reverse=True)


def test_min_size_drops_the_long_tail_of_one_gene_combinations():
    sets = {"A": ["g1", "g2", "g3"], "B": ["g3", "g4"], "C": ["g5"]}
    assert len(exclusive_combinations(sets)) == 4
    kept = exclusive_combinations(sets, min_size=2)
    assert list(kept["combination"]) == ["A"]


def test_the_element_lists_can_be_suppressed_for_a_size_only_table():
    table = exclusive_combinations(SETS, list_elements=False)
    assert (table["elements"] == "").all()
    assert table["size"].sum() > 0


def test_no_sets_is_refused():
    with pytest.raises(ValueError, match="no sets given"):
        exclusive_combinations({})


# --------------------------------------------------------------------------- #
# the sidebar
# --------------------------------------------------------------------------- #


def test_set_sizes_splits_each_set_into_exclusive_and_shared():
    table = set_sizes(SETS).set_index("set")
    assert table.loc["A", "size"] == 5
    assert table.loc["A", "exclusive"] == 2
    assert table.loc["A", "shared"] == 3
    assert table.loc["A", "fraction_exclusive"] == pytest.approx(0.4)
    assert table.loc["C", "exclusive"] == 3  # C shares with nobody


def test_set_sizes_reports_what_the_universe_cost():
    sets = {"A": ["g1", "g2", "Mm.Actb"], "B": ["g2", "g3"]}
    table = set_sizes(sets, universe=UNIVERSE).set_index("set")
    assert table.loc["A", "size"] == 2
    assert table.loc["A", "outside_universe"] == 1
    assert table.loc["B", "outside_universe"] == 0


# --------------------------------------------------------------------------- #
# attributing an overlap: did the annotation already force it?
# --------------------------------------------------------------------------- #

# Two pathways that share four genes in the annotation. P's selection takes three of the
# four shared genes, Q's takes two, and they agree on two of them. Every number below is
# countable off these six lines.
REFERENCE = {
    "P": ["s1", "s2", "s3", "s4", "p5", "p6"],
    "Q": ["s1", "s2", "s3", "s4", "q5", "q6", "q7"],
}
SELECTED = {
    "P": ["s1", "s2", "s3", "p5"],
    "Q": ["s1", "s2", "q5"],
}


def test_selection_overlap_separates_what_was_taken_from_what_was_available():
    row = selection_overlap(SELECTED, REFERENCE).iloc[0]
    assert list(selection_overlap(SELECTED, REFERENCE).columns) == list(SELECTION_COLUMNS)
    assert row["reference_intersection"] == 4  # s1..s4 were shared before anything was picked
    assert row["selected_intersection"] == 2  # s1, s2 were picked by both
    assert row["captured_a"] == 3  # P took s1, s2, s3 of the four
    assert row["captured_b"] == 2  # Q took s1, s2
    assert row["fraction_of_reference_intersection"] == pytest.approx(0.5)
    assert row["shared_elements"] == "s1, s2"


def test_the_descriptive_expectation_is_the_two_capture_rates_over_the_shared_pool():
    """3 of 4 and 2 of 4 taken independently would agree on 3*2/4 = 1.5 genes; they agree
    on 2. The number is a reading aid for `selected_intersection`, not a test statistic."""
    row = selection_overlap(SELECTED, REFERENCE).iloc[0]
    assert row["expected_if_unrelated"] == pytest.approx(1.5)
    assert row["fold_over_unrelated"] == pytest.approx(2 / 1.5)


def test_selection_overlap_reports_no_p_value_at_all():
    """The absence is the design. Both selections are functions of one shared ranking, so
    there is no sampling to place a null under, and a hypergeometric right tail here
    returns a tiny number for a structural fact."""
    table = selection_overlap(SELECTED, REFERENCE)
    assert not [column for column in table.columns if "p_value" in column or column == "fdr"]


def test_disjoint_reference_sets_say_so_rather_than_dividing_by_zero():
    reference = {"P": ["a1", "a2"], "Q": ["b1", "b2"]}
    row = selection_overlap({"P": ["a1"], "Q": ["b1"]}, reference).iloc[0]
    assert row["reference_intersection"] == 0
    assert row["selected_intersection"] == 0
    assert np.isnan(row["fraction_of_reference_intersection"])
    assert np.isnan(row["expected_if_unrelated"])
    assert "disjoint" in row["reason"]


def test_the_universe_restricts_the_selection_and_its_reference_together():
    """Comparing a selection counted over all genes against a reference counted over
    detected genes only would attribute the difference to the selection."""
    reference = {"P": ["g1", "g2", "Mm.Actb"], "Q": ["g1", "g2", "g3"]}
    selected = {"P": ["g1", "Mm.Actb"], "Q": ["g1"]}
    row = selection_overlap(selected, reference, universe=UNIVERSE).iloc[0]
    assert row["reference_a"] == 2
    assert row["selected_a"] == 1
    assert row["reference_intersection"] == 2


def test_a_selection_outside_its_reference_set_is_a_join_error_and_raises():
    """A leading edge is a subset of its pathway by construction. One that is not means the
    two tables were keyed differently, and every overlap below it is unattributable."""
    with pytest.raises(ValueError, match="not contained in its reference set"):
        selection_overlap({"P": ["s1", "elsewhere"], "Q": ["s1"]}, REFERENCE)


def test_a_selection_with_no_reference_set_raises():
    with pytest.raises(ValueError, match="no reference set"):
        selection_overlap({"P": ["s1"], "R": ["s1"]}, REFERENCE)


def test_one_selection_is_not_a_pairwise_table():
    with pytest.raises(ValueError, match="at least 2 selections"):
        selection_overlap({"P": ["s1"]}, REFERENCE)


# --------------------------------------------------------------------------- #
# one family against another: does the curated panel sit where the data moved?
# --------------------------------------------------------------------------- #

# A curated module of 5 and a data-derived edge of 5, sharing 4, over 100 genes.
CURATED = {"actomyosin": ["g1", "g2", "g3", "g4", "g5"], "unrelated": ["g80", "g81"]}
EDGES = {"MITOTIC_SPINDLE": ["g1", "g2", "g3", "g4", "g60"], "OTHER": ["g70", "g71", "g72"]}


def _cross(table, left, right):
    match = table[(table["left"] == left) & (table["right"] == right)]
    assert len(match) == 1, f"expected exactly one {left}/{right} row, got {len(match)}"
    return match.iloc[0]


def test_every_left_right_pair_appears_once_and_no_within_family_pair_does():
    table = cross_overlap_tests(CURATED, EDGES, universe=UNIVERSE)
    assert list(table.columns) == list(CROSS_COLUMNS)
    assert len(table) == 4  # 2 x 2, not 6 unordered pairs over the union
    assert set(table["left"]) == set(CURATED)
    assert set(table["right"]) == set(EDGES)


def test_the_counted_quantities_match_the_within_family_table_on_the_same_two_sets():
    """The two tables share one row builder, so "fold enrichment" cannot come to mean two
    different things depending on which function a driver happened to call."""
    table = cross_overlap_tests(CURATED, EDGES, universe=UNIVERSE)
    row = _cross(table, "actomyosin", "MITOTIC_SPINDLE")
    same = {"A": CURATED["actomyosin"], "B": EDGES["MITOTIC_SPINDLE"]}
    within = _row(set_overlap_tests(same, universe=UNIVERSE), "A", "B")
    for column in ("intersection", "jaccard", "overlap_coefficient", "expected", "p_value"):
        assert row[column] == pytest.approx(within[column])
    assert row["size_left"] == within["size_a"]
    assert row["size_right"] == within["size_b"]


def test_the_shared_genes_are_listed_because_on_a_hit_they_are_the_finding():
    """A curated actomyosin module overlapping a set labelled "mitotic spindle" is only
    interpretable once a reader can see that the shared genes are the myosins."""
    table = cross_overlap_tests(CURATED, EDGES, universe=UNIVERSE)
    row = _cross(table, "actomyosin", "MITOTIC_SPINDLE")
    assert row["shared_elements"] == "g1, g2, g3, g4"
    compact = cross_overlap_tests(CURATED, EDGES, universe=UNIVERSE, list_elements=False)
    assert (compact["shared_elements"] == "").all()


def test_a_real_concordance_is_significant_and_a_disjoint_pair_is_not():
    table = cross_overlap_tests(CURATED, EDGES, universe=UNIVERSE)
    assert _cross(table, "actomyosin", "MITOTIC_SPINDLE")["p_value"] < 1e-5
    assert _cross(table, "actomyosin", "OTHER")["p_value"] == pytest.approx(1.0)
    assert _cross(table, "unrelated", "OTHER")["intersection"] == 0


def test_fdr_corrects_over_the_full_cross_family():
    table = cross_overlap_tests(CURATED, EDGES, universe=UNIVERSE)
    smallest = table["p_value"].min()
    assert table.loc[table["p_value"].idxmin(), "fdr"] == pytest.approx(
        min(smallest * 4, 1.0), rel=1e-6
    )


def test_members_outside_the_universe_are_reported_per_side():
    curated = {"m": ["g1", "g2", "Mm.Actb"]}
    row = _cross(cross_overlap_tests(curated, {"e": ["g1", "g2"]}, universe=UNIVERSE), "m", "e")
    assert row["size_left"] == 2
    assert row["dropped_left"] == 1
    assert row["dropped_right"] == 0


def test_a_side_with_nothing_in_the_universe_is_not_tested_and_says_why():
    table = cross_overlap_tests({"m": ["Hs.NOTAGENE"]}, {"e": ["g1"]}, universe=UNIVERSE)
    row = table.iloc[0]
    assert np.isnan(row["p_value"])
    assert "same gene naming" in row["reason"]


def test_an_empty_family_or_universe_is_refused():
    with pytest.raises(ValueError, match="the left family is empty"):
        cross_overlap_tests({}, EDGES, universe=UNIVERSE)
    with pytest.raises(ValueError, match="the right family is empty"):
        cross_overlap_tests(CURATED, {}, universe=UNIVERSE)
    with pytest.raises(ValueError, match="universe is empty"):
        cross_overlap_tests(CURATED, EDGES, universe=[])


def test_one_set_per_family_is_allowed_unlike_the_within_family_table():
    """A single curated module against a single leading edge is a real question; the
    within-family table refuses one set because it would have no pair at all."""
    table = cross_overlap_tests({"m": ["g1"]}, {"e": ["g1"]}, universe=UNIVERSE)
    assert len(table) == 1
    assert table.iloc[0]["intersection"] == 1


# --- cross_membership: the circularity check, deliberately without a null ---


def test_the_naming_side_fraction_is_what_makes_a_naming_decision_circular():
    """A small panel wholly inside a large module leans entirely on that module.

    Both fractions are on the row because they carry different consequences: the module is
    barely touched, and yet every gene the cluster was named with is one the module tests.
    """
    panels = {"valve": ["FOXC2", "GATA2", "CLDN11"]}
    modules = {"valve_collecting": ["FOXC2", "GATA2", "CLDN11", *[f"g{i}" for i in range(37)]]}
    row = cross_membership(panels, modules).iloc[0]

    assert list(cross_membership(panels, modules).columns) == list(MEMBERSHIP_COLUMNS)
    assert row["intersection"] == 3
    assert row["fraction_of_left"] == pytest.approx(1.0)
    assert row["fraction_of_right"] == pytest.approx(3 / 40)
    assert row["shared_elements"] == "CLDN11;FOXC2;GATA2"


def test_a_panel_that_shares_nothing_gets_no_row_because_absence_is_the_clean_case():
    table = cross_membership({"clean": ["a", "b"]}, {"module": ["x", "y"]})
    assert table.empty
    assert list(table.columns) == list(MEMBERSHIP_COLUMNS)


def test_the_most_compromised_pair_reads_first():
    panels = {"whole": ["a", "b"], "partial": ["a", "c", "d", "e"]}
    modules = {"m": ["a", "b", "c"]}
    table = cross_membership(panels, modules)
    assert list(table["left"]) == ["whole", "partial"]
    assert table.iloc[0]["fraction_of_left"] == pytest.approx(1.0)
    assert table.iloc[1]["fraction_of_left"] == pytest.approx(0.5)


def test_there_is_no_p_value_and_that_is_the_point():
    """Two curated families overlap by construction; a null would dress that up."""
    table = cross_membership({"a": ["g1"]}, {"b": ["g1"]})
    assert not {"p_value", "fdr", "fold_enrichment"} & set(table.columns)
