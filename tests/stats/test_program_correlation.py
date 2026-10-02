"""Program correlation: is the n honest, is the covariate removed, is the overlap disclosed?

The failure this module exists to prevent is not a wrong coefficient — ``scores.corr()``
gets the coefficient right. It is a *right coefficient with a wrong n*, a coefficient that
is really the condition contrast read twice, and a coefficient that is arithmetic because
the two programs share genes. So the tests are built around cases where the naive answer
and the honest answer differ, and they check that the difference shows up in the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.program_correlation import (
    MIN_UNITS,
    program_correlation_tests,
)

RNG = np.random.default_rng(1337)


def _row(table: pd.DataFrame, a: str, b: str) -> pd.Series:
    hit = table[
        ((table["program_a"] == a) & (table["program_b"] == b))
        | ((table["program_a"] == b) & (table["program_b"] == a))
    ]
    assert len(hit) == 1, f"expected exactly one row for {a}/{b}, got {len(hit)}"
    return hit.iloc[0]


def _donor_dataset(per_donor: int = 250):
    """Simpson's paradox: coupled within every donor, unrelated between donors.

    The two donor-offset patterns are orthogonal by construction, so the *donors* carry
    no relationship at all. Every cell then gets a shared perturbation, which couples
    the two programs strongly *within* each donor. A cell-level correlation reports that
    within-donor coupling as if it were a relationship between programs across the
    cohort; averaging to the donor first reports the truth, which is nothing.
    """
    offset_x = np.array([1.0, -1, 1, -1, 1, -1, 1, -1]) * 2.0
    offset_y = np.array([1.0, 1, -1, -1, 1, 1, -1, -1]) * 2.0
    n_donors = len(offset_x)
    donors = np.repeat([f"D{i}" for i in range(n_donors)], per_donor)
    shared = RNG.normal(scale=3.0, size=n_donors * per_donor).reshape(n_donors, per_donor)
    # Centred within each donor, so the coupling is *entirely* within-donor and the donor
    # means are exactly the two orthogonal offsets. Without the centring the shared
    # perturbation leaks into the donor means — identically into both programs — and
    # re-creates at the donor level the very association the fixture means to exclude.
    shared = (shared - shared.mean(axis=1, keepdims=True)).ravel()
    scores = pd.DataFrame(
        {
            "program_x": np.repeat(offset_x, per_donor) + shared,
            "program_y": np.repeat(offset_y, per_donor) + shared,
        }
    )
    return scores, pd.DataFrame({"donor": donors})


# --------------------------------------------------------------------------- #
# the unit is named, never assumed
# --------------------------------------------------------------------------- #


def test_the_unit_and_its_count_are_recorded_on_every_row():
    scores, metadata = _donor_dataset()
    table = program_correlation_tests(scores, metadata, sample_col="donor")
    row = _row(table, "program_x", "program_y")
    assert row["unit"] == "donor"
    assert row["n_units"] == 8  # donors, not the 2000 cells
    assert row["method"] == "spearman"


def test_rows_are_the_units_when_no_sample_column_is_given_and_the_frame_says_so():
    scores = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "b": [2.0, 4, 6, 8, 10, 12]})
    row = _row(program_correlation_tests(scores), "a", "b")
    assert row["unit"] == "row"
    assert row["n_units"] == 6
    assert row["r"] == pytest.approx(1.0)


def test_averaging_within_the_unit_changes_the_answer_not_just_the_n():
    """The whole point: 2,000 cells say one thing and the 8 donors say the opposite."""
    scores, metadata = _donor_dataset()
    naive = float(scores.corr(method="spearman").loc["program_x", "program_y"])
    honest = _row(
        program_correlation_tests(scores, metadata, sample_col="donor"),
        "program_x",
        "program_y",
    )
    # Cell-level: a strong positive correlation which, at n = 2000, is unmissable.
    assert naive > 0.6
    # Donor-level: nothing, over the 8 observations that actually exist.
    assert honest["n_units"] == 8
    assert abs(honest["r"]) < 0.4
    assert honest["p_value"] > 0.2


def test_a_diagonal_is_never_returned():
    scores, metadata = _donor_dataset()
    table = program_correlation_tests(scores, metadata, sample_col="donor")
    assert len(table) == 1  # one pair from two programs, no self-comparisons
    assert not (table["program_a"] == table["program_b"]).any()


def test_the_fdr_family_is_the_pairs_that_were_actually_tested():
    scores = pd.DataFrame(RNG.normal(size=(20, 4)), columns=list("abcd"))
    table = program_correlation_tests(scores)
    assert len(table) == 6  # C(4, 2)
    assert (table["fdr"] >= table["p_value"]).all()


# --------------------------------------------------------------------------- #
# the condition is a common cause, and adjusting for it is reported
# --------------------------------------------------------------------------- #


def test_a_correlation_that_is_only_the_condition_collapses_under_adjustment():
    """Two programs both raised in disease correlate across samples without co-varying."""
    condition = ["Normal"] * 6 + ["Disease"] * 6
    lift = np.array([0.0] * 6 + [4.0] * 6)
    scores = pd.DataFrame(
        {
            "program_p": lift + RNG.normal(scale=0.3, size=12),
            "program_q": lift + RNG.normal(scale=0.3, size=12),
        }
    )
    metadata = pd.DataFrame({"condition": condition})
    row = _row(
        program_correlation_tests(scores, metadata, condition_col="condition"),
        "program_p",
        "program_q",
    )
    # 0.75 rather than 0.99: Spearman sees the two arms as perfectly separated blocks
    # and the within-arm ordering as noise, which caps the rank correlation below 1.
    assert row["r"] > 0.7, "pooled across arms they look tightly coupled"
    assert row["p_value"] < 1e-2
    assert abs(row["r_adjusted"]) < 0.6, "within arms the coupling is gone"
    assert row["p_adjusted"] > 0.05


def test_a_real_within_condition_relationship_survives_adjustment():
    condition = ["Normal"] * 6 + ["Disease"] * 6
    shared = RNG.normal(size=12)
    scores = pd.DataFrame(
        {
            "program_p": shared + RNG.normal(scale=0.1, size=12),
            "program_q": shared + RNG.normal(scale=0.1, size=12),
        }
    )
    metadata = pd.DataFrame({"condition": condition})
    row = _row(
        program_correlation_tests(scores, metadata, condition_col="condition"),
        "program_p",
        "program_q",
    )
    assert row["r_adjusted"] > 0.8
    assert row["p_adjusted"] < 0.01


def test_a_single_condition_level_is_nothing_to_adjust_for_and_is_not_an_error():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    metadata = pd.DataFrame({"condition": ["Disease"] * 10})
    row = _row(program_correlation_tests(scores, metadata, condition_col="condition"), "a", "b")
    assert np.isfinite(row["r"])
    assert np.isnan(row["r_adjusted"])


def test_a_unit_straddling_two_conditions_is_refused_rather_than_averaged():
    scores = pd.DataFrame(RNG.normal(size=(8, 2)), columns=["a", "b"])
    metadata = pd.DataFrame(
        {
            "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
            "condition": ["Normal", "Disease", "Normal", "Normal"]
            + ["Disease", "Disease", "Normal", "Normal"],
        }
    )
    with pytest.raises(ValueError, match="not constant within"):
        program_correlation_tests(scores, metadata, sample_col="donor", condition_col="condition")


def test_adjustment_is_withheld_rather_than_fitted_without_residual_freedom():
    """Three units and one covariate leaves zero degrees of freedom, so there is no fit."""
    scores = pd.DataFrame(RNG.normal(size=(3, 2)), columns=["a", "b"])
    metadata = pd.DataFrame({"condition": ["Normal", "Normal", "Disease"]})
    row = _row(
        program_correlation_tests(scores, metadata, condition_col="condition", min_units=3),
        "a",
        "b",
    )
    assert np.isfinite(row["r"])
    assert np.isnan(row["r_adjusted"])
    assert "residual degree" in row["reason"]


# --------------------------------------------------------------------------- #
# the condition is not the only common cause
# --------------------------------------------------------------------------- #


def test_a_correlation_that_is_only_sequencing_depth_collapses_under_adjustment():
    """The depth confound, which is the one every per-cell score in a study shares.

    Both programs are built from the same depth trend and nothing else, so the pooled
    coefficient is near 1 and there is no relationship left once depth is removed. The
    condition is deliberately *not* the culprit here: it is orthogonal to depth, so a
    condition-only adjustment would have left the artefact standing.
    """
    # The noise is comparable to the spacing between adjacent depths on purpose: with
    # Spearman, two programs whose ranks are *identical* cannot be decorrelated by any
    # adjustment, so a fixture too clean to reorder the ranks would test nothing.
    depth = np.linspace(7.0, 9.0, 30)
    scores = pd.DataFrame(
        {
            "program_p": depth + RNG.normal(scale=0.12, size=30),
            "program_q": depth + RNG.normal(scale=0.12, size=30),
        }
    )
    metadata = pd.DataFrame({"condition": ["Normal", "Disease"] * 15, "depth": depth})
    row = _row(
        program_correlation_tests(
            scores, metadata, condition_col="condition", covariate_cols=["depth"]
        ),
        "program_p",
        "program_q",
    )
    assert row["r"] > 0.95
    assert abs(row["r_adjusted"]) < 0.6
    assert row["p_adjusted"] > 0.05


def test_the_frame_names_what_was_removed_rather_than_leaving_it_to_be_assumed():
    scores = pd.DataFrame(RNG.normal(size=(14, 2)), columns=["a", "b"])
    metadata = pd.DataFrame(
        {"condition": ["Normal", "Disease"] * 7, "depth": np.linspace(7.0, 9.0, 14)}
    )
    row = _row(
        program_correlation_tests(
            scores, metadata, condition_col="condition", covariate_cols=["depth"]
        ),
        "a",
        "b",
    )
    assert row["adjusted_for"] == "condition, depth"
    assert row["n_units_adjusted"] == 14


def test_nothing_removed_is_recorded_as_nothing_removed():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    row = _row(program_correlation_tests(scores), "a", "b")
    assert row["adjusted_for"] == ""
    assert row["n_units_adjusted"] == 0


def test_a_covariate_that_is_constant_is_dropped_rather_than_spending_a_degree_of_freedom():
    """At nine donors a wasted degree of freedom is a quarter of the residual ones, and a
    constant column explains nothing — so it is left out of the design *and* out of the
    record of what was removed."""
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    metadata = pd.DataFrame({"condition": ["Normal"] * 5 + ["Disease"] * 5, "batch": [1.0] * 10})
    row = _row(
        program_correlation_tests(
            scores, metadata, condition_col="condition", covariate_cols=["batch"]
        ),
        "a",
        "b",
    )
    assert row["adjusted_for"] == "condition"


def test_a_covariate_is_averaged_within_the_unit_like_the_scores_are():
    """A per-cell nuisance variable becomes the unit's *mean* nuisance, because the unit is
    the level the coefficient is computed at.

    The depth here is per-cell and noisy; what drives both programs is the donor's depth
    level. Only an adjustment made on the aggregated covariate can remove it, so a
    coefficient that collapses is evidence the aggregation happened.
    """
    generator = np.random.default_rng(7)
    n_donors, per_donor = 12, 20
    levels = np.linspace(7.0, 9.0, n_donors)
    donors = np.repeat([f"D{i}" for i in range(n_donors)], per_donor)
    size = n_donors * per_donor
    metadata = pd.DataFrame(
        {
            "donor": donors,
            "depth": np.repeat(levels, per_donor) + generator.normal(scale=0.3, size=size),
        }
    )
    scores = pd.DataFrame(
        {
            "a": np.repeat(levels, per_donor) + generator.normal(scale=1.0, size=size),
            "b": np.repeat(levels, per_donor) + generator.normal(scale=1.0, size=size),
        }
    )
    row = _row(
        program_correlation_tests(scores, metadata, sample_col="donor", covariate_cols=["depth"]),
        "a",
        "b",
    )
    assert row["n_units"] == n_donors
    assert row["n_units_adjusted"] == n_donors
    assert row["r"] > 0.85
    assert abs(row["r_adjusted"]) < 0.4


def test_a_unit_missing_a_covariate_is_dropped_from_the_adjustment_only():
    """The unadjusted coefficient does not need the covariate, so it keeps every unit.
    Dropping the unit from both would let one missing depth value silently shrink the
    headline n, which is the opposite of what naming the unit was for."""
    depth = np.linspace(7.0, 9.0, 12)
    depth[3] = np.nan
    scores = pd.DataFrame({"a": RNG.normal(size=12), "b": RNG.normal(size=12)})
    metadata = pd.DataFrame({"depth": depth})
    row = _row(program_correlation_tests(scores, metadata, covariate_cols=["depth"]), "a", "b")
    assert row["n_units"] == 12
    assert row["n_units_adjusted"] == 11


def test_a_unit_with_no_condition_is_not_quietly_counted_as_the_reference_arm():
    """An all-zero dummy row is indistinguishable from the dropped first level, so a
    missing condition would be silently reassigned to it."""
    condition = ["Normal"] * 6 + ["Disease"] * 6
    condition[0] = None
    scores = pd.DataFrame({"a": RNG.normal(size=12), "b": RNG.normal(size=12)})
    row = _row(
        program_correlation_tests(
            scores, pd.DataFrame({"condition": condition}), condition_col="condition"
        ),
        "a",
        "b",
    )
    assert row["n_units"] == 12
    assert row["n_units_adjusted"] == 11


def test_a_non_numeric_covariate_is_refused_rather_than_coerced_to_nan():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    metadata = pd.DataFrame({"batch": ["A"] * 5 + ["B"] * 5})
    with pytest.raises(ValueError, match="not numeric"):
        program_correlation_tests(scores, metadata, covariate_cols=["batch"])


def test_a_covariate_alone_needs_no_condition():
    scores = pd.DataFrame(RNG.normal(size=(12, 2)), columns=["a", "b"])
    metadata = pd.DataFrame({"depth": np.linspace(7.0, 9.0, 12)})
    row = _row(program_correlation_tests(scores, metadata, covariate_cols=["depth"]), "a", "b")
    assert row["adjusted_for"] == "depth"
    assert np.isfinite(row["r_adjusted"])


def test_a_missing_covariate_column_names_the_ones_that_are_there():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="not a column of metadata"):
        program_correlation_tests(
            scores, pd.DataFrame({"depth": [1.0] * 10}), covariate_cols=["n_counts"]
        )


# --------------------------------------------------------------------------- #
# shared genes make a correlation arithmetic, so they are disclosed
# --------------------------------------------------------------------------- #


def test_shared_gene_counts_are_reported_per_pair_when_the_lists_are_supplied():
    scores = pd.DataFrame(RNG.normal(size=(12, 3)), columns=["outcome", "subset", "other"])
    genes = {
        "outcome": ["VIM", "FN1", "SPARC", "COL4A1", "TAGLN"],
        "subset": ["VIM", "FN1", "SPARC"],
        "other": ["KLF2", "KLF4"],
    }
    table = program_correlation_tests(scores, program_genes=genes)
    assert _row(table, "outcome", "subset")["shared_genes"] == 3
    assert _row(table, "outcome", "subset")["shares_genes"]
    assert _row(table, "outcome", "other")["shared_genes"] == 0
    assert not _row(table, "outcome", "other")["shares_genes"]


def test_without_gene_lists_the_overlap_is_unknown_not_zero():
    """A 0 would read as "checked, disjoint"; -1 reads as "not supplied"."""
    scores = pd.DataFrame(RNG.normal(size=(12, 2)), columns=["a", "b"])
    row = _row(program_correlation_tests(scores), "a", "b")
    assert row["shared_genes"] == -1
    assert not row["shares_genes"]


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_too_few_units_gets_a_coefficient_and_a_reason_instead_of_a_p_value():
    scores = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 3.0, 2.0]})
    row = _row(program_correlation_tests(scores), "a", "b")
    assert np.isfinite(row["r"])
    assert np.isnan(row["p_value"])
    assert "too few to test" in row["reason"]
    assert str(MIN_UNITS) in row["reason"]


def test_a_constant_program_is_not_correlatable_and_says_why():
    scores = pd.DataFrame({"a": RNG.normal(size=10), "flat": np.ones(10)})
    row = _row(program_correlation_tests(scores), "a", "flat")
    assert np.isnan(row["r"])
    assert np.isnan(row["p_value"])
    assert "constant" in row["reason"]


def test_non_finite_scores_reduce_the_n_they_are_missing_from():
    scores = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, np.nan, 7, 8], "b": [2.0, 1, 4, 3, 6, 5, 8, 7]})
    row = _row(program_correlation_tests(scores), "a", "b")
    assert row["n_units"] == 8  # the units are still 8...
    assert "too few" not in str(row["reason"])  # ...and 7 of them are usable
    assert np.isfinite(row["p_value"])


def test_one_program_is_not_a_correlation_table():
    with pytest.raises(ValueError, match="at least 2 programs"):
        program_correlation_tests(pd.DataFrame({"only": [1.0, 2.0, 3.0]}))


def test_an_unknown_method_is_refused_rather_than_silently_pearson():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="method must be one of"):
        program_correlation_tests(scores, method="kendall")


def test_naming_a_column_without_metadata_is_refused():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="metadata is required"):
        program_correlation_tests(scores, sample_col="donor")


def test_metadata_that_is_not_row_aligned_is_refused():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="row-aligned"):
        program_correlation_tests(scores, pd.DataFrame({"donor": ["D1"] * 9}), sample_col="donor")


def test_a_missing_column_names_the_ones_that_are_there():
    scores = pd.DataFrame(RNG.normal(size=(10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="not a column of metadata"):
        program_correlation_tests(
            scores, pd.DataFrame({"donor": ["D1"] * 10}), sample_col="patient"
        )


# --------------------------------------------------------------------------- #
# pearson, and the frame's shape
# --------------------------------------------------------------------------- #


def test_pearson_and_spearman_disagree_on_a_monotone_but_curved_relationship():
    x = np.arange(1, 13, dtype=float)
    scores = pd.DataFrame({"a": x, "b": x**4})
    spearman = _row(program_correlation_tests(scores), "a", "b")["r"]
    pearson = _row(program_correlation_tests(scores, method="pearson"), "a", "b")["r"]
    assert spearman == pytest.approx(1.0)
    assert pearson < 0.95


def test_the_frame_survives_a_csv_round_trip(tmp_path):
    scores, metadata = _donor_dataset()
    table = program_correlation_tests(scores, metadata, sample_col="donor")
    path = tmp_path / "correlation.csv"
    table.to_csv(path, index=False)
    reread = pd.read_csv(path)
    assert list(reread.columns) == list(table.columns)
    assert reread["n_units"].iloc[0] == 8
