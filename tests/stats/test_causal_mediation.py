"""Causal mediation: does the decomposition recover a mechanism it was given?

The estimator is short enough that the interesting tests are not "does the algebra
work" but "does each guard actually fire". A mediation table is easy to produce and
hard to falsify — it always returns six plausible numbers — so what is tested here
is mostly the four situations in which those six numbers must NOT be believed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stats.causal_mediation import (
    MEDIATION_COLUMNS,
    MEDIATION_TERMS,
    mediation_effects,
    mediation_grid,
)

# Small enough to be fast, large enough for the donor bootstrap to be an interval
# rather than a rumour.
N_BOOT = 400
N_SIMS = 400


def _paired_cohort(
    *,
    n_donors: int = 9,
    cells_per_sample: int = 20,
    samples_per_arm: int = 1,
    paired: bool = True,
    path_a: float = 1.0,
    path_b: float = 1.0,
    direct: float = 0.0,
    donor_spread: float = 0.5,
    noise: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a paired cohort whose true ACME is exactly ``path_a * path_b``.

    The donor random intercept enters the **mediator only**, and the outcome sees the
    donor solely through the mediator. That is deliberate: if the donor offset were
    added to the outcome as well it would confound M -> Y, the recovered path *b*
    would not be ``path_b``, and the fixture could no longer say what the right answer
    is. Within-donor correlation still exists (a donor's two samples share the offset,
    which propagates into their outcomes), so the clustering question is still live.

    ``samples_per_arm`` > 1 gives a donor several libraries per condition.
    ``paired=False`` makes each donor contribute one condition instead of both, which
    is the cross-sectional design where the donor offset does *not* cancel.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for donor_index in range(n_donors):
        donor_offset = rng.normal(0.0, donor_spread)
        arms = (("Normal", 0.0), ("Disease", 1.0))
        if not paired:
            arms = (arms[donor_index % 2],)
        for condition, treatment_code in arms:
            for replicate in range(samples_per_arm):
                mediator = donor_offset + path_a * treatment_code + rng.normal(0.0, noise)
                outcome = direct * treatment_code + path_b * mediator + rng.normal(0.0, noise)
                suffix = f"_{replicate}" if samples_per_arm > 1 else ""
                for cell_index in range(cells_per_sample):
                    rows.append(
                        {
                            "cell": f"D{donor_index}_{condition}{suffix}_{cell_index}",
                            "sample": f"D{donor_index}_{condition}{suffix}",
                            "donor": f"D{donor_index}",
                            "condition": condition,
                            # Per-cell jitter around the sample's value: the mediation is
                            # a property of the sample, cells are noisy reads of it.
                            "mediator_score": mediator + rng.normal(0.0, noise),
                            "outcome_score": outcome + rng.normal(0.0, noise),
                            "depth": rng.normal(0.0, 1.0),
                        }
                    )
    return pd.DataFrame(rows)


def _run(cells: pd.DataFrame, **kwargs) -> pd.DataFrame:
    defaults = {
        "sample": "sample",
        "donor": "donor",
        "treatment": "condition",
        "mediator": "mediator_score",
        "outcome": "outcome_score",
        "case": "Disease",
        "control": "Normal",
        "n_boot": N_BOOT,
        "n_sims": N_SIMS,
    }
    return mediation_effects(cells, **{**defaults, **kwargs})


def _term(result: pd.DataFrame, term: str, group: str = "all") -> pd.Series:
    match = result[(result["term"] == term) & (result["group"] == group)]
    assert len(match) == 1, f"expected exactly one {term!r} row for {group!r}, got {len(match)}"
    return match.iloc[0]


# --------------------------------------------------------------------------- #
# does it recover a mechanism it was given?
# --------------------------------------------------------------------------- #


def test_a_fully_mediated_effect_is_recovered_as_mediated():
    # Truth: a = 1, b = 1, direct = 0. So ACME = 1, total = 1, everything mediated.
    result = _run(_paired_cohort(path_a=1.0, path_b=1.0, direct=0.0))

    assert _term(result, "path_a")["estimate"] == pytest.approx(1.0, abs=0.25)
    assert _term(result, "path_b")["estimate"] == pytest.approx(1.0, abs=0.25)
    assert _term(result, "acme")["estimate"] == pytest.approx(1.0, abs=0.3)
    assert _term(result, "direct")["estimate"] == pytest.approx(0.0, abs=0.3)
    assert _term(result, "proportion_mediated")["estimate"] == pytest.approx(1.0, abs=0.3)
    assert _term(result, "acme")["p_value"] < 0.05


def test_an_unmediated_effect_is_not_reported_as_mediated():
    # The treatment moves the outcome directly and does not touch the mediator, so
    # path a is null and the ACME must be too even though the total effect is large.
    result = _run(_paired_cohort(path_a=0.0, path_b=1.0, direct=2.0))

    assert _term(result, "path_a")["estimate"] == pytest.approx(0.0, abs=0.25)
    assert _term(result, "acme")["estimate"] == pytest.approx(0.0, abs=0.3)
    assert _term(result, "acme")["p_value"] > 0.05
    assert _term(result, "direct")["estimate"] == pytest.approx(2.0, abs=0.3)
    assert _term(result, "total")["p_value"] < 0.05


def test_the_parts_sum_to_the_whole():
    # total = direct + ACME by construction; if this drifts, the decomposition is not
    # a decomposition and the proportion is not a proportion of anything.
    result = _run(_paired_cohort(path_a=0.8, path_b=1.5, direct=0.4))
    total = _term(result, "total")["estimate"]
    assert total == pytest.approx(
        _term(result, "direct")["estimate"] + _term(result, "acme")["estimate"]
    )
    assert _term(result, "proportion_mediated")["estimate"] == pytest.approx(
        _term(result, "acme")["estimate"] / total
    )


def test_the_result_is_deterministic_under_a_fixed_seed():
    cells = _paired_cohort()
    first = _run(cells, seed=7)
    second = _run(cells, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_every_reported_column_is_present_and_every_term_is_reported():
    result = _run(_paired_cohort())
    assert list(result.columns) == list(MEDIATION_COLUMNS)
    assert list(result["term"]) == list(MEDIATION_TERMS)


# --------------------------------------------------------------------------- #
# guard 1: the unit is a sample, never a cell
# --------------------------------------------------------------------------- #


def test_adding_cells_does_not_shrink_the_intervals():
    """The pseudoreplication guard, stated as the thing it prevents.

    Ten times the cells is not ten times the evidence — the treatment was applied to
    nine donors either way. If the aggregation were skipped, n would jump from 18 to
    thousands and the interval would collapse.
    """
    narrow = _run(_paired_cohort(cells_per_sample=5, seed=3))
    wide = _run(_paired_cohort(cells_per_sample=50, seed=3))

    def width(result: pd.DataFrame) -> float:
        row = _term(result, "acme")
        return float(row["ci_high"] - row["ci_low"])

    # Averaging more cells does denoise each sample's score a little, so the widths
    # are not identical; what must not happen is a collapse towards zero.
    assert width(wide) > 0.4 * width(narrow)
    # Ten times the cells, the same eighteen samples and nine donors.
    assert narrow["n_samples"].iloc[0] == wide["n_samples"].iloc[0] == 18
    assert narrow["n_donors"].iloc[0] == wide["n_donors"].iloc[0] == 9


def test_an_already_aggregated_frame_is_accepted_unchanged():
    cells = _paired_cohort()
    per_sample = (
        cells.groupby("sample", as_index=False)
        .agg(
            donor=("donor", "first"),
            condition=("condition", "first"),
            mediator_score=("mediator_score", "mean"),
            outcome_score=("outcome_score", "mean"),
            depth=("depth", "mean"),
        )
        .assign(sample=lambda frame: frame["sample"])
    )
    from_cells = _run(cells)
    from_samples = _run(per_sample)
    assert _term(from_samples, "acme")["estimate"] == pytest.approx(
        _term(from_cells, "acme")["estimate"]
    )


def test_a_sample_that_spans_two_donors_is_refused():
    # A sample column that does not identify a sample would silently average across
    # people; that must be an error, not a majority vote.
    cells = _paired_cohort()
    cells.loc[cells.index[0], "donor"] = "SOMEONE_ELSE"
    with pytest.raises(ValueError, match="does not identify a sample"):
        _run(cells)


# --------------------------------------------------------------------------- #
# guard 2: paired donors are not independent samples
# --------------------------------------------------------------------------- #


def test_repeated_samples_from_one_donor_widen_the_clustered_interval():
    """Where ignoring the donor structure inflates n outright.

    Eighteen donors, three libraries each, one condition per donor: fifty-four rows and
    still eighteen people. Nothing cancels here, because the treatment varies *between*
    donors, so the unclustered fit charges all fifty-four rows to the residual degrees
    of freedom and reports a tighter interval than the cohort supports.
    """
    result = _run(
        _paired_cohort(
            n_donors=18,
            samples_per_arm=3,
            paired=False,
            donor_spread=2.0,
            path_a=0.3,
            path_b=0.3,
            seed=11,
        )
    )
    assert result["n_samples"].iloc[0] == 54
    assert result["n_donors"].iloc[0] == 18

    path_a = _term(result, "path_a")
    assert path_a["ci_high"] - path_a["ci_low"] > 0.0
    assert path_a["p_value"] > path_a["p_value_unclustered"]


def test_clustering_can_also_tighten_an_interval_because_pairing_is_information():
    """The other direction, which is not a bug and must not be "corrected" away.

    In a strictly paired design the donor offset cancels inside each donor's
    case-minus-control contrast. The unclustered fit cannot see that — it charges
    between-donor spread to residual noise and reports path *a* as barely determined.
    Resampling whole donors carries both of a donor's conditions together, so the
    offsets cancel there too and the interval is *narrower*.

    So the guard is not "clustered intervals are wider". It is "clustered intervals
    are right", and the direction depends on the design.
    """
    result = _run(_paired_cohort(samples_per_arm=1, donor_spread=3.0, path_a=1.0, seed=11))
    path_a = _term(result, "path_a")
    assert path_a["p_value"] < path_a["p_value_unclustered"]
    # And it is the pairing, not luck: the true value is inside the tighter interval.
    assert path_a["ci_low"] < 1.0 < path_a["ci_high"]


def test_a_disagreement_between_the_two_inferences_is_flagged_on_the_row():
    """Scan many cohorts for one where the pairing changes the call, and check it says so.

    This is the finding the flag exists to surface, so the test asserts the flag is
    reachable rather than asserting it on a hand-picked seed that a numpy change could
    quietly invalidate.
    """
    flagged = None
    for seed in range(12):
        result = _run(
            _paired_cohort(donor_spread=3.0, path_a=0.25, path_b=0.25, noise=0.5, seed=seed)
        )
        hits = result[result["clustering_changes_the_call"] == True]  # noqa: E712
        if len(hits):
            flagged = hits.iloc[0]
            break
    assert flagged is not None, "no cohort in the sweep had the two inferences disagree"
    assert "changes this call" in flagged["reason"]
    assert flagged["p_value"] != flagged["p_value_unclustered"]


def test_agreeing_rows_are_not_flagged():
    result = _run(_paired_cohort(path_a=1.0, path_b=1.0))
    acme = _term(result, "acme")
    assert bool(acme["clustering_changes_the_call"]) is False
    assert acme["reason"] == ""


# --------------------------------------------------------------------------- #
# guard 3: a mediated fraction needs a total effect to be a fraction of
# --------------------------------------------------------------------------- #


def test_the_proportion_is_withheld_when_the_total_effect_straddles_zero():
    """The reference table's "-0.2121, CI (-9.825, 6.150)" case.

    A ratio whose denominator crosses zero is not a small effect with a wide
    interval; it is undefined. The row stays (so the table has all six terms) but the
    numbers are withheld and the reason is stated.
    """
    # No treatment effect at all on either path: the total effect is null.
    result = _run(_paired_cohort(path_a=0.0, path_b=1.0, direct=0.0, seed=5))

    total = _term(result, "total")
    assert total["ci_low"] < 0 < total["ci_high"], "fixture should have a null total effect"

    proportion = _term(result, "proportion_mediated")
    assert np.isnan(proportion["estimate"])
    assert np.isnan(proportion["ci_low"]) and np.isnan(proportion["ci_high"])
    assert np.isnan(proportion["p_value"])
    assert "not interpretable" in proportion["reason"]
    assert "ACME" in proportion["reason"]

    # And the ACME itself survives: it is still interpretable in its own units.
    assert np.isfinite(_term(result, "acme")["estimate"])


def test_the_proportion_is_reported_when_the_total_effect_is_clear():
    result = _run(_paired_cohort(path_a=1.0, path_b=1.0, direct=1.0))
    proportion = _term(result, "proportion_mediated")
    assert np.isfinite(proportion["estimate"])
    assert proportion["reason"] == ""


# --------------------------------------------------------------------------- #
# guard 4: mediator and outcome must not be built from the same genes
# --------------------------------------------------------------------------- #


def test_disjoint_gene_sets_are_graded_disjoint():
    result = _run(
        _paired_cohort(),
        mediator_features=["CLDN5", "CDH5"],
        outcome_features=["SNAI2", "TWIST1"],
    )
    assert set(result["circularity"]) == {"disjoint"}
    assert set(result["shared_features"]) == {0}


def test_a_partial_gene_overlap_is_graded_overlapping():
    result = _run(
        _paired_cohort(),
        mediator_features=["CLDN5", "CDH5", "TJP1", "OCLN"],
        outcome_features=["SNAI2", "TWIST1", "ACTA2", "CLDN5"],
    )
    assert set(result["circularity"]) == {"overlapping"}
    assert set(result["shared_features"]) == {1}
    assert 0.0 < result["feature_overlap"].iloc[0] < 1.0


def test_a_mediator_mostly_inside_the_outcome_is_graded_nested():
    # Path b is then partly definitional: the mediator largely IS a piece of the
    # outcome, so it would predict it even with no biology involved.
    result = _run(
        _paired_cohort(),
        mediator_features=["SNAI2", "TWIST1"],
        outcome_features=["SNAI2", "TWIST1", "ACTA2", "TAGLN"],
    )
    assert set(result["circularity"]) == {"nested"}


def test_omitting_the_gene_sets_leaves_the_grade_blank_rather_than_claiming_disjoint():
    # Not knowing the overlap and knowing it is zero are different states, and the
    # table must not report the reassuring one when it was never given the inputs.
    result = _run(_paired_cohort())
    assert result["circularity"].isna().all()
    assert result["shared_features"].isna().all()


# --------------------------------------------------------------------------- #
# refusals: a fit that cannot be believed is reported, not returned quietly
# --------------------------------------------------------------------------- #


def test_too_few_donors_is_refused_with_the_reason_recorded():
    result = _run(_paired_cohort(n_donors=3))
    assert len(result) == len(MEDIATION_TERMS)
    assert result["estimate"].isna().all()
    assert result["n_donors"].iloc[0] == 3
    assert "donors contribute" in result["reason"].iloc[0]


def test_a_one_armed_cohort_is_refused_rather_than_fitted():
    cells = _paired_cohort()
    only_disease = cells[cells["condition"] == "Disease"]
    result = _run(only_disease)
    assert result["estimate"].isna().all()
    assert "fewer than 2 samples" in result["reason"].iloc[0]


def test_a_treatment_column_with_neither_level_raises():
    cells = _paired_cohort().assign(condition="Something else")
    with pytest.raises(ValueError, match="holds neither"):
        _run(cells)


def test_an_absent_column_raises_rather_than_silently_dropping_it():
    with pytest.raises(KeyError, match="columns absent"):
        _run(_paired_cohort(), covariates=["not_a_column"])


# --------------------------------------------------------------------------- #
# groups and covariates
# --------------------------------------------------------------------------- #


def test_groups_are_fitted_separately():
    """A mediation that holds in one subtype and not another must not be averaged."""
    mediated = _paired_cohort(path_a=1.0, path_b=1.0, direct=0.0, seed=1).assign(subtype="A")
    unmediated = _paired_cohort(path_a=0.0, path_b=1.0, direct=2.0, seed=2).assign(subtype="B")
    # Distinct sample/donor ids per subtype, as separate donors would be.
    unmediated = unmediated.assign(
        sample=unmediated["sample"] + "_B", donor=unmediated["donor"] + "_B"
    )
    result = _run(pd.concat([mediated, unmediated], ignore_index=True), group="subtype")

    assert set(result["group"]) == {"A", "B"}
    assert _term(result, "acme", group="A")["p_value"] < 0.05
    assert _term(result, "acme", group="B")["p_value"] > 0.05


def test_a_cell_level_group_is_aggregated_per_sample_within_group():
    """A subtype lives on cells, so one sample contributes to every subtype it holds.

    Keying the aggregation on the sample alone would either refuse the fit or average
    a subtype together with its neighbours. Keying on sample-within-subtype is what
    makes "does this mediation hold in this subtype" answerable at all.
    """
    cells = _paired_cohort(cells_per_sample=40, seed=8)
    # Split every sample's cells across two subtypes: within-sample, not between.
    cells = cells.assign(
        subtype=np.where(cells.groupby("sample").cumcount() % 2 == 0, "capillary", "collecting")
    )
    result = _run(cells, group="subtype")

    assert set(result["group"]) == {"capillary", "collecting"}
    for subtype in ("capillary", "collecting"):
        block = result[result["group"] == subtype]
        # Each subtype still sees all eighteen samples and all nine donors.
        assert block["n_samples"].iloc[0] == 18
        assert block["n_donors"].iloc[0] == 9
        assert np.isfinite(_term(result, "acme", group=subtype)["estimate"])


def test_a_donor_inconsistency_is_still_caught_under_a_cell_level_group():
    cells = _paired_cohort(cells_per_sample=10, seed=8)
    cells = cells.assign(subtype="one")
    cells.loc[cells.index[0], "donor"] = "SOMEONE_ELSE"
    with pytest.raises(ValueError, match="does not identify a sample"):
        _run(cells, group="subtype")


def test_a_group_below_the_donor_floor_is_refused_without_losing_the_others():
    big = _paired_cohort(n_donors=9, seed=1).assign(subtype="big")
    small = _paired_cohort(n_donors=2, seed=2).assign(subtype="small")
    small = small.assign(sample=small["sample"] + "_s", donor=small["donor"] + "_s")
    result = _run(pd.concat([big, small], ignore_index=True), group="subtype")

    assert np.isfinite(_term(result, "acme", group="big")["estimate"])
    assert np.isnan(_term(result, "acme", group="small")["estimate"])
    assert "donors contribute" in _term(result, "acme", group="small")["reason"]


def test_a_covariate_is_entered_in_both_models():
    """Adjustment must change the fit, and adjusting for noise must not break it.

    The estimate is not asserted to be unchanged: at eighteen samples one extra
    column genuinely moves it, and pretending otherwise would only be testing a
    tolerance. What is asserted is that the covariate reached both models — a
    covariate silently dropped would return the identical number.
    """
    cells = _paired_cohort(seed=4)
    without = _term(_run(cells), "acme")["estimate"]
    adjusted = _term(_run(cells, covariates=["depth"]), "acme")["estimate"]

    assert adjusted != without
    # Still the same mechanism, not a different answer: a noise covariate cannot
    # abolish a fully-mediated effect.
    assert adjusted == pytest.approx(without, rel=0.5)


def test_a_covariate_collinear_with_the_treatment_is_refused_not_fitted():
    # Duplicating the treatment leaves path a unidentified; reporting whatever
    # least-squares returned from a rank-deficient design would be a wrong answer,
    # not a wide one.
    cells = _paired_cohort().assign(
        duplicate=lambda frame: (frame["condition"] == "Disease").astype(float)
    )
    result = _run(cells, covariates=["duplicate"])
    assert result["estimate"].isna().all()
    assert "not identified" in result["reason"].iloc[0]


# --------------------------------------------------------------------------- #
# the grid: several candidate mediators of one outcome are one test family
# --------------------------------------------------------------------------- #


def test_the_grid_stacks_candidates_and_corrects_across_the_acme_family():
    cells = _paired_cohort(path_a=1.0, path_b=1.0, seed=6)
    # A second candidate that carries no signal, as a decoy.
    rng = np.random.default_rng(0)
    cells = cells.assign(decoy_score=rng.normal(size=len(cells)))

    result = mediation_grid(
        cells,
        mediators=["mediator_score", "decoy_score"],
        outcome="outcome_score",
        sample="sample",
        donor="donor",
        treatment="condition",
        case="Disease",
        control="Normal",
        n_boot=N_BOOT,
        n_sims=N_SIMS,
    )

    assert set(result["mediator"]) == {"mediator_score", "decoy_score"}
    assert len(result) == 2 * len(MEDIATION_TERMS)

    acme = result[result["term"] == "acme"]
    assert acme["fdr"].notna().all()
    # Only the ACME rows are corrected: the other terms are components of the same
    # decomposition, not independent hypotheses.
    assert result[result["term"] != "acme"]["fdr"].isna().all()

    real = acme[acme["mediator"] == "mediator_score"].iloc[0]
    decoy = acme[acme["mediator"] == "decoy_score"].iloc[0]
    assert real["fdr"] < 0.05 < decoy["fdr"]


def test_the_grid_passes_each_mediators_own_gene_set_to_the_circularity_grade():
    cells = _paired_cohort().assign(decoy_score=0.0)
    result = mediation_grid(
        cells,
        mediators=["mediator_score", "decoy_score"],
        outcome="outcome_score",
        features={
            "mediator_score": ["CLDN5", "CDH5"],
            "decoy_score": ["SNAI2", "TWIST1"],
            "outcome_score": ["SNAI2", "TWIST1", "ACTA2"],
        },
        sample="sample",
        donor="donor",
        treatment="condition",
        case="Disease",
        control="Normal",
        n_boot=N_BOOT,
        n_sims=N_SIMS,
    )
    grades = result.drop_duplicates("mediator").set_index("mediator")["circularity"]
    assert grades["mediator_score"] == "disjoint"
    assert grades["decoy_score"] == "nested"
