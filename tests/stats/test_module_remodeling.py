"""Synthetic-fixture unit tests for the reusable module-remodeling stats.

Pure numpy/pandas fixtures built inline — no AnnData, no Rscript, no external data,
no skip markers. These always run and never trip the real-data skipif leak
(templated on test_mcp_diagnostics.py). They pin the statistical contract the
LEC manuscript depends on: donor-aware effect sizes, a multivariate PERMANOVA,
signature-argmax subtyping with an ambiguity guard, the signed EndoMT-style
contrast index, leading-edge concordance, and program correlations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from cellquorum.stats.module_remodeling import (
    _mixedlm_effect,
    _sign_flip_p,
    bh_fdr,
    declared_panel_membership,
    fdr_floor_reachability,
    leading_edge_jaccard,
    lmm_effect_sizes,
    module_gene_overlap,
    permanova_by_group,
    program_correlation_matrix,
    randomization_floor,
    recorrect_within_family,
    signature_argmax_labels,
    signed_program_contrast_index,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _paired_frame(effect: float, *, n_donors: int = 6, n_cells: int = 8, seed: int = 0):
    """Build a paired (donor × condition) per-cell score frame for ONE program.

    Each donor contributes ``n_cells`` cells per arm. Within a donor the SAME
    noise vector is mirrored across the two arms, so the only condition
    difference is the planted ``effect`` (added to every case cell). This makes
    the null (effect=0) condition contrast exactly ~0 with non-zero residual
    variance — deterministic, not flaky.
    """
    rng = np.random.default_rng(seed)
    rows = []
    idx = []
    k = 0
    for d in range(n_donors):
        donor = f"D{d}"
        base = rng.normal(0.0, 1.0)  # donor random intercept
        noise = rng.normal(0.0, 0.4, size=n_cells)  # mirrored across arms
        for cond, bump in (("Normal", 0.0), ("Lymphedema", effect)):
            for j in range(n_cells):
                rows.append(
                    {
                        "score": base + noise[j] + bump,
                        "donor_id": donor,
                        "condition": cond,
                        "sample_id": f"{donor}_{cond}",
                        "subtype": "CV",
                    }
                )
                idx.append(f"c{k}")
                k += 1
    meta = pd.DataFrame(rows, index=idx)
    scores = pd.DataFrame({"prog": meta.pop("score")}, index=idx)
    return scores, meta


# --------------------------------------------------------------------------- #
# bh_fdr                                                                        #
# --------------------------------------------------------------------------- #
def test_bh_fdr_matches_statsmodels_on_finite_and_passes_nan_through():
    p = np.array([0.001, 0.2, np.nan, 0.04, 0.5])
    out = bh_fdr(p)
    assert np.isnan(out[2])  # NaN position preserved, not corrected
    finite = np.array([0.001, 0.2, 0.04, 0.5])
    expected = multipletests(finite, method="fdr_bh")[1]
    got = out[[0, 1, 3, 4]]
    assert np.allclose(got, expected)


# --------------------------------------------------------------------------- #
# signature_argmax_labels                                                       #
# --------------------------------------------------------------------------- #
def test_signature_argmax_labels_assigns_dominant_and_guards_ambiguous():
    # Four CHOIR clusters; each of c0/c1/c2 dominant in one signature; c3 is a
    # perfect tie between sig_a and sig_b -> must be left 'unassigned'.
    cell_clusters = {
        "c0": (1.0, 0.0, 0.0),
        "c1": (0.0, 1.0, 0.0),
        "c2": (0.0, 0.0, 1.0),
        "c3": (0.5, 0.5, 0.0),
    }
    rows, clusters, idx = [], [], []
    n = 0
    for cl, (a, b, c) in cell_clusters.items():
        for _ in range(3):  # 3 cells/cluster; mean == the planted value
            rows.append({"sig_a": a, "sig_b": b, "sig_c": c})
            clusters.append(cl)
            idx.append(f"x{n}")
            n += 1
    scores = pd.DataFrame(rows, index=idx)
    cluster_labels = pd.Series(clusters, index=idx)

    out = signature_argmax_labels(scores, cluster_labels, min_margin=0.1)
    lab = out.set_index("cluster")["label"].to_dict()
    assert lab["c0"] == "sig_a"
    assert lab["c1"] == "sig_b"
    assert lab["c2"] == "sig_c"
    assert lab["c3"] == "unassigned"


# --------------------------------------------------------------------------- #
# signed_program_contrast_index (EndoMT index generalization)                   #
# --------------------------------------------------------------------------- #
def test_signed_contrast_index_orders_along_planted_axis_and_centers():
    # Cells arranged on a monotone axis: up programs rise, down program falls.
    n = 20
    t = np.linspace(0, 1, n)
    scores = pd.DataFrame(
        {
            "endomt_lec": t,
            "mesenchymal_gain": t,
            "lec_identity": 1.0 - t,
        },
        index=[f"c{i}" for i in range(n)],
    )
    idx = signed_program_contrast_index(
        scores, up=["endomt_lec", "mesenchymal_gain"], down=["lec_identity"]
    )
    # Monotone increasing along the axis; the mesenchymal end > the LEC end.
    assert idx.iloc[-1] > idx.iloc[0]
    assert np.all(np.diff(idx.values) > 0)
    # Standardized contrast is mean-centered.
    assert abs(float(idx.mean())) < 1e-9


# --------------------------------------------------------------------------- #
# lmm_effect_sizes                                                              #
# --------------------------------------------------------------------------- #
def test_lmm_effect_sizes_recovers_planted_positive_effect():
    scores, meta = _paired_frame(effect=1.5, seed=1)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["program"] == "prog" and row["group"] == "CV"
    assert row["effect"] > 1.0  # near the planted +1.5
    assert row["ci_low"] > 0.0  # CI excludes 0
    assert row["p_value"] < 0.05
    assert row["fdr"] < 0.05
    assert row["method"] == "lmm"
    assert row["n_donors"] == 6


def test_lmm_effect_sizes_null_effect_is_nonsignificant_and_near_zero():
    scores, meta = _paired_frame(effect=0.0, seed=2)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert abs(row["effect"]) < 0.1  # mirrored noise -> ~0 contrast
    assert row["p_value"] > 0.05


def test_lmm_effect_sizes_claims_no_method_when_a_single_donor_supports_none():
    """One donor total: nothing is estimable, and the row must say exactly that.

    The guard has to record the outcome rather than crash — but it also must not
    name a test that never ran. A single paired donor gives a paired t-test zero
    degrees of freedom, so this row used to come back all-NaN under
    ``method='paired_t'``: a label asserting a test produced this blank. The
    honest record is ``method='none'`` plus a reason.
    """
    scores, meta = _paired_frame(effect=1.0, n_donors=1, seed=3)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        min_donors_per_arm=2,
    )
    row = out.iloc[0]
    assert row["method"] == "none"
    assert not np.isfinite(row["effect"])
    assert row["reason"]


# --------------------------------------------------------------------------- #
# lmm_effect_sizes — the three ways a real effect became an unexplained blank  #
# --------------------------------------------------------------------------- #
def _unpaired_frame(
    effect: float, *, n_donors: int = 8, n_cells: int = 8, donor_sd: float = 0.0, seed: int = 0
):
    """Independent donors per arm: no donor appears in both conditions.

    The common cohort shape that is NOT the LEC manuscript's paired design, and
    the shape a *subtype* stratum collapses to whenever a CHOIR cluster happens
    to draw its case and control cells from disjoint donors. ``donor_sd=0``
    is the hard case on purpose: with no between-donor spread the mixed model's
    random-intercept variance sits on the boundary at zero.
    """
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    k = 0
    for d in range(n_donors):
        base = rng.normal(0.0, donor_sd)
        cond, bump = ("Lymphedema", effect) if d < n_donors // 2 else ("Normal", 0.0)
        for _ in range(n_cells):
            rows.append(
                {
                    "score": base + rng.normal(0.0, 0.4) + bump,
                    "donor_id": f"D{d}",
                    "condition": cond,
                    "sample_id": f"D{d}_{cond}",
                    "subtype": "CV",
                }
            )
            idx.append(f"c{k}")
            k += 1
    meta = pd.DataFrame(rows, index=idx)
    scores = pd.DataFrame({"prog": meta.pop("score")}, index=idx)
    return scores, meta


def test_lmm_effect_sizes_does_not_turn_an_unpaired_design_into_a_silent_null():
    """A planted +1.5 across 8 donors must not come back as a blank row.

    Reproduced on the shipped implementation: with independent donors and no
    between-donor spread, the mixed model is rejected (boundary fit) and the
    fallback is a *paired* t-test — which has no donor present in both arms to
    pair, so it returns NaN. The row is then labelled ``method='paired_t'`` with
    no effect, no p, and nothing saying why. That is the false-null failure mode
    the paired-DE lesson was about, arriving through the fallback instead of the
    design: a real effect rendered as an empty cell in the flagship dot-grid.

    The donor-level fallback has to match the design it is given — unpaired
    donors get an unpaired (Welch) test on per-donor means, which still collapses
    pseudoreplication to one value per donor and still gives a real answer.
    """
    scores, meta = _unpaired_frame(effect=1.5, donor_sd=0.0, seed=5)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["method"] != "paired_t", "no donor spans both arms; nothing can be paired"
    assert row["method"] in {"lmm", "welch_t"}
    assert np.isfinite(row["effect"]), f"planted +1.5 came back blank ({row.to_dict()})"
    assert row["effect"] > 0.5
    assert np.isfinite(row["p_value"]) and row["p_value"] < 0.05


def test_mixedlm_effect_keeps_a_nonconverged_fit_instead_of_discarding_it():
    """A non-converged optimizer is not the same thing as an unusable fit.

    Stopping the optimizer early (``maxiter=1``) leaves ``converged=False`` with a
    fixed effect, a standard error and a p-value that are all finite and, on this
    fixture, numerically indistinguishable from the fully converged answer. The
    shipped code gated on ``converged`` and threw exactly this away, silently
    substituting a t-test on donor means for the pseudoreplication-aware model
    the caller asked for — visible only as a changed value in ``method``.

    Keeping the fit obliges us to disclose it, which is what the ``converged``
    column is for. Tested through the helper because the public function
    deliberately exposes no iteration knob.
    """
    scores, meta = _paired_frame(effect=1.5, n_donors=6, seed=4)
    df = pd.DataFrame(
        {
            "score": scores["prog"].to_numpy(dtype=float),
            "cond": (meta["condition"] == "Lymphedema").to_numpy().astype(float),
            "donor": meta["donor_id"].to_numpy(),
        }
    )
    stopped_early = _mixedlm_effect(df, maxiter=1)
    assert stopped_early is not None, "a finite fixed effect must not be discarded"
    effect, ci_low, ci_high, p_value, converged = stopped_early
    assert converged is False, "a non-converged fit must be recorded, not laundered"
    assert abs(effect - 1.5) < 0.05
    assert np.isfinite(p_value) and p_value < 0.05
    assert ci_low < effect < ci_high


def test_a_flattened_donor_design_is_carried_by_the_sample_component():
    """Removing the donor means no longer makes the fit impossible — and must not.

    History: with only a donor random intercept, flattening the donor means left no
    between-donor variance for it to estimate, the Hessian was singular, ``fit()``
    raised, and the row fell through to a degenerate paired t-test that the shipped
    code labelled ``p_value = 0.0``. That fallback is still guarded (see
    ``test_lmm_effect_sizes_never_reports_a_p_value_of_exactly_zero``), but this
    fixture no longer reaches it: the sample variance component is estimable here
    even when the donor one is not, so the model is fitted.

    Which puts the burden on the other guard. Every paired difference in this
    fixture is exactly the planted 1.5, so there is no sample-level spread either,
    and the model's p-value comes out around 1e-56 — from six donor pairs, whose
    randomization set cannot go below ``2/2**6``. That number must not be quotable,
    and what makes it not quotable is the design floor being reported beside it.
    """
    scores, meta = _paired_frame(effect=1.5, n_donors=6, seed=4)
    scores = scores.copy()
    donor_mean = scores["prog"].groupby(meta["donor_id"]).transform("mean")
    scores["prog"] = scores["prog"] - donor_mean + scores["prog"].mean()

    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["method"] == "lmm"
    assert row["variance_components"] == "donor+sample"
    assert abs(row["effect"] - 1.5) < 1e-6, "the paired difference is exact; report it"
    assert row["n_pairs"] == 6
    assert row["design_floor_p"] == pytest.approx(2 / 2**6)
    assert row["p_below_design_floor"], (
        f"p={row['p_value']:.3g} is below the floor {row['design_floor_p']:.4g} this design can "
        "reach and must be flagged as resting on the model rather than the cohort"
    )
    # And the assumption-free answer is right there: six pairs all moving one way.
    assert row["donor_p"] == pytest.approx(2 / 2**6)
    assert row["donor_test"] == "sign_flip_exact"


# --------------------------------------------------------------------------- #
# the level condition varies at                                                #
# --------------------------------------------------------------------------- #
def _sample_noise_frame(
    effect: float,
    *,
    n_donors: int = 8,
    n_cells: int = 40,
    sample_sd: float = 1.2,
    cell_sd: float = 0.5,
    seed: int = 0,
):
    """A paired cohort with real sample-to-sample variability, and a known answer.

    ``_paired_frame`` mirrors each donor's noise across the arms, so its two samples
    per donor differ by exactly the planted effect and there is no sample-level
    spread at all. That is the one shape in which counting cells as replicates does
    no damage, and it is not the shape of a real cohort.

    Here each donor-arm *sample* carries its own offset on top of the donor's, and
    cells vary around that. This is what a captured library looks like: two libraries
    from the same donor differ from each other for reasons that have nothing to do
    with condition, and no number of cells inside one library says anything about how
    big that difference is.

    The offsets are drawn at random and then centred and rescaled, so the eight
    paired differences have mean exactly ``effect`` and standard deviation exactly
    ``sample_sd * sqrt(2)``. That makes the donor-level answer arithmetic rather than
    seed luck: at ``effect=1.0``, ``sample_sd=1.2`` and eight pairs, ``t = 1.667`` on
    7 df and ``p = 0.14`` for every seed. The shape of the noise is still random; only
    its first two moments are pinned.
    """
    rng = np.random.default_rng(seed)
    donor_offsets = rng.normal(0.0, 1.0, n_donors)
    raw = rng.normal(0.0, 1.0, n_donors)
    raw = raw - raw.mean()
    # sd of a paired difference when both arms carry an independent offset of sd
    # ``sample_sd``. Splitting it symmetrically between the arms keeps each sample's
    # own offset a real, per-sample quantity.
    delta = raw / raw.std(ddof=1) * (sample_sd * np.sqrt(2.0))

    rows, idx = [], []
    k = 0
    for d in range(n_donors):
        donor = f"D{d}"
        for cond, bump, half in (
            ("Normal", 0.0, -delta[d] / 2),
            ("Lymphedema", effect, +delta[d] / 2),
        ):
            sample_offset = donor_offsets[d] + half
            for _ in range(n_cells):
                rows.append(
                    {
                        "score": sample_offset + bump + rng.normal(0.0, cell_sd),
                        "donor_id": donor,
                        "condition": cond,
                        "sample_id": f"{donor}_{cond}",
                        "subtype": "CV",
                    }
                )
                idx.append(f"c{k}")
                k += 1
    meta = pd.DataFrame(rows, index=idx)
    scores = pd.DataFrame({"prog": meta.pop("score")}, index=idx)
    return scores, meta


def test_a_within_donor_contrast_is_tested_at_the_sample_not_the_cell():
    """The donor intercept alone counts cells as replicates of a sample's condition.

    Found on real data: ``score ~ condition + (1|donor)`` returned ``p = 3e-47`` for
    an NF-kB footprint across eight donor pairs in which exactly one donor moved in
    the reported direction. The donor random intercept absorbs the donor *mean*, and
    on a paired design that is not where condition lives — condition is assigned to
    the sample, so the sample is the replicate. With ~130 cells per sample the
    standard error came out roughly eleven times too small.

    This fixture makes the truth known: the planted effect is 1.0 against a
    sample-level spread of 1.2, which eight pairs cannot resolve. The
    donor-intercept-only model is asserted to be *wrong* on it, because a test that
    only checked the fixed model would pass on a fixture where both models agree.
    """
    scores, meta = _sample_noise_frame(effect=1.0, seed=11)

    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        sample_col="sample_id",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["method"] == "lmm"
    assert row["variance_components"] == "donor+sample"

    # The model with the sample component agrees with the design: no finding here.
    assert row["p_value"] > 0.05, (
        f"a 1.0 effect against a 1.2 sample-level spread over 8 pairs is not detectable, "
        f"but the model reported p={row['p_value']:.3g}"
    )
    assert not row["p_below_design_floor"]
    assert row["donor_p"] > 0.05, "the design's own randomization test finds nothing either"

    # And the model that was shipped is wildly significant on exactly this data.
    # This is the assertion that fails if the variance component is ever dropped
    # again. Note it is compared *with* the same small-sample df correction the fixed
    # model gets, so what is left is purely the standard error: the two fixes are
    # independent, and neither alone is enough. At the normal reference statsmodels
    # uses by default this same fit reads p ~ 1e-46.
    df = pd.DataFrame(
        {
            "score": scores["prog"].to_numpy(dtype=float),
            "cond": (meta["condition"] == "Lymphedema").to_numpy().astype(float),
            "donor": meta["donor_id"].to_numpy(),
            "sample": meta["sample_id"].to_numpy(),
        }
    )
    donor_only = _mixedlm_effect(df, nested=False)
    assert donor_only is not None
    assert donor_only[3] < 1e-4, (
        "the fixture must reproduce the bug for the fix to be worth testing; "
        f"donor-intercept-only p={donor_only[3]:.3g}"
    )
    assert donor_only[3] < row["p_value"] / 1e3, (
        "the sample component has to move the p-value by orders of magnitude here, "
        f"not trim it: {donor_only[3]:.3g} vs {row['p_value']:.3g}"
    )


def test_an_unpaired_design_keeps_the_donor_intercept_as_its_replicate_level():
    """No second component when condition is assigned between donors.

    With one sample per donor the sample and the donor are the same unit, so a
    sample variance component would be collinear with the donor intercept — and
    unnecessary, because the donor intercept is already sitting at the level
    condition varies at. Adding one everywhere would make the common unpaired
    cohort unfittable for no gain.
    """
    scores, meta = _unpaired_frame(effect=1.5, donor_sd=0.6, seed=5)

    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        sample_col="sample_id",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["variance_components"] == "donor"
    assert row["n_pairs"] == 0, "no donor spans both arms"
    # Four donors per arm: C(8,4) = 70 assignments, so 2/70 is the floor.
    assert row["design_floor_p"] == pytest.approx(2 / 70)
    assert row["donor_test"] == "label_perm_exact"


def test_the_randomization_floor_follows_the_design_not_the_cell_count():
    """Unit-test the floor itself on the three design shapes.

    It is a combinatorial property of who was assigned what, and it has to be
    computed from the design rather than from a rule of thumb: a paired cohort and
    an unpaired cohort of the same size have floors that differ by more than
    threefold at n=8, and quoting the wrong one either excuses a bad p-value or
    condemns a good one.
    """
    # Eight complete pairs: 2**8 sign assignments.
    donors = [f"D{d}" for d in range(8) for _ in range(2)]
    is_case = [c == 0 for _ in range(8) for c in range(2)]
    floor, n_pairs = randomization_floor(donors, is_case)
    assert n_pairs == 8
    assert floor == pytest.approx(2 / 2**8)

    # Eight donors, four per arm, none shared: C(8,4) = 70 assignments.
    floor, n_pairs = randomization_floor([f"D{d}" for d in range(8)], [True] * 4 + [False] * 4)
    assert n_pairs == 0
    assert floor == pytest.approx(2 / 70)

    # Mixed: three pairs plus two case-only and two control-only donors.
    donors = ["P0", "P0", "P1", "P1", "P2", "P2", "C0", "C1", "N0", "N1"]
    is_case = [True, False] * 3 + [True, True, False, False]
    floor, n_pairs = randomization_floor(donors, is_case)
    assert n_pairs == 3
    assert floor == pytest.approx(2 / (2**3 * 6))

    # One pair cannot produce a significant randomization p at all.
    floor, n_pairs = randomization_floor(["D0", "D0"], [True, False])
    assert (floor, n_pairs) == (1.0, 1)


def test_the_floor_makes_an_isolated_fdr_call_unreachable_in_a_wide_family():
    """A floor that looks survivable alone is not survivable inside a family.

    The number this protects against: eight donor pairs reach 0.0078, which reads as
    fine against 0.05, so a table of 45 tests looks correctable. It is not — BH's
    threshold for the smallest p-value in a 45-test family is 0.05/45 = 0.0011, well
    under the floor — so significance is available only to a *block* of at least eight
    sources moving together, and a single strong footprint cannot be called however
    large it is. Reporting "nothing survived correction" without this makes an
    arithmetic impossibility read as an absence of signal.
    """
    floor_8, _ = randomization_floor(
        [f"D{d}" for d in range(8) for _ in range(2)],
        [c == 0 for _ in range(8) for c in range(2)],
    )
    min_concordant, reachable = fdr_floor_reachability(floor_8, 45)
    assert (min_concordant, reachable) == (8, True)
    # 0.05 * 8 / 45 clears the floor; 0.05 * 7 / 45 does not. The boundary is the claim.
    assert 0.05 * min_concordant / 45 >= floor_8
    assert 0.05 * (min_concordant - 1) / 45 < floor_8

    # Seven pairs -- the reference's CV column -- needs a third of the family at once.
    floor_7, _ = randomization_floor(
        [f"D{d}" for d in range(7) for _ in range(2)],
        [c == 0 for _ in range(7) for c in range(2)],
    )
    assert fdr_floor_reachability(floor_7, 45) == (15, True)

    # A single test is corrected against itself, so the floor is the only bar.
    assert fdr_floor_reachability(floor_8, 1) == (1, True)

    # One pair has a floor of 1.0: no family size makes any of it reachable.
    floor_1, _ = randomization_floor(["D0", "D0"], [True, False])
    min_concordant, reachable = fdr_floor_reachability(floor_1, 45)
    assert min_concordant == 900
    assert not reachable

    # Degenerate inputs say "nothing is reachable" rather than raising.
    assert fdr_floor_reachability(np.nan, 45) == (0, False)
    assert fdr_floor_reachability(0.0078, 0) == (0, False)


def test_donor_p_is_the_exact_sign_flip_distribution():
    """The companion p-value is the paired design's own reference set, enumerated.

    Nine donors all moving the same way gives 2/2**9 whether they moved by 0.09 or
    by 1.81 — that is what a randomization test on signs measures, and printing it
    next to the model's p-value is what keeps a consistent-but-tiny shift from being
    read as a large one.
    """
    # Every difference the same sign -> only the two extreme sign assignments are as
    # extreme as the observed one.
    p, test = _sign_flip_p(np.arange(1.0, 10.0), seed=0)
    assert test == "sign_flip_exact"
    assert p == pytest.approx(2 / 2**9)

    # One donor against the other eight, and the total is no longer extreme.
    diff = np.array([-3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    p, _ = _sign_flip_p(diff, seed=0)
    assert p > 2 / 2**8

    # A perfectly balanced set cannot be significant at all.
    p, _ = _sign_flip_p(np.array([1.0, -1.0, 2.0, -2.0]), seed=0)
    assert p == 1.0


def test_the_concordant_donor_count_separates_a_cohort_shift_from_one_donor():
    """Two rows can share a p-value and a sign and mean opposite things.

    ``donor_p`` is a function of the sign pattern and ``effect`` is a function of the
    magnitudes, so neither on its own distinguishes "every donor moved a little" from
    "one donor moved a lot and the rest sat still" — and those are not the same claim
    about a disease. The concordant count is the column that separates them, and it
    needs no distributional assumption to be read.
    """
    scores, meta = _paired_frame(0.5, n_donors=9, n_cells=6, seed=3)
    table = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        sample_col="sample_id",
    )
    row = table.iloc[0]
    # The fixture adds the same bump to every case cell of every donor, so all nine
    # pairs move up and the count is the pair count.
    assert row["n_pairs"] == 9
    assert row["n_donors_concordant"] == 9
    assert row["donor_p"] == pytest.approx(2 / 2**9)

    # One donor carrying the whole effect reaches a comparable mean difference and a
    # far weaker sign pattern. The count is what makes the difference legible.
    diff_all_up = np.full(9, 0.5)
    diff_one_donor = np.array([4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert diff_all_up.mean() == pytest.approx(diff_one_donor.mean())
    assert int(np.sum(np.sign(diff_all_up) == np.sign(diff_all_up.mean()))) == 9
    assert int(np.sum(np.sign(diff_one_donor) == np.sign(diff_one_donor.mean()))) == 1

    # An unpaired stratum has no pairs to count, which is not the same as zero of them:
    # reporting 0 there would read as "no donor agreed".
    scores, meta = _paired_frame(0.4, n_donors=6, n_cells=5, seed=4)
    case_donors = {"D0", "D1", "D2"}
    keep = meta.apply(
        lambda r: (r["condition"] == "Lymphedema") == (r["donor_id"] in case_donors),
        axis=1,
    )
    table = lmm_effect_sizes(
        scores.loc[keep],
        meta.loc[keep],
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = table.iloc[0]
    assert row["n_pairs"] == 0
    assert np.isnan(row["n_donors_concordant"])
    assert row["donor_test"].startswith("label_perm")


def test_a_program_that_never_moved_is_not_unanimous():
    """Zero in both arms has no direction, so no donor can agree with it.

    Found on real data, not constructed: a NicheNet run tested a ligand no
    neutrophil expressed in any donor. Every paired difference was exactly 0, so
    ``sign(0) == sign(0)`` held nine times and the row reported **9 of 9 donors
    concordant** — the strongest-looking value in the column, on the one row that
    measured nothing at all. A caller reading concordance as evidence (which is
    the column's whole purpose) would promote it.

    The effect and the p-value already say nothing happened; the concordance
    count has to agree with them, because a column that is right except on its
    most extreme value cannot be used unsupervised.
    """
    scores, meta = _paired_frame(0.5, n_donors=9, n_cells=4, seed=11)
    scores["unexpressed"] = 0.0  # the program no cell expresses, in either arm

    table = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    flat = table[table["program"] == "unexpressed"].iloc[0]

    assert flat["n_pairs"] == 9
    assert flat["donor_effect"] == pytest.approx(0.0)
    assert flat["n_donors_concordant"] == 0

    # The program that did move is unaffected: the guard is on a zero mean
    # difference, not on the presence of tied donors.
    moved = table[table["program"] == "prog"].iloc[0]
    assert moved["n_donors_concordant"] == 9


def test_lmm_effect_sizes_never_reports_a_p_value_of_exactly_zero():
    """p == 0 is a reporting bug, and it reaches the FDR family as a zero.

    Reproduced on the shipped implementation: mirrored per-donor noise makes
    every paired delta identical, the paired t-statistic is infinite, and the
    fallback reports ``p_value = 0.0`` — which BH-corrects to ``fdr = 0.0`` and
    would be typeset as "p = 0" in a manuscript table. Zero variance among the
    paired deltas means the test is *undefined*, not infinitely significant. The
    effect size is still real and still reported; only the p-value is withheld,
    with the degeneracy recorded.

    ``min_donors_per_arm`` is set past the donor count to force the fallback
    path deterministically through the public API.
    """
    scores, meta = _paired_frame(effect=1.5, n_donors=6, seed=6)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        min_donors_per_arm=99,  # skip the LMM; exercise the donor-level fallback
    )
    row = out.iloc[0]
    assert row["method"] == "paired_t"
    assert abs(row["effect"] - 1.5) < 1e-6, "the effect size is well defined; report it"
    assert not (row["p_value"] == 0.0), "a t-test on constant deltas is undefined, not p=0"
    assert not np.isfinite(row["p_value"]), "an undefined test withholds its p-value"
    assert not (row["fdr"] == 0.0)
    assert isinstance(row["reason"], str) and row["reason"], "record why p is missing"
    assert "floating-point" in row["reason"]


def test_lmm_effect_sizes_withholds_p_when_the_delta_spread_is_only_rounding_noise():
    """Whether the identical deltas leave sd == 0 or sd == 1e-16 is seed luck.

    ``_paired_frame`` mirrors each donor's noise across the two arms, so every
    paired difference is exactly the planted effect — mathematically. In double
    precision the summation order decides whether the residual spread comes out
    as 0.0 (seeds 1 and 4) or as ~1e-16 (seed 6), and an exact-zero guard only
    catches the first. The second divides through to a t statistic of 1e16 and a
    p-value near 1e-82, which would be typeset as the strongest result in the
    table on the strength of rounding error.

    So the guard is relative: a spread at the floating-point noise floor of the
    values is not a measured spread, whichever way the arithmetic landed.
    """
    for seed in (1, 4, 6):
        scores, meta = _paired_frame(effect=1.5, n_donors=6, seed=seed)
        out = lmm_effect_sizes(
            scores,
            meta,
            donor_col="donor_id",
            condition_col="condition",
            group_col="subtype",
            case="Lymphedema",
            control="Normal",
            min_donors_per_arm=99,  # skip the LMM; exercise the donor-level fallback
        )
        row = out.iloc[0]
        assert abs(row["effect"] - 1.5) < 1e-6, f"seed={seed}"
        assert not np.isfinite(row["p_value"]), (
            f"seed={seed} reported p={row['p_value']!r} off a spread of "
            f"{np.finfo(float).eps:.0e}-scale rounding error"
        )


def test_lmm_effect_sizes_never_reports_p_zero_from_the_mixed_model_either():
    """The mixed model underflowed too, and it is the *primary* path.

    Caught by rendering the flagship figure: a well-powered planted effect
    produced ``fdr = 0.00e+00`` straight out of the LMM, because statsmodels
    returns the normal-approximation p-value and a large effect over a few hundred
    cells drives it below the smallest representable double. Guarding only the
    donor-level fallback left the path that runs on every real run unguarded, so
    the table a manuscript is built from still carried a literal zero.

    Referring the Wald statistic to a t on the design's own denominator df has since
    removed the mechanism rather than the symptom: a t tail decays polynomially, so
    on five df it takes ``t`` around 1e60 to underflow, and this fixture — extreme
    enough to return a literal zero from the normal — now reads 5e-08. The floor is
    kept because the donor-level fallback still uses it, and the assertion here is
    kept because "no literal zero on the primary path" is the property that matters,
    whichever arithmetic delivers it.
    """
    scores, meta = _paired_frame(effect=3.0, n_donors=6, n_cells=25, seed=8)
    # Break the mirrored noise so the LMM is fittable and richly powered.
    rng = np.random.default_rng(8)
    scores = scores.copy()
    scores["prog"] = scores["prog"] + rng.normal(0.0, 0.25, size=len(scores))

    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["method"] == "lmm"
    assert row["p_value"] > 0.0, "p underflowed to a literal zero"
    assert row["fdr"] > 0.0, "and BH carried the zero into the FDR column"
    assert row["p_value"] < 1e-6, "the fixture is meant to be extremely significant"


def test_lmm_effect_sizes_explains_every_blank_effect():
    """No row may be NaN without saying why — the whole table is the audit trail.

    One donor total: the LMM guard rejects it and a paired t-test on a single
    pair has no degrees of freedom. That is a legitimate blank; it just has to
    be a *labelled* blank, so a reader of the flagship figure's source table can
    tell an underpowered stratum from a bug.
    """
    scores, meta = _paired_frame(effect=1.0, n_donors=1, seed=3)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        min_donors_per_arm=2,
    )
    assert "reason" in out.columns
    blank = out[~np.isfinite(out["effect"])]
    assert len(blank) == 1
    assert blank.iloc[0]["reason"], "a NaN effect with no reason is indistinguishable from a bug"
    assert "donor" in blank.iloc[0]["reason"].lower()


def test_lmm_effect_sizes_reason_is_empty_when_there_is_nothing_to_explain():
    """The converse: a clean fit must not carry a spurious explanation."""
    scores, meta = _paired_frame(effect=1.5, seed=1)
    out = lmm_effect_sizes(
        scores,
        meta,
        donor_col="donor_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
    )
    row = out.iloc[0]
    assert row["method"] == "lmm"
    assert row["reason"] == ""


# --------------------------------------------------------------------------- #
# permanova_by_group                                                            #
# --------------------------------------------------------------------------- #
def _sample_frame(separation: float, seed: int, n_per_arm: int = 4):
    """Per-sample module vectors for one subtype, ``n_per_arm`` case + control samples.

    ``n_per_arm`` matters for any test that asserts a SMALL p-value: a permutation
    test cannot report below 1/(distinct splits), and 4-vs-4 has only C(8,4)=70 of
    them, which floors p at roughly 0.03 no matter how separated the arms are.
    """
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    n = 0
    for cond, shift in (("Normal", 0.0), ("Lymphedema", separation)):
        for _s in range(n_per_arm):
            v = rng.normal(0.0, 0.2, size=3) + np.array([shift, shift, 0.0])
            rows.append({"m1": v[0], "m2": v[1], "m3": v[2], "condition": cond})
            idx.append(f"s{n}")
            n += 1
    meta = pd.DataFrame(
        {"condition": [r.pop("condition") for r in rows]},
        index=idx,
    )
    meta["sample_id"] = idx
    meta["subtype"] = "CV"
    scores = pd.DataFrame(rows, index=idx)
    return scores, meta


def test_permanova_detects_planted_separation_and_ignores_null():
    s_sep, m_sep = _sample_frame(separation=3.0, seed=10)
    out_sep = permanova_by_group(
        s_sep,
        m_sep,
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    assert out_sep.iloc[0]["p_value"] < 0.05
    assert out_sep.iloc[0]["R2"] > 0.5

    s_null, m_null = _sample_frame(separation=0.0, seed=11)
    out_null = permanova_by_group(
        s_null,
        m_null,
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    assert out_null.iloc[0]["p_value"] > 0.05


def _multigroup_frame(n_groups: int, seed: int, n_per_arm: int = 8):
    """Per-sample vectors for ``n_groups`` groups, only the first one separated."""
    frames, metas = [], []
    for g in range(n_groups):
        s, m = _sample_frame(separation=3.0 if g == 0 else 0.0, seed=seed + g, n_per_arm=n_per_arm)
        s.index = m.index = [f"g{g}_{i}" for i in s.index]
        m["sample_id"] = list(m.index)
        m["subtype"] = f"g{g}"
        frames.append(s)
        metas.append(m)
    return pd.concat(frames), pd.concat(metas)


def test_permanova_adjusts_across_the_group_family():
    """One PERMANOVA per group is a FAMILY, and the family must be BH-adjusted.

    ``lmm_effect_sizes`` adjusts across module x group; PERMANOVA did not adjust at
    all, so a run with ten groups reported whichever group happened to permute
    below 0.05 as a finding. Ten independent nulls give a ~40% chance of at least
    one such group -- and the figure stars it.
    """
    scores, meta = _multigroup_frame(10, seed=40)
    out = permanova_by_group(
        scores,
        meta,
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=999,
        seed=1337,
    )
    assert "fdr" in out.columns, "the group family is unadjusted"
    # BH is monotone and never shrinks a p-value.
    assert (out["fdr"] >= out["p_value"] - 1e-12).all()
    # The planted group survives adjustment; the nine nulls do not become findings.
    assert out.iloc[0]["fdr"] < 0.05
    assert int((out["fdr"] < 0.05).sum()) == 1


def test_permanova_fdr_respects_the_configured_method():
    scores, meta = _multigroup_frame(6, seed=60)
    kw = dict(
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    bh = permanova_by_group(scores, meta, fdr_method="fdr_bh", **kw)
    by = permanova_by_group(scores, meta, fdr_method="bonferroni", **kw)
    # Bonferroni is never less conservative than BH on the same family.
    assert (by["fdr"] >= bh["fdr"] - 1e-12).all()
    assert (by["fdr"] > bh["fdr"]).any()


def test_permanova_is_deterministic_under_fixed_seed():
    s, m = _sample_frame(separation=2.0, seed=12)
    kw = dict(
        sample_col="sample_id",
        condition_col="condition",
        group_col="subtype",
        case="Lymphedema",
        control="Normal",
        n_permutations=199,
        seed=1337,
    )
    a = permanova_by_group(s, m, **kw).iloc[0]
    b = permanova_by_group(s, m, **kw).iloc[0]
    assert a["pseudo_F"] == b["pseudo_F"]
    assert a["p_value"] == b["p_value"]


# --------------------------------------------------------------------------- #
# concordance + correlations                                                    #
# --------------------------------------------------------------------------- #
def test_leading_edge_jaccard_numerics():
    modules = {"A": ["g1", "g2", "g3"]}
    edges = {"P": ["g2", "g3", "g4"]}
    out = leading_edge_jaccard(modules, edges)
    # |{g2,g3}| / |{g1,g2,g3,g4}| = 2/4
    assert abs(out.loc["A", "P"] - 0.5) < 1e-12


def test_module_gene_overlap_is_symmetric_with_unit_diagonal():
    modules = {"A": ["g1", "g2"], "B": ["g2", "g3", "g4"]}
    with pytest.warns(DeprecationWarning, match="set_overlap_tests"):
        out = module_gene_overlap(modules)
    assert out.loc["A", "A"] == 1.0 and out.loc["B", "B"] == 1.0
    assert abs(out.loc["A", "B"] - out.loc["B", "A"]) < 1e-12
    # |{g2}| / |{g1,g2,g3,g4}| = 1/4
    assert abs(out.loc["A", "B"] - 0.25) < 1e-12


def test_program_correlation_matrix_spearman_signs():
    scores = pd.DataFrame(
        {
            "p1": [1.0, 2.0, 3.0, 4.0],
            "p2": [2.0, 4.0, 6.0, 8.0],  # monotone increasing with p1
            "p3": [4.0, 3.0, 2.0, 1.0],  # monotone decreasing with p1
        }
    )
    with pytest.warns(DeprecationWarning, match="program_correlation_tests"):
        out = program_correlation_matrix(scores, method="spearman")
    assert abs(out.loc["p1", "p2"] - 1.0) < 1e-12
    assert abs(out.loc["p1", "p3"] + 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# a pre-specified panel, and the family it is corrected in                      #
# --------------------------------------------------------------------------- #
MODULES = {
    "adhesion": ["FN1", "ITGAV", "ITGB1"],
    "growth": ["VEGFC", "FLT4"],
}


def test_a_composite_item_needs_every_entity_declared_not_merely_one():
    """The rule that makes a panel a statement about the modules.

    ``FN1->ITGAV_ITGB1`` is three declared genes and qualifies. ``FN1->TIE1`` shares its
    ligand with the panel and must not: "any declared entity" would admit most of the
    resource, since a broadcast ligand touches hundreds of receptors.
    """
    out = declared_panel_membership(
        MODULES, ["FN1->ITGAV_ITGB1", "FN1->TIE1", "ANGPT2->ITGB1", "VEGFC->FLT4"]
    ).set_index("item")
    assert bool(out.loc["FN1->ITGAV_ITGB1", "in_panel"])
    assert bool(out.loc["VEGFC->FLT4", "in_panel"])
    assert not bool(out.loc["FN1->TIE1", "in_panel"])  # receptor undeclared
    assert not bool(out.loc["ANGPT2->ITGB1", "in_panel"])  # ligand undeclared
    # The counts are what let a reader see *why* a pair was excluded.
    assert out.loc["FN1->TIE1", ["n_entities", "n_declared_entities"]].tolist() == [2, 1]
    assert out.loc["FN1->ITGAV_ITGB1", ["n_entities", "n_declared_entities"]].tolist() == [3, 3]


def test_the_declared_sets_a_pair_belongs_to_are_named_and_stably_ordered():
    out = declared_panel_membership(MODULES, ["FN1->FLT4"]).set_index("item")
    # Both ends declared, in two different modules: the pair links them.
    assert bool(out.loc["FN1->FLT4", "in_panel"])
    assert out.loc["FN1->FLT4", "panel_sets"] == "adhesion;growth"


def test_panel_membership_deduplicates_and_survives_an_item_with_no_receptor():
    out = declared_panel_membership(MODULES, ["FN1->ITGB1", "FN1->ITGB1", "FN1"])
    assert len(out) == 2
    assert bool(out.set_index("item").loc["FN1", "in_panel"])


def test_a_single_gene_panel_is_configurable_through_the_separators():
    """The engine's composite keys are one convention among several."""
    out = declared_panel_membership(
        MODULES, ["FN1|ITGAV+ITGB1"], entity_sep="|", member_sep="+"
    ).set_index("item")
    assert bool(out.loc["FN1|ITGAV+ITGB1", "in_panel"])


def _scan_rows(n: int, *, family: str, floor: float = 2 / 2**9) -> pd.DataFrame:
    """A slice of a scan's output table, already carrying the scan's own family columns."""
    return pd.DataFrame(
        {
            "focus": family,
            "sign_test_p": np.linspace(0.001, 0.4, n),
            "sign_test_p_conservative": np.linspace(0.002, 0.8, n),
            "design_floor_p": floor,
            # What the scan corrected in: far too wide for its own floor to be reachable.
            "family_size": 7555,
            "family_best_floor_p": floor,
            "family_min_concordant": 591,
            "family_floor_reachable": True,
            "sign_test_fdr": 1.0,
            "sign_test_fdr_conservative": 1.0,
        }
    )


def test_restricting_a_family_recomputes_the_fdr_and_does_not_retest():
    panel = _scan_rows(30, family="LEC")
    out = recorrect_within_family(panel)
    # The p-values are untouched: only the multiplicity accounting moved.
    assert np.allclose(out["sign_test_p"], panel["sign_test_p"])
    # The FDR is now BH among 30, which is what the panel is.
    assert np.allclose(out["sign_test_fdr"], bh_fdr(panel["sign_test_p"].to_numpy()))
    assert (out["sign_test_fdr"] < 1.0).any()


def test_every_family_column_moves_with_the_family_not_just_the_fdr():
    """The mistake this exists to prevent: a right FDR beside a stale ``family_size``.

    ``family_min_concordant`` describes how many items must move together before BH can
    call one, so it is a property of the family in exactly the way the q-values are. A
    table reporting 591 next to a 30-item panel's FDRs is self-contradicting.
    """
    out = recorrect_within_family(_scan_rows(30, family="LEC"))
    assert (out["family_size"] == 30).all()
    assert (out["family_min_concordant"] == 3).all()  # ceil(0.0039 * 30 / 0.05)
    assert out["family_floor_reachable"].all()
    # And the scan's size is preserved rather than dropped, so a reader can see that the
    # panel is a restriction of something larger and not a discovery.
    assert (out["n_scanned"] == 7555).all()


def test_each_family_is_corrected_among_itself_when_the_table_holds_several():
    lec, bec = _scan_rows(30, family="LEC"), _scan_rows(12, family="BEC")
    out = recorrect_within_family(pd.concat([lec, bec], ignore_index=True), by=("focus",))
    assert set(out.loc[out["focus"] == "LEC", "family_size"]) == {30}
    assert set(out.loc[out["focus"] == "BEC", "family_size"]) == {12}
    # Same p-values in two families are two different FDRs — that is what BH means.
    assert np.allclose(
        out.loc[out["focus"] == "BEC", "sign_test_fdr"].to_numpy(),
        bh_fdr(bec["sign_test_p"].to_numpy()),
    )
    assert not np.allclose(
        out.loc[out["focus"] == "LEC", "sign_test_fdr"].to_numpy()[:12],
        out.loc[out["focus"] == "BEC", "sign_test_fdr"].to_numpy(),
    )


def test_a_second_family_is_not_left_without_a_scanned_count():
    """Pin the bug: recording ``n_scanned`` per family sees the column exist as soon as the
    first family wrote it, and every later family silently gets NaN."""
    out = recorrect_within_family(
        pd.concat(
            [_scan_rows(4, family="LEC"), _scan_rows(4, family="BEC")],
            ignore_index=True,
        ),
        by=("focus",),
    )
    assert (out["n_scanned"] == 7555).all()


def test_a_family_too_small_for_its_own_floor_is_reported_as_reachable_again():
    """The point of restricting: the floor becomes reachable, which is a design fact."""
    wide = recorrect_within_family(_scan_rows(7555, family="LEC"))
    narrow = recorrect_within_family(_scan_rows(30, family="LEC"))
    assert wide["family_min_concordant"].iloc[0] == 591
    assert narrow["family_min_concordant"].iloc[0] == 3


def test_an_absent_p_value_column_is_skipped_rather_than_invented():
    panel = _scan_rows(10, family="LEC").drop(columns=["sign_test_p_conservative"])
    out = recorrect_within_family(panel)
    assert "sign_test_p_conservative" not in out.columns
    assert "sign_test_fdr" in out.columns


def test_recorrecting_an_empty_table_is_a_no_op_and_a_bad_key_is_loud():
    empty = _scan_rows(0, family="LEC")
    assert recorrect_within_family(empty).empty
    with pytest.raises(KeyError, match="no such grouping column"):
        recorrect_within_family(_scan_rows(5, family="LEC"), by=("flow",))
