"""Each self-check must catch the specific defect it was written for.

A self-check that cannot fail is worse than no self-check, because it looks like coverage. So
every check here is tested against a **reproduction of the real bug it exists to catch**, not
against a hypothetical. All four of these shipped with a fully green test suite and were found
only because a human asked a question:

    posterior_not_rescaled            a calibrated probability run through a robust z; 22,541
                                      cells changed state
    fallback_nulls_are_nested         a fallback that partitioned instead of nesting; damage
                                      detection fell from 100% to 10%
    no_coherent_population_removed     a coherent low-RNA population excluded wholesale
    masks_agree_with_state            an eligibility mask written from a stale verdict

Each test therefore comes in a pair: the defect is detected, and the correct version passes. A
check that fires on both is a check that always fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.stages.qc.selfcheck import (
    check_core_fraction_plausible,
    check_fallback_nulls_are_nested,
    check_masks_agree_with_state,
    check_no_coherent_population_removed,
    check_posterior_not_rescaled,
    run_self_check,
)

CELLS = pd.Index([f"cell_{i}" for i in range(600)])


# ═══ The 22,541-cell regression: a calibrated probability, rescaled ═════════════════


def _realistic_posterior(seed: int = 0) -> pd.Series:
    """A miQC posterior with the shape the real one had: median 0.035, MAD 0.015."""
    rng = np.random.default_rng(seed)
    body = rng.gamma(2.0, 0.02, size=len(CELLS) - 60)
    tail = rng.uniform(0.6, 1.0, size=60)
    return pd.Series(np.concatenate([body, tail]), index=CELLS).clip(0, 1)


def test_a_rescaled_posterior_is_caught() -> None:
    """The exact defect: severity = robust z of the posterior instead of the posterior.

    With median 0.035 and MAD 0.015, a cell at posterior 0.10 lands 4.3 sigma out and scores
    0.59 — the model itself says there is a 10% chance anything is wrong. Cells reaching the
    0.50 bar went from 10.7% to 18.8% of a 201,923-cell cohort.
    """
    posterior = _realistic_posterior()
    location = posterior.median()
    scale = (posterior - location).abs().median() * 1.4826
    z = ((posterior - location) / scale).clip(lower=0.0)
    rescaled = z / (z + 3.0)

    check = check_posterior_not_rescaled(rescaled, posterior)
    assert not check.passed
    assert check.verdict == "fail"
    assert "transform of the posterior" in check.detail


def test_the_unrescaled_posterior_passes() -> None:
    """The control. A check that fires on the correct version too is worthless."""
    posterior = _realistic_posterior()
    check = check_posterior_not_rescaled(posterior.copy(), posterior)
    assert check.passed


def test_an_absent_posterior_is_reported_not_silently_passed() -> None:
    """ "Could not check" must be distinguishable from "checked and fine"."""
    check = check_posterior_not_rescaled(None, None)
    assert check.passed
    assert check.verdict == "warn"
    assert "nothing to compare" in check.detail


# ═══ Damage detection 100% -> 10%: a fallback that partitioned ═════════════════════


def _partitioned_grouping() -> tuple[pd.Series, dict[str, pd.Series]]:
    """The bug: cells that fell back to `sample` share a key with nobody else.

    Reproduced the way it happened — the fallback level's key was only ever written for the
    cells that fell back to it, so their group *was* the fallback set.
    """
    level = pd.Series(["sample_x_lineage"] * 500 + ["sample"] * 100, index=CELLS)
    fine = pd.Series(["S1|L0"] * 500 + [None] * 100, index=CELLS)
    coarse = pd.Series([None] * 500 + ["S1"] * 100, index=CELLS)
    return level, {"sample_x_lineage": fine, "sample": coarse}


def _nested_grouping() -> tuple[pd.Series, dict[str, pd.Series]]:
    """The fix: the coarse key is written for every cell, so the fallback null is wider."""
    level = pd.Series(["sample_x_lineage"] * 500 + ["sample"] * 100, index=CELLS)
    fine = pd.Series(["S1|L0"] * 500 + [None] * 100, index=CELLS)
    coarse = pd.Series(["S1"] * 600, index=CELLS)
    return level, {"sample_x_lineage": fine, "sample": coarse}


def test_a_partitioned_fallback_is_caught() -> None:
    """Cells fall back when their own group is unusable, which selects for damaged barcodes.

    So a partitioned fallback estimates their null *from damage*, which inverts the result
    rather than degrading it.
    """
    level, keys = _partitioned_grouping()
    check = check_fallback_nulls_are_nested(level, keys)
    assert not check.passed
    assert check.verdict == "fail"
    assert "only other fallback cells" in check.detail


def test_a_nested_fallback_passes() -> None:
    """The control: a genuinely wider reference class must not trip the check."""
    level, keys = _nested_grouping()
    assert check_fallback_nulls_are_nested(level, keys).passed


def test_a_single_level_run_is_not_flagged() -> None:
    """Nesting is not applicable when nothing fell back."""
    level = pd.Series(["sample_x_lineage"] * 600, index=CELLS)
    keys = {"sample_x_lineage": pd.Series(["S1|L0"] * 600, index=CELLS)}
    assert check_fallback_nulls_are_nested(level, keys).passed


# ═══ The rare-population failure, as a gate ════════════════════════════════════════


def test_a_coherent_population_being_removed_is_caught() -> None:
    """A large group losing most of its non-multiplet cells to damage evidence."""
    audit = pd.DataFrame(
        {
            "n_cells": [18002, 4000],
            "damage_excluded_fraction": [0.87, 0.12],
            "multiplet_fraction": [0.01, 0.02],
            "suspect": [False, False],
            "vulnerable": [True, False],
        },
        index=["L12", "L0"],
    )
    check = check_no_coherent_population_removed(audit)
    assert not check.passed
    assert "L12" in check.detail


def test_a_handful_of_sub_floor_barcodes_is_not_a_population() -> None:
    """Five barcodes below the gene floor must not fail a run.

    On the real cohort the only vulnerable lineage was a five-cell `unassigned` group — barcodes
    the floor had already judged. Failing a 201,923-cell run over five of them would make the
    gate something people switch off.
    """
    audit = pd.DataFrame(
        {
            "n_cells": [5],
            "damage_excluded_fraction": [1.0],
            "multiplet_fraction": [0.0],
            "suspect": [True],
            "vulnerable": [True],
        },
        index=["unassigned"],
    )
    assert check_no_coherent_population_removed(audit).passed


def test_a_doublet_cluster_does_not_trip_the_population_check() -> None:
    """Multiplet-driven exclusion is already factored out of `vulnerable` by the audit.

    The real case: a 2,111-cell lineage, 83% excluded, 76% of it called doublets, and the
    *lowest* absolute severity of any lineage. Excluding it is QC working.
    """
    audit = pd.DataFrame(
        {
            "n_cells": [2111],
            "damage_excluded_fraction": [0.28],
            "multiplet_fraction": [0.76],
            "suspect": [False],
            "vulnerable": [False],
        },
        index=["L17"],
    )
    assert check_no_coherent_population_removed(audit).passed


# ═══ A mask written from a stale verdict ═══════════════════════════════════════════


def test_a_non_core_cell_holding_fit_permission_is_caught() -> None:
    """The failure that started this whole area: a verdict that controls nothing.

    A mask permitting a borderline cell to fit means non-core cells shape the biological
    reference, which is precisely what the eligibility model exists to prevent.
    """
    state = pd.Series(["core"] * 400 + ["borderline"] * 200, index=CELLS)
    stale = pd.Series(True, index=CELLS)
    check = check_masks_agree_with_state(state, stale)
    assert not check.passed
    assert "200" in check.detail


def test_a_consistent_mask_passes() -> None:
    """The control, including the legitimate case of a core cell losing FIT to a doublet call."""
    state = pd.Series(["core"] * 400 + ["borderline"] * 200, index=CELLS)
    mask = pd.Series([True] * 380 + [False] * 220, index=CELLS)
    assert check_masks_agree_with_state(state, mask).passed


# ═══ The blunt guard, and the whole report ═════════════════════════════════════════


def test_a_manifold_defined_by_a_minority_is_questioned() -> None:
    """Warned rather than failed: a badly degraded experiment can be legitimately mostly bad."""
    state = pd.Series(["core"] * 100 + ["borderline"] * 500, index=CELLS)
    check = check_core_fraction_plausible(state, minimum_core=0.50)
    assert not check.passed
    assert check.verdict == "warn"


def test_a_healthy_run_passes_every_check() -> None:
    """End to end on a consistent run: nothing fires."""
    posterior = _realistic_posterior()
    state = pd.Series(["core"] * 540 + ["borderline"] * 60, index=CELLS)
    level, keys = _nested_grouping()

    report = run_self_check(
        state,
        metabolic_severity=posterior.copy(),
        mito_posterior=posterior,
        null_level=level,
        null_keys=keys,
        lineage_audit=pd.DataFrame(
            {"n_cells": [600], "vulnerable": [False], "suspect": [False]}, index=["L0"]
        ),
        fit_mask=pd.Series([True] * 540 + [False] * 60, index=CELLS),
    )
    assert not report.failures()
    assert all(report.summary().values())


def test_the_report_names_every_check_it_ran() -> None:
    """Provenance has to record what was verified, not only that something was."""
    state = pd.Series(["core"] * 600, index=CELLS)
    summary = run_self_check(state).summary()

    assert set(summary) == {
        "posterior_not_rescaled",
        "fallback_nulls_are_nested",
        "no_coherent_population_removed",
        "masks_agree_with_state",
        "core_fraction_plausible",
    }


def test_a_negligible_lineage_cannot_stop_a_run() -> None:
    """A noisy flag on a tiny group must not fail a 200,000-cell cohort.

    Found by running the gate: a two-library smoke subset flagged a lineage at 60% damage-driven
    exclusion that sits at 28% on the full cohort, because a per-lineage null estimated from two
    libraries is noisy. A gate that stops a run on that is a gate people switch off, which makes
    it worth less than no gate at all.
    """
    audit = pd.DataFrame(
        {
            "n_cells": [80, 40_000],
            "damage_excluded_fraction": [0.9, 0.05],
            "multiplet_fraction": [0.0, 0.01],
            "suspect": [False, False],
            "vulnerable": [True, False],
        },
        index=["L_tiny", "L_big"],
    )
    # 80 of 40,080 cells is 0.2% of the cohort — above the cell floor, below the share floor.
    assert check_no_coherent_population_removed(audit).passed


def test_a_material_lineage_still_stops_a_run() -> None:
    """The control: the mast-cell scale case must still fail."""
    audit = pd.DataFrame(
        {
            "n_cells": [18_002, 40_000],
            "damage_excluded_fraction": [0.87, 0.05],
            "multiplet_fraction": [0.01, 0.01],
            "suspect": [False, False],
            "vulnerable": [True, False],
        },
        index=["L12", "L_big"],
    )
    check = check_no_coherent_population_removed(audit)
    assert not check.passed
    assert "L12" in check.detail
