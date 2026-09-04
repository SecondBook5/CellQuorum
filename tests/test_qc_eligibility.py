"""Tests for QC eligibility masks.

The property under test throughout: **only core cells may fit anything.** Everything else
follows from that, and violating it is how questionable cells end up defining the
biological reference that every later stage is measured against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.eligibility import (
    Analysis,
    EligibilityMasks,
    Permission,
    build_eligibility_masks,
    fit_mask,
)
from cellquorum.stages.qc.evidence import QCStateInitial

CELLS = pd.Index(["core_a", "core_b", "borderline_a", "borderline_b", "quarantine_a"])
STATE = pd.Series(
    [
        str(QCStateInitial.CORE),
        str(QCStateInitial.CORE),
        str(QCStateInitial.BORDERLINE),
        str(QCStateInitial.BORDERLINE),
        str(QCStateInitial.QUARANTINE),
    ],
    index=CELLS,
)


# ═══ The central invariant ═════════════════════════════════════════════════════════


@pytest.mark.parametrize("analysis", list(Analysis))
def test_only_core_cells_may_ever_fit(analysis) -> None:
    """No non-core cell may determine parameters, statistics, or structure.

    Args:
        analysis: The analysis whose fit permission is checked.
    """
    masks = build_eligibility_masks(STATE)
    column = EligibilityMasks.column_name(analysis, Permission.FIT)
    if column not in masks.masks:
        pytest.skip(f"no state grants FIT on {analysis}")

    fitting = masks.masks[column]
    assert fitting.loc[["core_a", "core_b"]].all()
    assert not fitting.loc[["borderline_a", "borderline_b", "quarantine_a"]].any()


def test_borderline_receives_a_transform_but_never_fits_the_manifold() -> None:
    """Projected, not joined. That distinction is the whole anti-circularity design.

    A borderline cell may legitimately receive an embedding coordinate while being
    forbidden from influencing the model that produced it.
    """
    masks = build_eligibility_masks(STATE)

    assert not masks.mask(Analysis.MANIFOLD, Permission.FIT).loc["borderline_a"]
    assert masks.mask(Analysis.MANIFOLD, Permission.TRANSFORM).loc["borderline_a"]


def test_borderline_is_excluded_from_statistical_inference() -> None:
    """DE, trajectory and communication are what a questionable cell would most distort."""
    masks = build_eligibility_masks(STATE)

    for analysis in (
        Analysis.DIFFERENTIAL_EXPRESSION,
        Analysis.TRAJECTORY,
        Analysis.CELL_CELL_COMMUNICATION,
    ):
        column = EligibilityMasks.column_name(analysis, Permission.INFERENCE)
        assert not masks.masks[column].loc["borderline_a"]


def test_borderline_may_still_be_annotated() -> None:
    """Rescue needs a provisional identity, so annotation must reach borderline cells."""
    masks = build_eligibility_masks(STATE)
    assert masks.mask(Analysis.ANNOTATION, Permission.TRANSFORM).loc["borderline_a"]


def test_quarantined_cells_inform_nothing() -> None:
    """A quarantined cell may appear in a figure and contribute to no conclusion."""
    masks = build_eligibility_masks(STATE)

    for analysis in Analysis:
        for permission in (Permission.FIT, Permission.INFERENCE):
            column = EligibilityMasks.column_name(analysis, permission)
            if column in masks.masks:
                assert not masks.masks[column].loc[
                    "quarantine_a"
                ], f"quarantined cell was granted {permission} on {analysis}"


def test_quarantined_cells_may_still_receive_an_embedding() -> None:
    """So a figure can show what was excluded and where it sat."""
    masks = build_eligibility_masks(STATE)
    assert masks.mask(Analysis.MANIFOLD, Permission.TRANSFORM).loc["quarantine_a"]


# ═══ Multiplet: not damaged, but not one cell ══════════════════════════════════════


def test_probable_multiplet_loses_counting_and_comparing_analyses() -> None:
    """A doublet can be an excellent library that is simply not one biological cell.

    So it keeps annotation — knowing *what* was doubleted is how you discover that one
    population is being disproportionately called — and loses anything that counts or
    compares cells.
    """
    multiplet = pd.Series([True, False, False, False, False], index=CELLS)
    masks = build_eligibility_masks(STATE, probable_multiplet=multiplet)

    assert not masks.mask(Analysis.COMPOSITION, Permission.INFERENCE).loc["core_a"]
    assert not masks.mask(Analysis.MANIFOLD, Permission.FIT).loc["core_a"]
    # ...but the other core cell is unaffected.
    assert masks.mask(Analysis.MANIFOLD, Permission.FIT).loc["core_b"]
    # ...and annotation survives.
    assert masks.mask(Analysis.ANNOTATION, Permission.INFERENCE).loc["core_a"]


def test_multiplet_flag_is_optional() -> None:
    """Datasets without doublet detection must still get masks."""
    masks = build_eligibility_masks(STATE)
    assert masks.mask(Analysis.MANIFOLD, Permission.FIT).loc["core_a"]


# ═══ The cohort-derived-quantity helper ═══════════════════════════════════════════


def test_fit_mask_names_the_population_a_cohort_statistic_must_use() -> None:
    """The PFlog1pPF trap: its cohort-estimated target is a fitted model that looks like config."""
    mask = fit_mask(STATE, Analysis.MANIFOLD)

    assert mask.loc[["core_a", "core_b"]].all()
    assert not mask.loc[["borderline_a", "quarantine_a"]].any()
    assert mask.index.equals(CELLS)


# ═══ Shape, naming, and reporting ═════════════════════════════════════════════════


def test_no_all_false_masks_are_emitted() -> None:
    """An all-False column reads as "this analysis excluded everyone", which is misleading."""
    masks = build_eligibility_masks(STATE)
    for name, mask in masks.masks.items():
        assert mask.any(), f"{name} is all-False and should not have been written"


def test_column_names_are_stable_and_prefixed() -> None:
    """Downstream stages and figures key off these names, so they are part of the contract."""
    assert EligibilityMasks.column_name(Analysis.MANIFOLD, Permission.FIT) == "qc_fit_manifold"
    assert (
        EligibilityMasks.column_name(Analysis.DIFFERENTIAL_EXPRESSION, Permission.INFERENCE)
        == "qc_inference_de"
    )


def test_obs_frame_is_aligned_and_boolean() -> None:
    """Masks land on obs as booleans indexed like the cells they describe."""
    frame = build_eligibility_masks(STATE).to_obs_frame()

    assert frame.index.equals(CELLS)
    assert all(dtype is np.dtype(bool) for dtype in frame.dtypes)


def test_summary_counts_eligible_cells_per_mask() -> None:
    """Provenance records what the verdict permitted, not only what it decided."""
    summary = build_eligibility_masks(STATE).summary()

    assert summary["qc_fit_manifold"] == 2
    assert summary["qc_transform_manifold"] == 5


def test_requesting_a_mask_that_was_never_written_raises() -> None:
    """Better a KeyError than an empty result that looks like a legitimate exclusion.

    A permission no state grants is never written, so asking for it is a caller error and
    must not quietly return an all-False mask.
    """
    masks = build_eligibility_masks(STATE)
    with pytest.raises(KeyError):
        masks.masks["qc_fit_a_permission_that_does_not_exist"]


def test_states_with_no_cells_do_not_break_the_masks() -> None:
    """A cohort with nothing quarantined is the normal, good case."""
    clean = pd.Series([str(QCStateInitial.CORE)] * 3, index=["a", "b", "c"])
    masks = build_eligibility_masks(clean)

    assert masks.mask(Analysis.MANIFOLD, Permission.FIT).all()
    assert masks.summary()["qc_fit_manifold"] == 3
