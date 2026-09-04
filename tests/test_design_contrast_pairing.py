"""Tests for how a named contrast resolves pairing against the project design.

``Contrast.paired`` used to default to ``False`` while
``validate_design_against_obs`` preferred the contrast's value unconditionally. The
result was that adding a named contrast to a matched cohort silently turned pairing
OFF: donor baseline variance stayed in the residual, and real effects came back as
nulls. That is the same paired-design artifact that already forced a round of science
re-runs on this data, arriving by a different route, so the inheritance rule is
pinned here rather than left to the schema's default.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.config.design import (
    Contrast,
    DesignConfig,
    validate_design_against_obs,
)


def make_matched_obs(n_donors: int = 4) -> pd.DataFrame:
    """
    Build an obs table where every donor contributes both arms.

    A fully matched cohort is the only shape where the pairing question has a
    consequence: unpaired is estimable, just wrong to prefer.

    Args:
        n_donors: Number of donors, each appearing in both conditions.

    Returns:
        An obs table with ``donor_id`` and ``condition``.
    """

    rows = []
    for index in range(n_donors):
        for condition in ("Lymphedema", "Normal"):
            rows.extend([{"donor_id": f"D{index}", "condition": condition}] * 10)
    return pd.DataFrame(rows)


PAIRED_DESIGN = DesignConfig(
    donor_col="donor_id",
    condition_col="condition",
    case="Lymphedema",
    control="Normal",
    paired=True,
)


def test_a_contrast_that_says_nothing_inherits_a_paired_design() -> None:
    """
    Verify an unstated ``paired`` follows the design rather than defaulting off.

    This is the regression. The manifests for this project declare
    ``design.paired: true`` and name contrasts without mentioning pairing, so under
    the old default every named comparison was analysed unpaired on 9 matched donors.
    """

    contrast = Contrast(name="LE_vs_Normal", case="Lymphedema", control="Normal")

    assert contrast.paired is None

    result = validate_design_against_obs(
        make_matched_obs(), design=PAIRED_DESIGN, contrast=contrast
    )

    assert result.paired is True


def test_an_explicit_false_still_overrides_a_paired_design() -> None:
    """
    Verify a deliberate unpaired contrast is honoured.

    Inheritance must not become coercion: a cross-cohort comparison inside an
    otherwise within-donor study is legitimately unpaired, and writing
    ``paired: false`` is a visible act a reviewer can see in the config.
    """

    contrast = Contrast(name="LE_vs_Normal", case="Lymphedema", control="Normal", paired=False)

    result = validate_design_against_obs(
        make_matched_obs(), design=PAIRED_DESIGN, contrast=contrast
    )

    assert result.paired is False


def test_an_explicit_true_overrides_an_unpaired_design() -> None:
    """
    Verify inheritance runs in both directions.

    A project may leave ``design.paired`` off while one particular comparison is
    within-donor; the contrast is the right place to say so.
    """

    unpaired_design = DesignConfig(
        donor_col="donor_id",
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
        paired=False,
    )
    contrast = Contrast(name="LE_vs_Normal", case="Lymphedema", control="Normal", paired=True)

    result = validate_design_against_obs(
        make_matched_obs(), design=unpaired_design, contrast=contrast
    )

    assert result.paired is True


def test_no_contrast_uses_the_design_as_before() -> None:
    """
    Verify the contrast-free path is unchanged.

    Every stage in the engine currently validates with ``contrast=None``, so this is
    the path the shipped numbers came from and it must not move.
    """

    result = validate_design_against_obs(make_matched_obs(), design=PAIRED_DESIGN)

    assert result.paired is True


@pytest.mark.parametrize("declared", [None, True, False])
def test_pairing_never_depends_on_the_contrast_name(declared: bool | None) -> None:
    """
    Verify the resolved comparison is otherwise untouched by the pairing field.

    Guards against a fix that resolves pairing correctly while disturbing the
    case/control tokens the same helper returns.
    """

    contrast = Contrast(name="LE_vs_Normal", case="Lymphedema", control="Normal", paired=declared)

    result = validate_design_against_obs(
        make_matched_obs(), design=PAIRED_DESIGN, contrast=contrast
    )

    assert (result.case, result.control) == ("Lymphedema", "Normal")
    assert len(result.complete_pair_donors) == 4
