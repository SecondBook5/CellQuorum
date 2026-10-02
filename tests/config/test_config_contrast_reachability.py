"""Tests for the check that a declared contrast is one the engine will run.

``contrasts`` is not consumed anywhere: the comparison is taken from
``design.case``/``design.control``, the DE stage config declares no case/control
fields of its own, and nothing reads ``ContrastsConfig`` outside its own helpers.
The block is still written to ``provenance/resolved_config.json``, where a named
contrast carrying its own case/control reads as the record of what was compared.
So a contrast that disagrees with the design is a config that describes a
comparison the run never performed, and that has to fail at load rather than
resolve into a plausible-looking provenance file.
"""

from __future__ import annotations

import pytest

from cellquorum.config.models import CellQuorumConfig


def make_config(
    *, design: dict[str, object], contrasts: list[dict[str, object]]
) -> CellQuorumConfig:
    """
    Build a config with only the design and contrast sections set.

    Args:
        design: Values for the ``design`` block.
        contrasts: Contrast entries for the ``contrasts`` block.

    Returns:
        The validated config.
    """

    return CellQuorumConfig.model_validate(
        {"design": design, "contrasts": {"contrasts": contrasts}}
    )


MATCHED_DESIGN: dict[str, object] = {
    "donor_col": "donor_id",
    "condition_col": "condition",
    "case": "Lymphedema",
    "control": "Normal",
    "paired": True,
}


def test_a_contrast_matching_the_design_is_accepted() -> None:
    """
    Verify the shape every manifest in the portfolio uses still loads.

    All six lineage manifests and the manuscript manifest declare exactly one
    contrast that restates the design's comparison. That is harmless
    documentation, and a check that rejected it would be a check nobody could
    keep enabled.
    """

    config = make_config(
        design=MATCHED_DESIGN,
        contrasts=[{"name": "LE_vs_Normal", "case": "Lymphedema", "control": "Normal"}],
    )

    assert config.contrasts.get("LE_vs_Normal").case == "Lymphedema"


def test_no_contrasts_at_all_is_accepted() -> None:
    """
    Verify the check is inert when the block is absent.

    Most configs never declare a contrast; the comparison lives in ``design``.
    """

    config = make_config(design=MATCHED_DESIGN, contrasts=[])

    assert config.contrasts.contrasts == []


def test_a_contrast_that_diverges_from_the_design_is_rejected() -> None:
    """
    Verify the silent-mis-description case halts.

    This is the whole point. Editing the *named* contrast and leaving ``design``
    alone is the natural way to change "the comparison", and it would leave the
    run comparing the old levels while provenance advertised the new ones.
    """

    with pytest.raises(ValueError, match="Multi-contrast DE is not wired"):
        make_config(
            design=MATCHED_DESIGN,
            contrasts=[{"name": "Severe_vs_Mild", "case": "Severe", "control": "Mild"}],
        )


def test_the_rejection_names_both_comparisons() -> None:
    """
    Verify the error shows what was asked for and what would have run.

    A config error that does not print the divergence makes the author re-derive
    it, and the two pairs are the entire content of the mistake.
    """

    with pytest.raises(ValueError) as excinfo:
        make_config(
            design=MATCHED_DESIGN,
            contrasts=[{"name": "Severe_vs_Mild", "case": "Severe", "control": "Mild"}],
        )

    message = str(excinfo.value)
    assert "'Severe'" in message and "'Mild'" in message
    assert "'Lymphedema'" in message and "'Normal'" in message
    assert "Severe_vs_Mild" in message


def test_a_contrast_without_a_design_comparison_is_rejected() -> None:
    """
    Verify a contrast cannot be the only statement of the comparison.

    With ``design.case``/``design.control`` unset, the sole declaration of what to
    compare is the one field nothing reads, so the run would perform no
    comparison at all while the config looked complete.
    """

    with pytest.raises(ValueError, match="design.case/design.control are unset"):
        make_config(
            design={"donor_col": "donor_id", "condition_col": "condition"},
            contrasts=[{"name": "LE_vs_Normal", "case": "Lymphedema", "control": "Normal"}],
        )


def test_a_contrast_paired_flag_that_contradicts_the_design_is_rejected() -> None:
    """
    Verify an unreadable pairing override cannot be written silently.

    ``Contrast.paired`` reads as an override and is not one -- pairing is resolved
    from ``design`` and then auto-promoted per cell type. Someone writing
    ``paired: false`` on a contrast of a matched design believes they turned
    pairing off; they did not, and the run would still block on donor.
    """

    with pytest.raises(ValueError, match="never read"):
        make_config(
            design=MATCHED_DESIGN,
            contrasts=[
                {
                    "name": "LE_vs_Normal",
                    "case": "Lymphedema",
                    "control": "Normal",
                    "paired": False,
                }
            ],
        )


def test_a_contrast_paired_flag_agreeing_with_the_design_is_accepted() -> None:
    """
    Verify a redundant-but-true pairing restatement is allowed.

    It is documentation, not a contradiction, and rejecting it would punish being
    explicit.
    """

    config = make_config(
        design=MATCHED_DESIGN,
        contrasts=[
            {
                "name": "LE_vs_Normal",
                "case": "Lymphedema",
                "control": "Normal",
                "paired": True,
            }
        ],
    )

    assert config.contrasts.get("LE_vs_Normal").paired is True


def test_duplicate_contrast_names_are_rejected() -> None:
    """
    Verify a repeated name cannot shadow another contrast.

    ``ContrastsConfig.get`` returns the first match, so the second entry is
    unreachable by the only handle it has.
    """

    with pytest.raises(ValueError, match="duplicate name"):
        make_config(
            design=MATCHED_DESIGN,
            contrasts=[
                {"name": "LE_vs_Normal", "case": "Lymphedema", "control": "Normal"},
                {"name": "LE_vs_Normal", "case": "Lymphedema", "control": "Normal"},
            ],
        )
