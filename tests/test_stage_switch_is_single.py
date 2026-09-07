"""Whether a stage runs is declared in exactly one place.

It used to be declared in two: the ``stages:`` block, which the planner reads, and an
``enabled`` field on the stage's own config block, which the run gate reads. Five stages
defaulted to opposite values in the two, so a config asking for one of them got a plan that
listed it, a banner that printed it, a progress line that reached it — and then
``skipped: disabled by config``.

That is not a cosmetic inconsistency. ``stages.feature_selection: true`` produced no
``var['highly_variable']``, so the PCA basis and the scVI latent space were both built from
all ~33,000 genes instead of ~2,000, on a 202,000-cell cohort, and every cell-type call
downstream inherited it. Nothing warned, because from each component's own point of view
nothing was wrong.

These tests pin the reconciliation rather than the five particular stages, so a stage added
later cannot reintroduce the split.
"""

from __future__ import annotations

import pytest

from cellquorum.config.models import CellQuorumConfig


def _config(**overrides: object) -> CellQuorumConfig:
    return CellQuorumConfig.model_validate({"project": {"name": "switch_test"}, **overrides})


def _stage_names_with_own_enabled(config: CellQuorumConfig) -> list[str]:
    """Stage flags in ``stages:`` whose config block also carries an ``enabled`` field."""

    names = []
    for name in type(config.stages).model_fields:
        block = getattr(config, name, None)
        if block is not None and hasattr(block, "enabled"):
            names.append(name)
    return names


# ═══ The two switches can never disagree ═══════════════════════════════════════════


def test_no_stage_has_two_switches_that_disagree_by_default() -> None:
    """The bug, stated as a property: no stage may default to run-and-skip.

    Asserting over every stage is the point. Fixing the five known offenders by hand would
    leave the thirty-sixth stage free to arrive with the same split.
    """

    config = _config()
    disagreements = {
        name: (getattr(config.stages, name), getattr(config, name).enabled)
        for name in _stage_names_with_own_enabled(config)
        if bool(getattr(config.stages, name)) != bool(getattr(config, name).enabled)
    }
    assert (
        not disagreements
    ), f"these stages would be planned and then skipped, or vice versa: {disagreements}"


@pytest.mark.parametrize("requested", [True, False])
def test_the_stages_block_decides(requested: bool) -> None:
    """``stages:`` is authoritative, including for the stage that hit this in production."""

    config = _config(stages={"feature_selection": requested})
    assert config.feature_selection.enabled is requested
    assert config.stages.feature_selection is requested


def test_every_stage_honours_the_stages_block() -> None:
    """Not just feature_selection: enabling any stage there reaches its own gate."""

    for name in _stage_names_with_own_enabled(_config()):
        config = _config(stages={name: True})
        assert getattr(config, name).enabled is True, f"stages.{name}: true did not reach {name}"


def test_a_block_level_enabled_is_mirrored_up_to_the_plan() -> None:
    """The older style still works, and the planner now agrees with it.

    Configs in the wild set ``enabled`` inside the stage block. Those must keep working, and
    the plan must reflect them — otherwise the banner advertises a stage list that is not
    what runs.
    """

    config = _config(subclustering={"enabled": True})
    assert config.stages.subclustering is True
    assert config.subclustering.enabled is True


def test_requested_in_stages_but_disabled_in_the_block_is_refused() -> None:
    """The dangerous pair: reads as "run it", behaves as a silent skip."""

    with pytest.raises(ValueError, match="declared once"):
        _config(stages={"qc": True}, qc={"enabled": False})


def test_switched_off_in_stages_with_settings_left_behind_is_fine() -> None:
    """The other direction is unambiguous, and refusing it would break working configs.

    Turning a stage off in ``stages:`` while its settings block still carries
    ``enabled: true`` is how a config gets narrowed to a later slice of the pipeline. It is
    also exactly what the old two-flag AND did, so it must keep meaning "off".
    """

    config = _config(stages={"clustering": False}, clustering={"enabled": True})
    assert config.stages.clustering is False
    assert config.clustering.enabled is False


def test_saying_both_and_agreeing_is_fine() -> None:
    """Redundant but harmless: most shipped configs state both."""

    config = _config(stages={"clustering": True}, clustering={"enabled": True})
    assert config.clustering.enabled is True


def test_component_switches_inside_a_stage_are_untouched() -> None:
    """``qc.doublets.enabled`` decides whether a PART of a stage runs.

    That is a different question with only one place to ask it, so the reconciliation must
    not reach into nested blocks — which would tie doublet detection to whether QC runs.
    """

    config = _config(stages={"qc": True}, qc={"doublets": {"enabled": False}})
    assert config.qc.enabled is True
    assert config.qc.doublets.enabled is False
