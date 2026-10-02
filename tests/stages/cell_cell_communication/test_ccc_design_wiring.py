"""The CCC stage bridges config.design -> multinichenet's case/control keys.

Real bug, found live: a manifest declared config.design.case/control (the
project's one declared LE-vs-Normal contrast) and commented that
multinichenet "runs off the design.case/control contrast already declared" --
but CellCellCommunicationStage._augment_config only ever bridged sample_col.
multinichenet's own config.get("case")/get("control") stayed None, so the
run always skipped with "no contrast (case/control) declared" despite the
contrast being right there in config.design. This pins the same
declare-the-question-once bridge differential_expression already has.
"""

from __future__ import annotations

from cellquorum.config.cohort import CohortConfig
from cellquorum.config.design import DesignConfig
from cellquorum.config.models import CellQuorumConfig
from cellquorum.stages.cell_cell_communication.stage import CellCellCommunicationStage


class _Ctx:
    """Minimal pipeline context wrapping a real config object."""

    def __init__(self, config):
        self.config = config


def test_augment_config_pulls_case_control_from_design():
    """A populated config.design supplies case/control to the CCC config, so
    multinichenet's own case/control read is no longer always None."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = CellCellCommunicationStage()

    augmented = stage._augment_config(_Ctx(config), {"methods": [{"method": "multinichenet"}]})

    assert augmented["case"] == "LE"
    assert augmented["control"] == "Normal"
    assert augmented["condition_col"] == "condition"


def test_augment_config_cohort_condition_key_wins_over_design():
    """Cohort's condition key takes precedence over design's, same as DE."""

    config = CellQuorumConfig(
        cohort=CohortConfig(condition_key="group"),
        design=DesignConfig(condition_col="condition", case="LE", control="Normal"),
    )
    stage = CellCellCommunicationStage()

    augmented = stage._augment_config(_Ctx(config), {})

    assert augmented["condition_col"] == "group"
    assert augmented["case"] == "LE"
    assert augmented["control"] == "Normal"


def test_augment_config_stage_value_wins_over_design():
    """An explicit case/control already in the CCC config block is not
    overwritten by design -- the stage value always wins."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = CellCellCommunicationStage()

    augmented = stage._augment_config(_Ctx(config), {"case": "Explicit", "control": "AlsoExplicit"})

    assert augmented["case"] == "Explicit"
    assert augmented["control"] == "AlsoExplicit"


def test_augment_config_absent_design_leaves_case_control_unset():
    """No design block: case/control stay unset, so multinichenet degrades to
    its existing clean skip rather than crashing on a missing config."""

    config = CellQuorumConfig()
    stage = CellCellCommunicationStage()

    augmented = stage._augment_config(_Ctx(config), {})

    assert augmented.get("case") is None
    assert augmented.get("control") is None
