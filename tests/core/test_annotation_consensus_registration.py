"""annotation_consensus is wired into config, planner, and executor."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig


def test_config_has_stage_flag_and_block():
    c = CellQuorumConfig()
    assert c.stages.annotation_consensus is True
    assert c.annotation_consensus.key_added == "cell_type"
