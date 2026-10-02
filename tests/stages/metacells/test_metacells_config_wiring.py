"""Verify the metacells stage is wired into the top-level config."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.stages.metacells.config import MetacellsConfig


def test_stage_flag_present_and_default_false() -> None:
    cfg = CellQuorumConfig()
    assert cfg.stages.metacells is False


def test_metacells_sub_block_is_metacells_config() -> None:
    cfg = CellQuorumConfig()
    assert isinstance(cfg.metacells, MetacellsConfig)
    assert cfg.metacells.method == "metacells"
    assert cfg.metacells.enabled is False
