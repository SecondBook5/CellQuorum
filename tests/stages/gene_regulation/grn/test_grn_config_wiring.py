"""Verify the grn stage is wired into the top-level config."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.stages.gene_regulation.grn.config import GrnConfig


def test_stage_flag_present_and_default_true() -> None:
    cfg = CellQuorumConfig()
    assert cfg.stages.grn is True


def test_grn_sub_block_is_grn_config() -> None:
    cfg = CellQuorumConfig()
    assert isinstance(cfg.grn, GrnConfig)
    assert cfg.grn.method == "pyscenic"
