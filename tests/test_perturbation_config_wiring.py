"""Verify the perturbation stage is wired into the top-level config."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.gene_regulation.perturbation.config import PerturbationConfig


def test_stage_flag_present_and_default_true() -> None:
    cfg = CellQuorumConfig()
    assert cfg.stages.perturbation is True


def test_perturbation_sub_block_is_perturbation_config() -> None:
    cfg = CellQuorumConfig()
    assert isinstance(cfg.perturbation, PerturbationConfig)
    assert cfg.perturbation.method == "celloracle"
