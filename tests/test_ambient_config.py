"""Tests for AmbientCorrectionConfig + stage flag + planner ordering."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import PipelinePlanner
from cellquorum.stages.ambient_correction.config import AmbientCorrectionConfig


def test_ambient_defaults():
    c = AmbientCorrectionConfig()
    # On by default. Ambient mRNA contaminates every droplet, and every stage
    # after this one inherits the error, so not correcting is the choice that
    # needs justifying — not correcting. Safe as a default because the stage
    # skips itself with an explicit reason when its inputs are absent.
    assert c.enabled is True
    assert c.method == "soupx"
    assert c.round_to_int is True
    assert c.cellranger_root is None


def test_ambient_strict():
    with pytest.raises(ValidationError):
        AmbientCorrectionConfig(bogus=1)


def test_top_level_has_ambient_and_flag():
    cfg = CellQuorumConfig()
    assert isinstance(cfg.ambient_correction, AmbientCorrectionConfig)
    # BOTH gates must be on, not just the sub-config. The planner reads
    # stages.ambient_correction; leaving that False while setting
    # ambient_correction.enabled True silently plans the stage as disabled, which
    # is how it read "Disabled by configuration" on a run that had asked for it.
    assert cfg.stages.ambient_correction is True


def test_planner_orders_ambient_first_by_default():
    # It plans BEFORE qc: correcting counts after filtering cells on those counts
    # would be the wrong order.
    cfg = CellQuorumConfig()
    names = [s.name for s in PipelinePlanner(cfg).build_plan().stages]
    assert "ambient_correction" in names
    assert names.index("ambient_correction") < names.index("qc")


def test_planner_can_still_turn_ambient_off():
    cfg = CellQuorumConfig()
    cfg.stages.ambient_correction = False
    plan = PipelinePlanner(cfg).build_plan()
    ambient = next(s for s in plan.stages if s.name == "ambient_correction")
    assert ambient.enabled is False
