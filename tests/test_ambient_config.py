"""Tests for AmbientCorrectionConfig + stage flag + planner ordering."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import PipelinePlanner
from cellquorum.qc.ambient.config import AmbientCorrectionConfig


def test_ambient_defaults():
    c = AmbientCorrectionConfig()
    assert c.enabled is False
    assert c.method == "soupx"
    assert c.round_to_int is True
    assert c.cellranger_root is None


def test_ambient_strict():
    with pytest.raises(ValidationError):
        AmbientCorrectionConfig(bogus=1)


def test_top_level_has_ambient_and_flag():
    cfg = CellQuorumConfig()
    assert isinstance(cfg.ambient_correction, AmbientCorrectionConfig)
    assert cfg.stages.ambient_correction is False  # off by default


def test_planner_orders_ambient_first_when_enabled():
    # Enable it and confirm it plans BEFORE qc.
    cfg = CellQuorumConfig()
    cfg.stages.ambient_correction = True
    names = [s.name for s in PipelinePlanner(cfg).build_plan().stages]
    assert "ambient_correction" in names
    assert names.index("ambient_correction") < names.index("qc")
