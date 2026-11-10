"""The perturbation stage must be planned in canonical order (after grn, before trajectory)."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import build_pipeline_plan


def test_perturbation_stage_is_planned_in_canonical_order() -> None:
    order = build_pipeline_plan(CellQuorumConfig()).enabled_stage_names()
    assert "perturbation" in order
    assert order.index("grn") < order.index("perturbation")
    assert order.index("perturbation") < order.index("trajectory")
