"""annotation_consensus is wired into config, planner, and executor."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import build_default_stage_registry


def test_config_has_stage_flag_and_block():
    c = CellQuorumConfig()
    assert c.stages.annotation_consensus is True
    assert c.annotation_consensus.key_added == "cell_type"


def test_executor_registry_has_stage():
    reg = build_default_stage_registry()
    assert reg.get("annotation_consensus") is not None
    assert reg.get("annotation_consensus").name == "annotation_consensus"


def test_planner_orders_consensus_after_reference_mapping():
    from cellquorum.backends.registry import build_default_backend_registry
    from cellquorum.core.planner import build_pipeline_plan

    c = CellQuorumConfig()
    plan = build_pipeline_plan(c, backend_registry=build_default_backend_registry())
    names = [s.name for s in plan.stages]
    assert "annotation_consensus" in names
    assert names.index("annotation_consensus") > names.index("reference_mapping")
    assert names.index("annotation_consensus") < names.index("annotation_diagnostics")
