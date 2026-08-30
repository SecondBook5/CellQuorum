"""
Test that the de_viz stage is wired into config selection, planner, and executor.

This suite verifies the wiring checklist for Task 5:
1. Config defaults are present (StageSelectionConfig.de_viz, CellQuorumConfig.de_viz).
2. The planner schedules de_viz AFTER enrichment_viz and BEFORE trajectory_viz.
3. The default executor registry includes a DeVizStage instance under "de_viz".
"""

from cellquorum.stages.comparative.differential_expression.viz.stage import DeVizStage
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import build_default_stage_registry
from cellquorum.core.planner import build_pipeline_plan


def test_config_defaults() -> None:
    """Test that de_viz config defaults are present and typed."""
    cfg = CellQuorumConfig()
    # StageSelectionConfig.de_viz should exist and default to True.
    assert hasattr(cfg.stages, "de_viz")
    assert cfg.stages.de_viz is True
    # CellQuorumConfig.de_viz should exist and be a DeVizConfig instance.
    assert hasattr(cfg, "de_viz")
    assert cfg.de_viz is not None
    # The config should have at least the base attributes.
    assert hasattr(cfg.de_viz, "enabled")
    assert hasattr(cfg.de_viz, "fc_cut")
    assert hasattr(cfg.de_viz, "fdr_cut")


def test_planner_schedules_de_viz_after_enrichment_viz() -> None:
    """Test that the planner schedules de_viz after enrichment_viz and before trajectory_viz."""
    cfg = CellQuorumConfig()
    plan = build_pipeline_plan(cfg)
    stage_names = [s.name for s in plan.stages]
    # All three stages should be present.
    assert "enrichment_viz" in stage_names
    assert "de_viz" in stage_names
    assert "trajectory_viz" in stage_names
    # de_viz should come after enrichment_viz.
    enrichment_viz_idx = stage_names.index("enrichment_viz")
    de_viz_idx = stage_names.index("de_viz")
    trajectory_viz_idx = stage_names.index("trajectory_viz")
    assert (
        de_viz_idx > enrichment_viz_idx
    ), f"de_viz (index {de_viz_idx}) should come after enrichment_viz (index {enrichment_viz_idx})"
    # de_viz should come before trajectory_viz.
    assert de_viz_idx < trajectory_viz_idx, (
        f"de_viz (index {de_viz_idx}) should come before "
        f"trajectory_viz (index {trajectory_viz_idx})"
    )


def test_executor_registry_includes_de_viz() -> None:
    """Test that the default stage registry includes a DeVizStage instance."""
    registry = build_default_stage_registry()
    # The registry should have a "de_viz" entry.
    assert "de_viz" in registry.registered_stage_names()
    # The entry should be a DeVizStage instance.
    stage = registry.get("de_viz")
    assert stage is not None
    assert isinstance(stage, DeVizStage)
