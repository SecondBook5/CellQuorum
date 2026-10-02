"""Tests for state_scoring_viz stage registration and configuration."""

from __future__ import annotations


def test_config_defaults():
    from cellquorum.config.models import CellQuorumConfig

    cfg = CellQuorumConfig()
    assert hasattr(cfg.stages, "state_scoring_viz")
    assert cfg.stages.state_scoring_viz is True
    assert hasattr(cfg, "state_scoring_viz")
    assert cfg.state_scoring_viz.enabled is True
    assert cfg.state_scoring_viz.score_source == "aucell"


def test_method_registered():
    import cellquorum.stages.state_scoring.viz  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("state_scoring_viz", "state_score_violin")


def test_stage_default_methods_list():
    from types import SimpleNamespace

    from cellquorum.stages.state_scoring.viz.stage import StateScoringVizStage

    stage = StateScoringVizStage()
    ctx = SimpleNamespace(config=SimpleNamespace(state_scoring_viz=None, design=None))
    aug = stage._augment_config(ctx, {})
    names = [m["method"] for m in aug["methods"]]
    assert names == ["state_score_violin"]


def test_planner_schedules_right_after_state_scoring():
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.planner import build_pipeline_plan

    cfg = CellQuorumConfig()
    plan = build_pipeline_plan(cfg)
    names = [s.name for s in plan.stages]
    i = names.index("state_scoring")
    assert names[i + 1] == "state_scoring_viz"


def test_executor_registry_includes_state_scoring_viz():
    from cellquorum.core.executor import build_default_stage_registry
    from cellquorum.stages.state_scoring.viz.stage import StateScoringVizStage

    registry = build_default_stage_registry()
    assert "state_scoring_viz" in registry.registered_stage_names()
    stage = registry.get("state_scoring_viz")
    assert isinstance(stage, StateScoringVizStage)


def test_registration_idempotent():
    import importlib

    import cellquorum.stages.state_scoring.viz
    from cellquorum.methods.registry import METHOD_REGISTRY

    importlib.reload(cellquorum.stages.state_scoring.viz)
    assert METHOD_REGISTRY.has("state_scoring_viz", "state_score_violin")
