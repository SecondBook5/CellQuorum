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


def test_stage_default_methods_list():
    from types import SimpleNamespace

    from cellquorum.stages.state_scoring.viz.stage import StateScoringVizStage

    stage = StateScoringVizStage()
    ctx = SimpleNamespace(config=SimpleNamespace(state_scoring_viz=None, design=None))
    aug = stage._augment_config(ctx, {})
    names = [m["method"] for m in aug["methods"]]
    assert names == ["state_score_violin"]


def test_registration_idempotent():
    import importlib

    import cellquorum.stages.state_scoring.viz
    from cellquorum.methods.registry import METHOD_REGISTRY

    importlib.reload(cellquorum.stages.state_scoring.viz)
    assert METHOD_REGISTRY.has("state_scoring_viz", "state_score_violin")
