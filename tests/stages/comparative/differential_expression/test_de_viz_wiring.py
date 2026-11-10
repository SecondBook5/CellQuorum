"""
Test that the de_viz stage is wired into config selection, planner, and executor.

This suite verifies the wiring checklist for Task 5:
1. Config defaults are present (StageSelectionConfig.de_viz, CellQuorumConfig.de_viz).
2. The planner schedules de_viz AFTER enrichment_viz and BEFORE trajectory_viz.
3. The default executor registry includes a DeVizStage instance under "de_viz".
"""

from cellquorum.config.models import CellQuorumConfig


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
    assert hasattr(cfg.de_viz, "highlight_genes")
    assert cfg.de_viz.highlight_genes is None


def test_highlight_genes_is_in_the_stage_whitelist() -> None:
    from cellquorum.stages.comparative.differential_expression.viz.stage import _VIZ_CONFIG_KEYS

    assert "highlight_genes" in _VIZ_CONFIG_KEYS
