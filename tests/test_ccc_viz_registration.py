"""Tests for CCC visualization stage registration and configuration."""

from __future__ import annotations


def test_ccc_viz_config_defaults():
    from cellquorum.ccc_viz.config import CccVizConfig

    cfg = CccVizConfig()
    assert cfg.enabled is True
    assert cfg.top_k == 15
    assert cfg.figure_formats == ["pdf", "png"]
    assert cfg.dpi == 300
    assert cfg.sources is None
    assert cfg.levels is None
