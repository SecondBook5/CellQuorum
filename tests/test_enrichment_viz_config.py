"""Tests for EnrichmentVizConfig defaults and wiring into CellQuorumConfig."""

from cellquorum.stages.comparative.enrichment.viz.config import EnrichmentVizConfig
from cellquorum.config.models import CellQuorumConfig


def test_enrichment_viz_config_defaults():
    cfg = EnrichmentVizConfig()
    assert cfg.enabled is True
    assert cfg.top_k == 12
    assert cfg.figure_formats == ["pdf", "png"]
    assert cfg.dpi == 300
    assert cfg.collections is None
    assert cfg.resources is None


def test_enrichment_viz_config_is_strict():
    # StrictBaseModel forbids unknown keys — a biology-agnostic guard.
    import pydantic

    try:
        EnrichmentVizConfig(unknown_biology_key="hallmark")
    except pydantic.ValidationError:
        return
    raise AssertionError("EnrichmentVizConfig should reject unknown keys")


def test_cellquorum_config_has_enrichment_viz():
    cfg = CellQuorumConfig()
    assert isinstance(cfg.enrichment_viz, EnrichmentVizConfig)
    assert cfg.enrichment_viz.enabled is True
