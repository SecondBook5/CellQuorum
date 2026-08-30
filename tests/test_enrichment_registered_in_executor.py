"""Test that EnrichmentStage is registered in the executor and PipelineConfig."""

from __future__ import annotations

from cellquorum.stages.comparative.enrichment.stage import EnrichmentStage
from cellquorum.core.executor import build_default_stage_registry


def test_enrichment_stage_registered():
    """Verify enrichment stage is in the default registry."""
    reg = build_default_stage_registry()
    assert "enrichment" in reg.stages
    assert isinstance(reg.stages["enrichment"], EnrichmentStage)


def test_pipeline_config_has_enrichment():
    """Verify CellQuorumConfig has enrichment attribute with correct defaults."""
    from cellquorum.config.models import CellQuorumConfig

    cfg = CellQuorumConfig()
    assert cfg.enrichment.enabled is True
    assert cfg.enrichment.gene_set_collections == ["hallmark", "reactome"]
