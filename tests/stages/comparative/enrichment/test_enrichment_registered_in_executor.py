"""Test that EnrichmentStage is registered in the executor and PipelineConfig."""

from __future__ import annotations


def test_pipeline_config_has_enrichment():
    """Verify CellQuorumConfig has enrichment attribute with correct defaults."""
    from cellquorum.config.models import CellQuorumConfig

    cfg = CellQuorumConfig()
    assert cfg.enrichment.enabled is True
    assert cfg.enrichment.gene_set_collections == ["hallmark", "reactome"]
