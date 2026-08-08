from __future__ import annotations

from cellquorum.enrichment.config import EnrichmentConfig


def test_defaults():
    c = EnrichmentConfig()
    assert c.enabled is True
    assert c.methods == []
    assert c.layer == "cellquorum_normalized"
    assert c.cell_type_col == "cell_type"
    assert c.counts_layer == "counts"
    assert c.seed == 42
    assert c.min_size == 10
    assert c.max_size == 500
    assert c.gsea_permutations == 1000
    assert c.fdr == 0.05
    assert c.gene_set_collections == ["hallmark", "reactome"]
    assert c.activity_resources == ["collectri", "progeny"]


def test_strict_rejects_unknown_field():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EnrichmentConfig(not_a_field=1)
