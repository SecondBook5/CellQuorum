"""The enrichment_viz stage must be registered in the default executor registry."""

from cellquorum.core.executor import build_default_stage_registry
from cellquorum.stages.comparative.enrichment.viz.stage import EnrichmentVizStage


def test_enrichment_viz_registered():
    registry = build_default_stage_registry()
    stage = registry.get("enrichment_viz")
    assert isinstance(stage, EnrichmentVizStage)


def test_enrichment_viz_in_sorted_stage_names():
    names = build_default_stage_registry().registered_stage_names()
    assert "enrichment_viz" in names
    # alphabetical: between enrichment and feature_selection
    assert (
        names.index("enrichment") < names.index("enrichment_viz") < names.index("feature_selection")
    )
