"""FeatureSelectionConfig + planner ordering for the feature_selection stage."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import PipelinePlanner


def test_feature_selection_config_defaults():
    c = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    assert c.feature_selection.enabled is False
    assert c.feature_selection.method == "seurat_v3"
    assert c.feature_selection.n_top_genes == 2000
    assert c.feature_selection.counts_layer == "counts"
    assert c.stages.feature_selection is True


def test_feature_selection_ordered_between_preprocessing_and_dimensionality():
    # The planner lists all stages regardless of enabled flag, so this test verifies
    # the slot ordering in the canonical stage sequence.
    c = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    plan = PipelinePlanner(c).build_plan()
    names = [s.name for s in plan.stages]
    assert "feature_selection" in names
    assert names.index("preprocessing") < names.index("feature_selection")
    assert names.index("feature_selection") < names.index("dimensionality")
