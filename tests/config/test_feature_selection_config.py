"""FeatureSelectionConfig + planner ordering for the feature_selection stage."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig


def test_feature_selection_config_defaults():
    c = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    assert c.feature_selection.enabled is False
    assert c.feature_selection.method == "seurat_v3"
    assert c.feature_selection.n_top_genes == 3000
    assert c.feature_selection.counts_layer == "counts"
    assert c.feature_selection.write_figures is True
    # Both switches agree. This line used to assert True against a block default of False,
    # pinning the state that made `stages.feature_selection: true` reach the stage and skip
    # it — leaving PCA and scVI on all ~33,000 genes.
    assert c.stages.feature_selection is False
