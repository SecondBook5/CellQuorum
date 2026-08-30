"""Tests for reference mapping configuration."""

from __future__ import annotations

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import PipelinePlanner


def test_reference_mapping_config_defaults():
    """Test default reference mapping configuration values."""

    c = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    rm = c.reference_mapping
    assert rm.enabled is False
    assert rm.method == "scarches"
    assert rm.label_key == "cell_type"
    assert rm.seeds == [0, 1, 2, 3, 4]
    assert rm.cv_folds == 3
    assert rm.knn_k == 30
    assert rm.key_added == "ref_state"
    assert c.stages.reference_mapping is True


def test_reference_mapping_robustness_knobs_are_configurable():
    """Seeds, cv_folds, and knn_k are user-tunable robustness/cost knobs."""

    c = CellQuorumConfig.model_validate(
        {
            "project": {"name": "t"},
            "reference_mapping": {"seeds": [0, 1, 2], "cv_folds": 5, "knn_k": 15},
        }
    )
    rm = c.reference_mapping
    assert rm.seeds == [0, 1, 2]
    assert rm.cv_folds == 5
    assert rm.knn_k == 15


def test_reference_mapping_is_atlas_agnostic_via_config():
    """Test that non-KC atlas config validates identically."""

    # A NON-KC atlas config validates identically — proves nothing KC is
    # hardcoded.
    c = CellQuorumConfig.model_validate(
        {
            "project": {"name": "t"},
            "reference_mapping": {
                "enabled": True,
                "atlas_h5ad": "/data/lung_atlas.h5ad",
                "label_key": "lung_celltype",
                "atlas_batch_key": "donor",
                "reference_filters": [{"column": "status", "keep": ["healthy"]}],
                "force_genes": ["SFTPC", "SCGB1A1"],
                "seeds": [0],
            },
        }
    )
    assert c.reference_mapping.atlas_h5ad.endswith("lung_atlas.h5ad")
    assert c.reference_mapping.label_key == "lung_celltype"


def test_reference_mapping_ordered_after_annotation():
    """Test that reference_mapping is ordered after annotation in the plan."""

    c = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    names = [s.name for s in PipelinePlanner(c).build_plan().stages]
    assert names.index("annotation") < names.index("reference_mapping")


def test_annotation_diagnostics_ordered_after_reference_mapping():
    """Transferred reference labels must exist before annotation diagnostics runs."""

    c = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    names = [s.name for s in PipelinePlanner(c).build_plan().stages]
    assert names.index("reference_mapping") < names.index("annotation_diagnostics")


def test_scarches_method_is_registered():
    """Test that ScArchesMethod is auto-registered in METHOD_REGISTRY."""
    # Import triggers auto-registration.
    import cellquorum.stages.annotation.reference_mapping  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("reference_mapping", "scarches")
    # Verify dispatch resolves to the method.
    method = METHOD_REGISTRY.get("reference_mapping", "scarches")
    assert method.name == "scarches"
    assert method.stage_category == "reference_mapping"
