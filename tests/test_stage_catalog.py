"""Tests for the stage catalog primitives and behavioral-identity locks."""

import pytest

from cellquorum.config.models import CellQuorumConfig, StageSelectionConfig
from cellquorum.core.stage_catalog import (
    StageCatalog,
    StageCatalogError,
    StageSpec,
    register_planned_stage,
    register_stage,
)
from cellquorum.core.stages import all_stage_specs


def test_register_stage_sets_name_and_category_and_registers():
    catalog = StageCatalog()

    @register_stage(
        name="demo",
        order=10,
        config_flag="demo",
        config_field="demo",
        category="demo_cat",
        catalog=catalog,
    )
    class DemoStage:
        pass

    assert DemoStage.name == "demo"
    assert DemoStage.stage_category == "demo_cat"
    (spec,) = catalog.specs()
    assert spec.name == "demo"
    assert spec.order == 10
    assert spec.config_flag == "demo"
    assert spec.config_field == "demo"
    assert spec.category == "demo_cat"
    assert spec.is_implemented is True
    assert spec.factory is DemoStage


def test_register_stage_without_category_leaves_stage_category_unset():
    catalog = StageCatalog()

    @register_stage(
        name="plain", order=10, config_flag="plain", config_field="plain", catalog=catalog
    )
    class PlainStage:
        pass

    assert PlainStage.name == "plain"
    assert "stage_category" not in vars(PlainStage)
    assert catalog.specs()[0].category is None


def test_register_planned_stage_has_no_factory_or_config_field():
    catalog = StageCatalog()
    register_planned_stage(name="future", order=20, config_flag="future", catalog=catalog)
    (spec,) = catalog.specs()
    assert spec.is_implemented is False
    assert spec.factory is None
    assert spec.config_field is None
    assert spec.category is None


def test_specs_are_sorted_by_order():
    catalog = StageCatalog()
    register_planned_stage(name="b", order=30, config_flag="b", catalog=catalog)
    register_planned_stage(name="a", order=10, config_flag="a", catalog=catalog)
    register_planned_stage(name="c", order=20, config_flag="c", catalog=catalog)
    assert [s.name for s in catalog.specs()] == ["a", "c", "b"]


def test_duplicate_name_raises():
    catalog = StageCatalog()
    register_planned_stage(name="dup", order=10, config_flag="dup", catalog=catalog)
    with pytest.raises(StageCatalogError):
        register_planned_stage(name="dup", order=20, config_flag="dup2", catalog=catalog)


def test_duplicate_order_raises():
    catalog = StageCatalog()
    register_planned_stage(name="one", order=10, config_flag="one", catalog=catalog)
    with pytest.raises(StageCatalogError):
        register_planned_stage(name="two", order=10, config_flag="two", catalog=catalog)


def test_implemented_filters_out_planned():
    catalog = StageCatalog()

    @register_stage(name="real", order=10, config_flag="real", config_field="real", catalog=catalog)
    class RealStage:
        pass

    register_planned_stage(name="planned", order=20, config_flag="planned", catalog=catalog)
    assert [s.name for s in catalog.implemented()] == ["real"]
    assert len(catalog) == 2


def test_frozen_spec_is_immutable():
    from dataclasses import FrozenInstanceError

    spec = StageSpec(
        name="x", order=10, config_flag="x", config_field="x", category=None, factory=None
    )
    with pytest.raises(FrozenInstanceError):
        spec.name = "y"  # frozen dataclass


def test_get_missing_name_raises_keyerror():
    catalog = StageCatalog()
    with pytest.raises(KeyError):
        catalog.get("nonexistent")


def test_empty_catalog_has_no_specs():
    catalog = StageCatalog()
    assert catalog.specs() == ()
    assert len(catalog) == 0


PLANNED = {"integration_gate", "state_scoring", "discovery", "composition", "molecular_inference"}

# The canonical 35-stage order, frozen. Any reorder/add/remove is a deliberate
# change that must update this list together with the config models.
GOLDEN_STAGE_ORDER = [
    "ambient_correction",
    "qc",
    "preprocessing",
    "feature_selection",
    "dimensionality",
    "integration",
    "integration_gate",
    "clustering",
    "annotation",
    "subclustering",
    "adjudication",
    "reference_mapping",
    "annotation_consensus",
    "annotation_diagnostics",
    "population_identity",
    "integration_benchmark",
    "state_scoring",
    "discovery",
    "composition",
    "embeddings",
    "differential_expression",
    "differential_abundance",
    "enrichment",
    "enrichment_viz",
    "de_viz",
    "coexpression",
    "grn",
    "perturbation",
    "molecular_inference",
    "trajectory",
    "trajectory_viz",
    "cell_cell_communication",
    "multicellular_programs",
    "ccc_network",
    "ccc_viz",
]

# The 30 implemented stages, alphabetical — mirrors the executor registry
# snapshot in tests/test_pipeline_executor.py:302-333.
GOLDEN_IMPLEMENTED_SORTED = sorted(n for n in GOLDEN_STAGE_ORDER if n not in PLANNED)

# CellQuorumConfig fields that are NOT stage sub-blocks.
NON_STAGE_CONFIG_FIELDS = {
    "project",
    "paths",
    "input",
    "run",
    "compute",
    "r",
    "report",
    "stages",
    "markers",
    "cohort",
    "design",
    "contrasts",
}

# The 5 implemented stages that deliberately have NO method-registry category
# (they are structural/reconciliation steps, not method-dispatch stages). Frozen
# here as an INDEPENDENT source of truth: the identity invariant below pins each
# stage's declared category against this list rather than against the decorator's
# own value, so a mistyped or misplaced category= argument fails loudly.
STAGES_WITHOUT_CATEGORY = {
    "qc",
    "preprocessing",
    "adjudication",
    "annotation_consensus",
    "population_identity",
}


def test_catalog_order_matches_golden():
    assert [s.name for s in all_stage_specs()] == GOLDEN_STAGE_ORDER


def test_catalog_implemented_set_matches_golden():
    impl = sorted(s.name for s in all_stage_specs() if s.is_implemented)
    assert impl == GOLDEN_IMPLEMENTED_SORTED
    assert len(GOLDEN_IMPLEMENTED_SORTED) == 30


def test_orders_are_unique_and_ascending():
    orders = [s.order for s in all_stage_specs()]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_every_stage_flag_matches_selection_config_fields():
    flags = {s.config_flag for s in all_stage_specs()}
    assert flags == set(StageSelectionConfig.model_fields)


def test_every_implemented_config_field_matches_config_sub_blocks():
    fields = {s.config_field for s in all_stage_specs() if s.is_implemented}
    stage_fields = set(CellQuorumConfig.model_fields) - NON_STAGE_CONFIG_FIELDS
    assert fields == stage_fields


def test_planned_and_implemented_invariants():
    for spec in all_stage_specs():
        if spec.name in PLANNED:
            assert spec.factory is None
            assert spec.config_field is None
        else:
            assert spec.factory is not None
            assert spec.config_field is not None


def test_category_matches_golden_identity_invariant():
    """Pin each stage's category against an INDEPENDENT golden, not itself.

    Every method-dispatch stage's category equals its own name; the five
    structural stages in STAGES_WITHOUT_CATEGORY have no category. Asserting
    against the golden set (not against the decorator's own copied value)
    makes this a real lock: a mistyped or misplaced ``category=`` fails here.
    """
    for spec in all_stage_specs():
        if not spec.is_implemented:
            continue
        if spec.name in STAGES_WITHOUT_CATEGORY:
            assert spec.category is None, f"{spec.name} should have no category"
        else:
            assert (
                spec.category == spec.name
            ), f"{spec.name} category should equal its name, got {spec.category!r}"


def test_config_field_equals_stage_name_for_implemented_stages():
    """Every implemented stage reads the config sub-block named after itself.

    Pins config_field against the stage's own name (an independent identity),
    catching a per-stage config_field swap that the set-equality check in
    test_every_implemented_config_field_matches_config_sub_blocks cannot see.
    """
    for spec in all_stage_specs():
        if spec.is_implemented:
            assert (
                spec.config_field == spec.name
            ), f"{spec.name} config_field should equal its name, got {spec.config_field!r}"


def test_factory_instances_expose_declared_name_and_category():
    """The decorator actually writes name/stage_category onto each stage class."""
    for spec in all_stage_specs():
        if not spec.is_implemented:
            continue
        inst = spec.factory()
        assert inst.name == spec.name
        if spec.category is not None:
            assert inst.stage_category == spec.category


def test_ccc_network_flag_is_network_analysis():
    spec = next(s for s in all_stage_specs() if s.name == "ccc_network")
    assert spec.config_flag == "network_analysis"
    assert spec.config_field == "ccc_network"


def test_planner_plan_order_matches_catalog():
    from cellquorum.core.planner import build_pipeline_plan

    plan = build_pipeline_plan(CellQuorumConfig())
    names = [st["name"] for st in plan.to_dict()["stages"]]
    assert names == GOLDEN_STAGE_ORDER
