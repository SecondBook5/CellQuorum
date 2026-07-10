"""Integration: dimensionality + clustering run in the real executor loop."""

from __future__ import annotations

from cellquorum.core.executor import build_default_stage_registry
from cellquorum.core.planner import PipelinePlanner
from cellquorum.methods.registry import METHOD_REGISTRY


def test_methods_self_registered():
    # Importing the packages must register their methods.
    import cellquorum.clustering  # noqa: F401
    import cellquorum.dimensionality  # noqa: F401

    assert METHOD_REGISTRY.get("dimensionality", "pca") is not None
    assert METHOD_REGISTRY.get("clustering", "leiden") is not None


def test_stages_in_default_registry():
    reg = build_default_stage_registry()
    assert reg.get("dimensionality") is not None
    assert reg.get("clustering") is not None


def test_planner_orders_new_stages_after_preprocessing():
    from cellquorum.config.models import CellQuorumConfig

    planner = PipelinePlanner(CellQuorumConfig())
    plan = planner.build_plan()
    names = [s.name for s in plan.stages]
    assert names.index("dimensionality") > names.index("preprocessing")
    # Clustering runs after annotation (its canonical slot), which also implies
    # it runs after dimensionality given the fixed stage order.
    assert names.index("clustering") > names.index("annotation")
    assert names.index("clustering") > names.index("dimensionality")
