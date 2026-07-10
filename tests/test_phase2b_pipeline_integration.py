"""Integration: integration + annotation stages register and order correctly."""

from __future__ import annotations

from cellquorum.core.executor import build_default_stage_registry
from cellquorum.methods.registry import METHOD_REGISTRY


def test_integration_annotation_methods_registered() -> None:
    """
    Verify that integration and annotation methods self-register.

    When the cellquorum.integration and cellquorum.annotation modules are
    imported, their methods must automatically register with METHOD_REGISTRY.
    This test ensures the required methods are available for stage construction.
    """

    # Import the integration module to trigger self-registration.
    # Import the annotation module to trigger self-registration.
    import cellquorum.annotation  # noqa: F401
    import cellquorum.integration  # noqa: F401

    # Confirm the Harmony integration method registered.
    assert METHOD_REGISTRY.get("integration", "harmony") is not None

    # Confirm the scVI integration method registered.
    assert METHOD_REGISTRY.get("integration", "scvi") is not None

    # Confirm the marker-vote annotation method registered.
    assert METHOD_REGISTRY.get("annotation", "marker_vote") is not None


def test_stages_in_default_registry() -> None:
    """
    Verify that integration and annotation stages exist in the default registry.

    The default stage registry must include both the integration stage and the
    annotation stage, ensuring that a minimal pipeline can construct them.
    """

    # Build the default stage registry.
    reg = build_default_stage_registry()

    # Confirm the integration stage exists.
    assert reg.get("integration") is not None

    # Confirm the annotation stage exists.
    assert reg.get("annotation") is not None


def test_planner_orders_integration_before_annotation_before_clustering() -> None:
    """
    Verify that the pipeline planner orders stages correctly.

    The pipeline must ensure that dimensionality reduction happens before
    integration, integration happens before annotation, and annotation happens
    before clustering. This maintains proper data flow.
    """

    # Import the configuration and planner classes.
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.planner import PipelinePlanner

    # Build the stage order from a default configuration.
    stage_names = [s.name for s in PipelinePlanner(CellQuorumConfig()).build_plan().stages]

    # Confirm dimensionality reduction happens before integration.
    assert stage_names.index("dimensionality") < stage_names.index("integration")

    # Confirm integration happens before annotation.
    assert stage_names.index("integration") < stage_names.index("annotation")

    # Confirm annotation happens before clustering.
    assert stage_names.index("annotation") < stage_names.index("clustering")
