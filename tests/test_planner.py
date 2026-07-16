"""Tests for CellQuorum pipeline planning utilities."""

from __future__ import annotations

from cellquorum.backends.base import BackendRequirement, BaseBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import PipelinePlanner, build_pipeline_plan


def test_pipeline_planner_builds_plan_from_default_config() -> None:
    """
    Verify that PipelinePlanner builds a plan from the default config.

    The planner is the first defense against hidden workflow behavior. It should
    translate validated configuration into a clear stage-level plan before any
    expensive analysis runs.
    """

    # Build the default CellQuorum configuration.
    config = CellQuorumConfig()

    # Build a planner from the config.
    planner = PipelinePlanner(config=config)

    # Build the pipeline plan.
    plan = planner.build_plan()

    # Confirm the plan keeps the selected profile.
    assert plan.profile == "standard"

    # Confirm core stages are enabled.
    assert "qc" in plan.enabled_stage_names()

    # Confirm molecular inference is enabled as a gated capability.
    assert "molecular_inference" in plan.enabled_stage_names()

    # Confirm adjudication is enabled as a gated capability.
    assert "adjudication" in plan.enabled_stage_names()

    # Confirm population identity evidence is enabled as a gated capability.
    assert "population_identity" in plan.enabled_stage_names()

    # Confirm cell-cell communication is enabled as a gated capability.
    assert "cell_cell_communication" in plan.enabled_stage_names()

    # Confirm network analysis is enabled as a gated capability.
    assert "network_analysis" in plan.enabled_stage_names()


def test_pipeline_planner_respects_disabled_stage_flags() -> None:
    """
    Verify that disabled stage flags are reflected in the plan.

    Stage booleans should mean "allowed to run" or "not allowed to run." A
    disabled stage should appear as disabled in the plan rather than silently
    disappearing.
    """

    # Build a config with selected stages disabled.
    config = CellQuorumConfig(
        stages={
            "qc": True,
            "preprocessing": True,
            "integration": False,
            "annotation": False,
            "state_scoring": True,
            "discovery": True,
            "subclustering": False,
            "composition": True,
            "differential_expression": True,
            "molecular_inference": True,
            "cell_cell_communication": False,
            "network_analysis": True,
        }
    )

    # Build the pipeline plan.
    plan = build_pipeline_plan(config)

    # Confirm disabled integration appears in disabled stages.
    assert "integration" in plan.disabled_stage_names()

    # Confirm disabled annotation appears in disabled stages.
    assert "annotation" in plan.disabled_stage_names()

    # Confirm disabled subclustering appears in disabled stages.
    assert "subclustering" in plan.disabled_stage_names()

    # Confirm disabled communication appears in disabled stages.
    assert "cell_cell_communication" in plan.disabled_stage_names()

    # Confirm enabled molecular inference remains enabled.
    assert "molecular_inference" in plan.enabled_stage_names()


def test_pipeline_plan_to_dict_is_json_serializable_shape() -> None:
    """
    Verify that PipelinePlan.to_dict returns a stable serializable structure.

    The planner output should later be written to provenance and displayed by the
    CLI. This requires a predictable dictionary shape.
    """

    # Build the default CellQuorum configuration.
    config = CellQuorumConfig()

    # Build the pipeline plan.
    plan = build_pipeline_plan(config)

    # Convert the plan to a dictionary.
    payload = plan.to_dict()

    # Confirm the profile key is present.
    assert payload["profile"] == "standard"

    # Confirm stages are represented as a list.
    assert isinstance(payload["stages"], list)

    # Confirm backend status rows are represented as a list.
    assert isinstance(payload["backend_status_table"], list)

    # Confirm planner warnings are represented as a list.
    assert isinstance(payload["warnings"], list)

    # Confirm the first stage row has the standard keys.
    first_stage = payload["stages"][0]

    # Confirm the stage name key is present.
    assert "name" in first_stage

    # Confirm the enabled key is present.
    assert "enabled" in first_stage

    # Confirm the status key is present.
    assert "status" in first_stage

    # Confirm the reason key is present.
    assert "reason" in first_stage


def test_pipeline_planner_uses_custom_backend_registry() -> None:
    """
    Verify that PipelinePlanner can use a custom backend registry.

    Custom registries are important for tests, staged environments, and future
    profile-specific backend selection.
    """

    # Create a custom backend registry.
    registry = BackendRegistry()

    # Register a simple available backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Register an unavailable GPU backend for warning behavior.
    registry.register(
        BaseBackend(
            name="gpu",
            kind="gpu",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_gpu_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register an unavailable RAPIDS backend for warning behavior.
    registry.register(
        BaseBackend(
            name="rapids",
            kind="rapids",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_rapids_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register an unavailable R backend for warning behavior.
    registry.register(
        BaseBackend(
            name="r",
            kind="r",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_r_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register an unavailable Rscript backend for warning behavior.
    registry.register(
        BaseBackend(
            name="rscript",
            kind="rscript",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_rscript_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Build a config that prefers GPU and allows R.
    config = CellQuorumConfig(
        compute={
            "prefer_gpu": True,
            "fallback_to_cpu": True,
        },
        r={
            "enabled": True,
        },
    )

    # Build a planner using the custom registry.
    planner = PipelinePlanner(config=config, backend_registry=registry)

    # Build the pipeline plan.
    plan = planner.build_plan()

    # Confirm the custom backend status table only contains registered test backends.
    assert {row["name"] for row in plan.backend_status_table} == {
        "python",
        "gpu",
        "rapids",
        "r",
        "rscript",
    }

    # Confirm a GPU warning was generated.
    assert any("GPU acceleration is preferred" in warning for warning in plan.warnings)

    # Confirm an R warning was generated.
    assert any("R-backed methods are enabled" in warning for warning in plan.warnings)


def test_pipeline_planner_does_not_warn_when_gpu_preference_disabled() -> None:
    """
    Verify that GPU warnings are not generated when GPU preference is disabled.

    CPU-first users should not see unnecessary GPU warnings when they explicitly
    configure the run not to prefer GPU acceleration.
    """

    # Create a custom backend registry.
    registry = BackendRegistry()

    # Register unavailable GPU backend.
    registry.register(
        BaseBackend(
            name="gpu",
            kind="gpu",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_gpu_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register unavailable RAPIDS backend.
    registry.register(
        BaseBackend(
            name="rapids",
            kind="rapids",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_rapids_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register available R backend to avoid R warning.
    registry.register(BaseBackend(name="r", kind="r"))

    # Register available Rscript backend to avoid R warning.
    registry.register(BaseBackend(name="rscript", kind="rscript"))

    # Build a config that does not prefer GPU.
    config = CellQuorumConfig(
        compute={
            "backend": "cpu",
            "prefer_gpu": False,
        }
    )

    # Build the pipeline plan.
    plan = build_pipeline_plan(config, backend_registry=registry)

    # Confirm no GPU warning was generated.
    assert not any("GPU acceleration is preferred" in warning for warning in plan.warnings)


def test_pipeline_planner_does_not_warn_when_r_disabled() -> None:
    """
    Verify that R warnings are not generated when R support is disabled.

    Users should be able to run Python-only workflows without R installation
    warnings if they explicitly disable R-backed methods.
    """

    # Create a custom backend registry.
    registry = BackendRegistry()

    # Register available GPU backend to avoid GPU warning.
    registry.register(BaseBackend(name="gpu", kind="gpu"))

    # Register available RAPIDS backend to avoid GPU warning.
    registry.register(BaseBackend(name="rapids", kind="rapids"))

    # Register unavailable R backend.
    registry.register(
        BaseBackend(
            name="r",
            kind="r",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_r_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Register unavailable Rscript backend.
    registry.register(
        BaseBackend(
            name="rscript",
            kind="rscript",
            requirement_list=[
                BackendRequirement(
                    name="cellquorum_missing_rscript_marker_package_12345",
                    requirement_type="python_package",
                    required=True,
                )
            ],
        )
    )

    # Build a config with R disabled.
    config = CellQuorumConfig(
        r={
            "enabled": False,
        }
    )

    # Build the pipeline plan.
    plan = build_pipeline_plan(config, backend_registry=registry)

    # Confirm no R warning was generated.
    assert not any("R-backed methods are enabled" in warning for warning in plan.warnings)
