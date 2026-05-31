"""Smoke tests for the CellQuorum execution-frame pipeline."""

from __future__ import annotations

# Import JSON to inspect written provenance files.
import json

# Import Path for temporary output directory checks.
from pathlib import Path

# Import pandas to inspect provenance CSV artifacts.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import the public API entry point.
from cellquorum import run_pipeline

# Import backend primitives for custom test registries.
from cellquorum.backends.base import BaseBackend

# Import the backend registry.
from cellquorum.backends.registry import BackendRegistry

# Import the validated top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import pipeline bootstrap utilities.
from cellquorum.core.pipeline import (
    PipelineRunResult,
    bootstrap_pipeline_run,
    bootstrap_pipeline_run_from_config_file,
    build_pipeline_context,
    resolve_output_dir,
    write_pipeline_provenance,
)


def build_test_backend_registry() -> BackendRegistry:
    """
    Build a small deterministic backend registry for smoke tests.

    The real default registry checks Python, R, Rscript, GPU, and RAPIDS. That is
    useful in the actual planner, but smoke tests should avoid depending on the
    machine's GPU, R, or RAPIDS installation state. This helper creates a minimal
    available registry that keeps tests focused on pipeline behavior rather than
    local environment variability.

    Returns:
        BackendRegistry containing one available Python backend.
    """

    # Create an empty backend registry.
    registry = BackendRegistry()

    # Register a simple available Python backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Return the deterministic test registry.
    return registry


def build_test_config() -> CellQuorumConfig:
    """
    Build a deterministic CellQuorum config for smoke tests.

    The config disables GPU preference and R support so planner warnings do not
    depend on the local machine. Advanced stages remain enabled as gated
    capabilities because that is the intended default behavior.

    Returns:
        Validated CellQuorumConfig for test runs.
    """

    # Build and return a validated test configuration.
    return CellQuorumConfig(
        project={
            "name": "smoke_project",
            "organism": "human",
            "species_id": 9606,
        },
        run={
            "profile": "standard",
            "random_seed": 42,
        },
        compute={
            "backend": "cpu",
            "prefer_gpu": False,
            "fallback_to_cpu": True,
            "n_jobs": 1,
        },
        r={
            "enabled": False,
        },
    )


def test_resolve_output_dir_prefers_explicit_output_dir(tmp_path: Path) -> None:
    """
    Verify that explicit output_dir overrides config paths.

    Programmatic users and CLI users need a reliable way to force a run into a
    specific output directory regardless of the YAML defaults.
    """

    # Build a test configuration with a run root.
    config = CellQuorumConfig(paths={"run_root": str(tmp_path / "runs")})

    # Resolve the output directory with an explicit override.
    resolved = resolve_output_dir(config, output_dir=tmp_path / "explicit_output")

    # Confirm the explicit output directory was used.
    assert resolved == (tmp_path / "explicit_output").resolve()


def test_resolve_output_dir_uses_config_output_dir(tmp_path: Path) -> None:
    """
    Verify that paths.output_dir is used when no explicit override is supplied.

    This supports project YAML files that define a fixed run output location.
    """

    # Build a test configuration with an explicit output directory.
    config = CellQuorumConfig(paths={"output_dir": str(tmp_path / "configured_output")})

    # Resolve the output directory without an override.
    resolved = resolve_output_dir(config)

    # Confirm the configured output directory was used.
    assert resolved == (tmp_path / "configured_output").resolve()


def test_resolve_output_dir_uses_run_root_and_project_name(tmp_path: Path) -> None:
    """
    Verify that paths.run_root plus project.name forms a default output path.

    This gives users a clean convention when they provide a run root but do not
    specify a run-specific output directory.
    """

    # Build a test configuration with a run root and project name.
    config = CellQuorumConfig(
        project={"name": "project_from_run_root"},
        paths={"run_root": str(tmp_path / "runs")},
    )

    # Resolve the output directory.
    resolved = resolve_output_dir(config)

    # Confirm the project-named directory under run_root was used.
    assert resolved == (tmp_path / "runs" / "project_from_run_root").resolve()


def test_resolve_output_dir_rejects_missing_output_location() -> None:
    """
    Verify that output directory resolution fails clearly when no path is known.

    CellQuorum should not silently write outputs into the current working
    directory because that is bad for reproducibility and cleanup.
    """

    # Build a config with no output_dir and no run_root.
    config = CellQuorumConfig(paths={"output_dir": None, "run_root": None})

    # Confirm a clear error is raised when no output path can be resolved.
    with pytest.raises(ValueError, match="Could not resolve an output directory"):
        resolve_output_dir(config)


def test_build_pipeline_context_creates_standard_run_directories(tmp_path: Path) -> None:
    """
    Verify that build_pipeline_context creates the standard run layout.

    Every future stage should be able to rely on standardized results, figures,
    reports, objects, provenance, logs, and scratch directories.
    """

    # Build a deterministic test config.
    config = build_test_config()

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Build the pipeline context.
    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "context_run",
        backend_registry=registry,
    )

    # Confirm the root directory exists.
    assert context.paths.root.exists()

    # Confirm the results directory exists.
    assert context.paths.results.exists()

    # Confirm the figures directory exists.
    assert context.paths.figures.exists()

    # Confirm the reports directory exists.
    assert context.paths.reports.exists()

    # Confirm the objects directory exists.
    assert context.paths.objects.exists()

    # Confirm the provenance directory exists.
    assert context.paths.provenance.exists()

    # Confirm the logs directory exists.
    assert context.paths.logs.exists()

    # Confirm the scratch directory exists.
    assert context.paths.scratch.exists()

    # Confirm the context stores the configured run seed.
    assert context.random_seed == 42

    # Confirm the context stores project metadata.
    assert context.metadata["project_name"] == "smoke_project"

    # Confirm the injected backend registry was preserved.
    assert context.backend_registry is registry


def test_write_pipeline_provenance_creates_expected_files(tmp_path: Path) -> None:
    """
    Verify that initial pipeline provenance artifacts are written.

    The execution frame should produce a reproducible audit trail before any
    expensive analysis stages are added.
    """

    # Build a deterministic test config.
    config = build_test_config()

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Build the pipeline context.
    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "provenance_run",
        backend_registry=registry,
    )

    # Build the plan using the public pipeline plan path.
    from cellquorum.core.planner import build_pipeline_plan

    # Create a pipeline plan using the deterministic backend registry.
    plan = build_pipeline_plan(config, backend_registry=registry)

    # Write the pipeline provenance files.
    artifact_manager = write_pipeline_provenance(
        config=config,
        plan=plan,
        context=context,
    )

    # Confirm the resolved config JSON exists.
    assert (context.paths.provenance / "resolved_config.json").exists()

    # Confirm the pipeline plan JSON exists.
    assert (context.paths.provenance / "pipeline_plan.json").exists()

    # Confirm the stage plan CSV exists.
    assert (context.paths.provenance / "stage_plan.csv").exists()

    # Confirm the backend status JSON exists.
    assert (context.paths.provenance / "backend_status.json").exists()

    # Confirm the backend status CSV exists.
    assert (context.paths.provenance / "backend_status.csv").exists()

    # Confirm the planner warnings JSON exists.
    assert (context.paths.provenance / "planner_warnings.json").exists()

    # Confirm the run metadata JSON exists.
    assert (context.paths.provenance / "run_metadata.json").exists()

    # Confirm the artifact manifest CSV exists.
    assert (context.paths.provenance / "artifact_manifest.csv").exists()

    # Confirm the artifact manager tracked the manifest artifact.
    assert artifact_manager.artifacts[-1].name == "artifact_manifest"


def test_bootstrap_pipeline_run_returns_structured_result_and_writes_provenance(
    tmp_path: Path,
) -> None:
    """
    Verify that bootstrap_pipeline_run creates a structured execution frame.

    This is the core smoke test for the pipeline bootstrapper. It confirms that
    config validation, context creation, planning, provenance writing, and run
    result packaging all work together.
    """

    # Build a deterministic test config.
    config = build_test_config()

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Bootstrap the pipeline run.
    result = bootstrap_pipeline_run(
        config,
        output_dir=tmp_path / "bootstrap_run",
        backend_registry=registry,
    )

    # Confirm the result has the expected structured type.
    assert isinstance(result, PipelineRunResult)

    # Confirm the validated config was retained.
    assert result.config is config

    # Confirm the pipeline plan contains the standard profile.
    assert result.plan.profile == "standard"

    # Confirm QC is enabled in the stage plan.
    assert "qc" in result.plan.enabled_stage_names()

    # Confirm network analysis is enabled as a gated capability.
    assert "network_analysis" in result.plan.enabled_stage_names()

    # Confirm the context root points to the requested output directory.
    assert result.context.paths.root == (tmp_path / "bootstrap_run").resolve()

    # Confirm the artifact manifest was written.
    assert (result.context.paths.provenance / "artifact_manifest.csv").exists()


def test_bootstrap_pipeline_run_from_config_file_loads_yaml_and_runs(tmp_path: Path) -> None:
    """
    Verify that file-based pipeline bootstrapping works.

    This is the execution path that the CLI and future workflow wrappers will
    use when given a YAML configuration file.
    """

    # Create a temporary config file.
    config_path = tmp_path / "config.yaml"

    # Write a valid deterministic config.
    config_path.write_text(
        """
project:
  name: yaml_bootstrap_project
run:
  profile: publication
  random_seed: 99
compute:
  backend: cpu
  prefer_gpu: false
r:
  enabled: false
""",
        encoding="utf-8",
    )

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Bootstrap the pipeline run from the config file.
    result = bootstrap_pipeline_run_from_config_file(
        config_path,
        output_dir=tmp_path / "yaml_bootstrap_run",
        backend_registry=registry,
    )

    # Confirm the project name was loaded from YAML.
    assert result.config.project.name == "yaml_bootstrap_project"

    # Confirm the publication profile was loaded from YAML.
    assert result.plan.profile == "publication"

    # Confirm the random seed was loaded from YAML.
    assert result.context.random_seed == 99

    # Confirm provenance was written.
    assert (result.context.paths.provenance / "resolved_config.json").exists()


def test_public_run_pipeline_accepts_config_model(tmp_path: Path) -> None:
    """
    Verify that the public API accepts a validated CellQuorumConfig object.

    Programmatic users should be able to construct configs in Python and call
    run_pipeline without writing a YAML file.
    """

    # Build a deterministic test config.
    config = build_test_config()

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Run through the public API.
    result = run_pipeline(
        config,
        output_dir=tmp_path / "api_model_run",
        backend_registry=registry,
    )

    # Confirm the public API returned a structured pipeline result.
    assert isinstance(result, PipelineRunResult)

    # Confirm provenance was written.
    assert (result.context.paths.provenance / "pipeline_plan.json").exists()


def test_public_run_pipeline_accepts_config_dictionary(tmp_path: Path) -> None:
    """
    Verify that the public API accepts a plain configuration dictionary.

    Dictionary input is useful in notebooks, tests, and dynamic programmatic
    workflows where writing YAML would add friction.
    """

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Run through the public API with a dictionary config.
    result = run_pipeline(
        {
            "project": {
                "name": "dict_api_project",
            },
            "compute": {
                "backend": "cpu",
                "prefer_gpu": False,
            },
            "r": {
                "enabled": False,
            },
        },
        output_dir=tmp_path / "api_dict_run",
        backend_registry=registry,
    )

    # Confirm the dictionary was validated into a config.
    assert result.config.project.name == "dict_api_project"

    # Confirm provenance was written.
    assert (result.context.paths.provenance / "run_metadata.json").exists()


def test_public_run_pipeline_accepts_yaml_path(tmp_path: Path) -> None:
    """
    Verify that the public API accepts a YAML config path.

    YAML-path execution is the best reproducible project mode and should share
    the same bootstrap behavior as model and dictionary inputs.
    """

    # Create a temporary YAML config file.
    config_path = tmp_path / "config.yaml"

    # Write a valid config file.
    config_path.write_text(
        """
project:
  name: path_api_project
compute:
  backend: cpu
  prefer_gpu: false
r:
  enabled: false
""",
        encoding="utf-8",
    )

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Run through the public API with a YAML path.
    result = run_pipeline(
        config_path,
        output_dir=tmp_path / "api_path_run",
        backend_registry=registry,
    )

    # Confirm the YAML project name was loaded.
    assert result.config.project.name == "path_api_project"

    # Confirm provenance was written.
    assert (result.context.paths.provenance / "backend_status.csv").exists()


def test_public_run_pipeline_rejects_unsupported_config_input(tmp_path: Path) -> None:
    """
    Verify that the public API rejects unsupported config inputs.

    Clear type errors are important because programmatic API users may pass
    partially constructed objects during development.
    """

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Confirm unsupported config input raises a clear error.
    with pytest.raises(TypeError, match="path, CellQuorumConfig, or dictionary"):
        run_pipeline(
            12345,  # type: ignore[arg-type]
            output_dir=tmp_path / "bad_api_run",
            backend_registry=registry,
        )


def test_pipeline_provenance_files_have_expected_content(tmp_path: Path) -> None:
    """
    Verify that written provenance files contain expected structured content.

    This guards against accidentally writing empty files or incorrectly shaped
    provenance payloads while the execution frame evolves.
    """

    # Build a deterministic test config.
    config = build_test_config()

    # Build a deterministic test backend registry.
    registry = build_test_backend_registry()

    # Bootstrap the pipeline run.
    result = bootstrap_pipeline_run(
        config,
        output_dir=tmp_path / "content_run",
        backend_registry=registry,
    )

    # Load the written pipeline plan JSON.
    plan_payload = json.loads(
        (result.context.paths.provenance / "pipeline_plan.json").read_text(encoding="utf-8")
    )

    # Confirm the plan payload contains the expected profile.
    assert plan_payload["profile"] == "standard"

    # Confirm the plan payload contains stage rows.
    assert len(plan_payload["stages"]) > 0

    # Load the written stage plan CSV.
    stage_plan = pd.read_csv(result.context.paths.provenance / "stage_plan.csv")

    # Confirm QC appears in the stage plan CSV.
    assert "qc" in set(stage_plan["name"])

    # Load the written run metadata JSON.
    run_metadata = json.loads(
        (result.context.paths.provenance / "run_metadata.json").read_text(encoding="utf-8")
    )

    # Confirm the run metadata stores the project-derived run ID.
    assert run_metadata["run_id"] == "smoke_project"

    # Confirm the run metadata stores standardized paths.
    assert "paths" in run_metadata
