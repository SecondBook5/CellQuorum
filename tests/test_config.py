"""Tests for CellQuorum validated configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cellquorum.config.models import (
    CellQuorumConfig,
    ComputeConfig,
    PathConfig,
    ProjectConfig,
    RConfig,
    ReportConfig,
    RunConfig,
    StageSelectionConfig,
)


def test_default_cellquorum_config_builds_successfully() -> None:
    """
    Verify that the top-level CellQuorum config can be built with defaults.

    A reusable pipeline needs a valid default configuration so the CLI, planner,
    and smoke tests can start from a known baseline before project-specific YAML
    overrides are introduced.
    """

    # Build the default top-level configuration.
    config = CellQuorumConfig()

    # Confirm the default project name is present.
    assert config.project.name == "cellquorum_project"

    # Confirm the default organism is human.
    assert config.project.organism == "human"

    # Confirm the default species ID is human.
    assert config.project.species_id == 9606

    # Confirm the default run profile is standard.
    assert config.run.profile == "standard"

    # Confirm the default compute backend is automatic.
    assert config.compute.backend == "auto"

    # Confirm report generation is enabled by default.
    assert config.report.enabled is True

    # Confirm quality control is enabled by default.
    assert config.stages.qc is True

    # Confirm molecular inference is enabled as a gated capability by default.
    assert config.stages.molecular_inference is True

    # Confirm adjudication is enabled as a gated capability.
    assert config.stages.adjudication is True

    # Confirm population identity evidence is enabled as a gated capability.
    assert config.stages.population_identity is True

    # Confirm cell-cell communication is enabled as a gated capability by default.
    assert config.stages.cell_cell_communication is True

    # Confirm network analysis is enabled as a gated capability by default.
    assert config.stages.network_analysis is True


def test_project_config_strips_valid_project_name() -> None:
    """
    Verify that ProjectConfig strips whitespace from project names.

    Project names appear in reports and provenance, so surrounding whitespace
    should be cleaned without forcing the user to fix harmless formatting.
    """

    # Build a project config with surrounding whitespace.
    config = ProjectConfig(name="  lymphedema_project  ")

    # Confirm the project name was cleaned.
    assert config.name == "lymphedema_project"


def test_project_config_rejects_empty_project_name() -> None:
    """
    Verify that ProjectConfig rejects empty project names.

    Empty project names produce unclear reports and poor provenance metadata, so
    they should fail validation immediately.
    """

    # Confirm whitespace-only project names raise a validation error.
    with pytest.raises(ValidationError, match="Project name cannot be empty"):
        ProjectConfig(name="   ")


def test_path_config_accepts_path_values() -> None:
    """
    Verify that PathConfig accepts path-like values.

    CellQuorum separates code, data, run outputs, and scratch paths. This test
    confirms that path fields can be supplied as strings and normalized into Path
    objects by Pydantic.
    """

    # Build a path config using string paths. These are deliberately synthetic: the
    # test only exercises Pydantic's string-to-Path coercion and never touches the
    # filesystem, so naming a real machine's drive layout here would falsely imply the
    # test depends on it.
    config = PathConfig(
        data_root="/data/cellquorum/inputs",
        run_root="/data/cellquorum/runs",
        scratch_root="/data/cellquorum/scratch",
        manifest="examples/minimal_scrna/manifest.csv",
        output_dir="runs/test_run",
    )

    # Confirm the data root was converted to a Path.
    assert config.data_root == Path("/data/cellquorum/inputs")

    # Confirm the run root was converted to a Path.
    assert config.run_root == Path("/data/cellquorum/runs")

    # Confirm the scratch root was converted to a Path.
    assert config.scratch_root == Path("/data/cellquorum/scratch")

    # Confirm the manifest path was converted to a Path.
    assert config.manifest == Path("examples/minimal_scrna/manifest.csv")

    # Confirm the output directory was converted to a Path.
    assert config.output_dir == Path("runs/test_run")


def test_run_config_accepts_supported_profiles() -> None:
    """
    Verify that RunConfig accepts all supported analysis profiles.

    Profiles keep CellQuorum easy to use while allowing advanced modules to be
    organized behind simple user-facing modes.
    """

    # Define every supported run profile.
    profiles = [
        "standard",
        "publication",
        "regulatory",
        "communication",
        "trajectory",
        "perturbation",
        "full",
    ]

    # Confirm each supported profile validates successfully.
    for profile in profiles:
        # Build the run config for this profile.
        config = RunConfig(profile=profile)

        # Confirm the profile was retained.
        assert config.profile == profile


def test_run_config_rejects_unsupported_profile() -> None:
    """
    Verify that RunConfig rejects unsupported profiles.

    Unsupported profile names usually indicate a typo in YAML or a CLI override,
    so they should fail before pipeline execution begins.
    """

    # Confirm unsupported profile names fail validation.
    with pytest.raises(ValidationError):
        RunConfig(profile="everything")  # type: ignore[arg-type]


def test_run_config_rejects_negative_random_seed() -> None:
    """
    Verify that RunConfig rejects negative random seeds.

    Random seeds should be non-negative so stochastic stages can use them
    consistently across Python, NumPy, Scanpy, scVI, and future backends.
    """

    # Confirm negative random seeds raise a validation error.
    with pytest.raises(ValidationError, match="random_seed must be non-negative"):
        RunConfig(random_seed=-1)


def test_compute_config_accepts_supported_backends() -> None:
    """
    Verify that ComputeConfig accepts supported compute backends.

    Compute backend preferences are later checked against actual backend
    availability by the backend registry.
    """

    # Define every supported compute backend.
    backends = ["auto", "cpu", "gpu", "rapids"]

    # Confirm each backend validates successfully.
    for backend in backends:
        # Build the compute config for this backend.
        config = ComputeConfig(backend=backend)

        # Confirm the backend was retained.
        assert config.backend == backend


def test_compute_config_rejects_unsupported_backend() -> None:
    """
    Verify that ComputeConfig rejects unsupported backend names.

    Backend typos should fail during config validation rather than causing
    confusing planner or execution errors later.
    """

    # Confirm unsupported compute backend names fail validation.
    with pytest.raises(ValidationError):
        ComputeConfig(backend="cuda")  # type: ignore[arg-type]


def test_compute_config_rejects_invalid_n_jobs() -> None:
    """
    Verify that ComputeConfig rejects invalid worker counts.

    Worker counts should be positive integers because zero workers is not a
    meaningful execution mode.
    """

    # Confirm n_jobs=0 raises a validation error.
    with pytest.raises(ValidationError):
        ComputeConfig(n_jobs=0)

    # 0 and negatives must stay rejected even though the field now accepts a
    # string: joblib and dask both read them as "all cores", so letting one
    # through would give a step MORE parallelism than the config asked for.
    with pytest.raises(ValidationError):
        ComputeConfig(n_jobs=-1)
    with pytest.raises(ValidationError):
        ComputeConfig(n_jobs="all")  # type: ignore[arg-type]

    # "auto" is the one non-integer accepted, and the default.
    assert ComputeConfig().n_jobs == "auto"
    assert ComputeConfig(n_jobs="auto").n_jobs == "auto"
    assert ComputeConfig(n_jobs=4).n_jobs == 4


def test_r_config_accepts_supported_preferred_backends() -> None:
    """
    Verify that RConfig accepts supported R backend preferences.

    R can be routed through automatic selection, rpy2, or Rscript depending on
    environment availability and HPC requirements.
    """

    # Define every supported R backend preference.
    backends = ["auto", "r", "rscript"]

    # Confirm each R backend preference validates successfully.
    for backend in backends:
        # Build the R config for this backend preference.
        config = RConfig(preferred_backend=backend)

        # Confirm the preference was retained.
        assert config.preferred_backend == backend


def test_r_config_rejects_unsupported_preferred_backend() -> None:
    """
    Verify that RConfig rejects unsupported R backend names.

    Invalid R backend settings should fail before the planner tries to choose a
    backend.
    """

    # Confirm unsupported R backend names fail validation.
    with pytest.raises(ValidationError):
        RConfig(preferred_backend="renv")  # type: ignore[arg-type]


def test_r_config_rejects_invalid_timeout() -> None:
    """
    Verify that RConfig rejects invalid timeout values.

    Timeout values should be positive so backend checks do not behave
    unexpectedly.
    """

    # Confirm a zero-second timeout fails validation.
    with pytest.raises(ValidationError):
        RConfig(timeout_seconds=0)


def test_report_config_defaults_to_html_and_markdown() -> None:
    """
    Verify that ReportConfig enables practical report outputs by default.

    HTML and Markdown should be generated by default, while PDF can remain
    optional because it may need additional system dependencies.
    """

    # Build the default report config.
    config = ReportConfig()

    # Confirm report generation is enabled.
    assert config.enabled is True

    # Confirm HTML report generation is enabled.
    assert config.html is True

    # Confirm Markdown report generation is enabled.
    assert config.markdown is True

    # Confirm PDF report generation is disabled by default.
    assert config.pdf is False

    # Confirm report failures do not fail runs by default during early development.
    assert config.fail_on_report_error is False


def test_stage_selection_config_defaults_to_major_capabilities_enabled() -> None:
    """
    Verify that StageSelectionConfig enables major analysis capabilities by default.

    These defaults mean the stages are allowed to run if their method gates,
    metadata requirements, sample support requirements, and backend requirements
    are satisfied. They do not mean every advanced method should blindly run on
    every dataset.
    """

    # Build the default stage selection config.
    config = StageSelectionConfig()

    # Confirm QC is enabled by default.
    assert config.qc is True

    # Confirm preprocessing is enabled by default.
    assert config.preprocessing is True

    # Confirm integration is enabled by default.
    assert config.integration is True

    # Confirm annotation is enabled by default.
    assert config.annotation is True

    # Confirm state scoring is enabled by default.
    assert config.state_scoring is True

    # Confirm discovery is enabled by default.
    assert config.discovery is True

    # Confirm subclustering is enabled by default.
    assert config.subclustering is True

    # Confirm composition analysis is enabled by default.
    assert config.composition is True

    # Confirm population identity evidence is enabled by default.
    assert config.population_identity is True

    # Confirm differential expression is enabled by default.
    assert config.differential_expression is True

    # Confirm molecular inference is enabled as a gated capability by default.
    assert config.molecular_inference is True

    # Confirm cell-cell communication is enabled as a gated capability by default.
    assert config.cell_cell_communication is True

    # Confirm network analysis is enabled as a gated capability by default.
    assert config.network_analysis is True


def test_cellquorum_config_rejects_unknown_top_level_fields() -> None:
    """
    Verify that CellQuorumConfig rejects unknown top-level fields.

    Strict validation prevents misspelled YAML keys from silently changing or
    skipping pipeline behavior.
    """

    # Confirm unknown top-level keys fail validation.
    with pytest.raises(ValidationError):
        CellQuorumConfig(unexpected_key=True)  # type: ignore[call-arg]


def test_nested_config_rejects_unknown_fields() -> None:
    """
    Verify that nested configuration models reject unknown fields.

    Strict validation should apply at every config level, not only at the top
    level.
    """

    # Confirm unknown nested project keys fail validation.
    with pytest.raises(ValidationError):
        CellQuorumConfig(project={"name": "test", "bad_field": "bad"})  # type: ignore[arg-type]


def test_cellquorum_config_rejects_auto_backend_without_cpu_fallback() -> None:
    """
    Verify that automatic backend selection requires CPU fallback.

    If backend selection is automatic, the planner needs permission to fall back
    to CPU when GPU/RAPIDS are unavailable. Explicit GPU-only execution should
    use compute.backend='gpu' or compute.backend='rapids'.
    """

    # Confirm the inconsistent backend fallback policy fails validation.
    with pytest.raises(ValidationError, match="compute.backend='auto'"):
        CellQuorumConfig(
            compute={
                "backend": "auto",
                "fallback_to_cpu": False,
            }
        )


def test_cellquorum_config_allows_explicit_gpu_without_cpu_fallback() -> None:
    """
    Verify that explicit GPU-only execution can disable CPU fallback.

    Users should be able to request GPU-only execution intentionally, as long as
    the backend selection is explicit rather than automatic.
    """

    # Build a config with explicit GPU backend and no CPU fallback.
    config = CellQuorumConfig(
        compute={
            "backend": "gpu",
            "fallback_to_cpu": False,
        }
    )

    # Confirm the explicit GPU backend was retained.
    assert config.compute.backend == "gpu"

    # Confirm CPU fallback is disabled.
    assert config.compute.fallback_to_cpu is False


def test_cellquorum_config_accepts_nested_model_instances() -> None:
    """
    Verify that CellQuorumConfig accepts nested model instances.

    Programmatic API users may construct nested Pydantic models directly instead
    of loading YAML through Hydra/OmegaConf.
    """

    # Build the top-level config from nested model instances.
    config = CellQuorumConfig(
        project=ProjectConfig(name="programmatic_project"),
        run=RunConfig(profile="publication", random_seed=42),
        compute=ComputeConfig(backend="cpu", prefer_gpu=False),
        r=RConfig(enabled=False),
        report=ReportConfig(pdf=True),
        stages=StageSelectionConfig(molecular_inference=True),
    )

    # Confirm the project model was retained.
    assert config.project.name == "programmatic_project"

    # Confirm the run model was retained.
    assert config.run.profile == "publication"

    # Confirm the compute model was retained.
    assert config.compute.backend == "cpu"

    # Confirm the R model was retained.
    assert config.r.enabled is False

    # Confirm the report model was retained.
    assert config.report.pdf is True

    # Confirm the stage model was retained.
    assert config.stages.molecular_inference is True
