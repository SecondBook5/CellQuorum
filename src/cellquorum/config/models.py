"""Validated configuration models for CellQuorum."""

from __future__ import annotations

# Import Path for filesystem-like configuration fields.
from pathlib import Path

# Import Literal for constrained string configuration options.
from typing import Literal

# Import Pydantic primitives for strict runtime validation.
from pydantic import Field, field_validator, model_validator

# Import the shared strict base model used by CellQuorum configuration models.
from cellquorum.config.base import StrictBaseModel

# Import the preprocessing configuration model.
from cellquorum.preprocessing.config import PreprocessingConfig

# Import the QC configuration model.
from cellquorum.qc.config import QCConfig


class ProjectConfig(StrictBaseModel):
    """
    Store project-level metadata.

    Project metadata appears in reports, provenance files, and output directory
    names. It should be biologically descriptive but not hard-code disease logic
    into the package.

    Args:
        name: Short project name used in reports and run metadata.
        description: Optional human-readable project description.
        organism: Organism label, usually human or mouse.
        species_id: NCBI taxonomy identifier, such as 9606 for human.
    """

    # Store a short project name.
    name: str = "cellquorum_project"

    # Store an optional human-readable project description.
    description: str | None = None

    # Store the organism label.
    organism: str = "human"

    # Store the NCBI taxonomy identifier.
    species_id: int = 9606

    @field_validator("name")
    @classmethod
    def validate_project_name(cls, value: str) -> str:
        """
        Validate the project name.

        Project names are used in reports and sometimes in paths, so they should
        not be empty or whitespace-only.

        Args:
            value: Candidate project name.

        Returns:
            Cleaned project name.

        Raises:
            ValueError: If the project name is empty.
        """

        # Strip surrounding whitespace from the project name.
        cleaned = value.strip()

        # Reject empty project names.
        if not cleaned:
            raise ValueError("Project name cannot be empty.")

        # Return the cleaned project name.
        return cleaned


class PathConfig(StrictBaseModel):
    """
    Store input and output path configuration.

    Code should live inside the repository, while large data and run outputs can
    live elsewhere. This config makes that separation explicit without forcing a
    particular filesystem layout.

    Args:
        data_root: Optional root directory for datasets.
        run_root: Optional root directory for CellQuorum run outputs.
        scratch_root: Optional root directory for temporary files.
        manifest: Optional manifest path.
        output_dir: Optional explicit output directory for the current run.
    """

    # Store an optional root directory for datasets.
    data_root: Path | None = None

    # Store an optional root directory for run outputs.
    run_root: Path | None = None

    # Store an optional root directory for temporary files.
    scratch_root: Path | None = None

    # Store an optional sample manifest path.
    manifest: Path | None = None

    # Store an optional explicit output directory.
    output_dir: Path | None = None


class InputConfig(StrictBaseModel):
    """
    Store input data configuration.

    This config describes the primary input data object for a CellQuorum run.
    The first supported input mode is an AnnData h5ad file. File existence is
    checked later by the I/O layer so configs can remain portable across systems.

    Args:
        h5ad: Optional path to an AnnData h5ad file.
        counts_layer: Optional AnnData layer containing raw counts.
    """

    # Store an optional AnnData h5ad input path.
    h5ad: Path | None = None

    # Store an optional AnnData layer containing raw counts.
    counts_layer: str | None = None

    @field_validator("h5ad")
    @classmethod
    def validate_h5ad_suffix(cls, value: Path | None) -> Path | None:
        """
        Validate the AnnData h5ad path suffix.

        Args:
            value: Candidate h5ad path.

        Returns:
            Validated h5ad path or None.

        Raises:
            ValueError: If the path does not end with .h5ad.
        """

        # Allow omitted h5ad paths for programmatic runs.
        if value is None:
            return None

        # Reject unsupported suffixes without checking file existence.
        if value.suffix.lower() != ".h5ad":
            raise ValueError("input.h5ad must point to a '.h5ad' file. " f"Received: {value.name}")

        # Return the validated path.
        return value

    @field_validator("counts_layer")
    @classmethod
    def validate_counts_layer(cls, value: str | None) -> str | None:
        """
        Validate the optional raw-counts layer name.

        Args:
            value: Candidate layer name.

        Returns:
            Cleaned layer name or None.

        Raises:
            ValueError: If the layer name is empty.
        """

        # Allow omitted layer names.
        if value is None:
            return None

        # Strip harmless surrounding whitespace.
        cleaned = value.strip()

        # Reject empty layer names.
        if not cleaned:
            raise ValueError("input.counts_layer cannot be empty.")

        # Return the cleaned layer name.
        return cleaned


class RunConfig(StrictBaseModel):
    """
    Store run-level execution settings.

    These settings control execution identity, reproducibility, and overall
    workflow profile. Profiles keep the user interface simple while allowing
    advanced internal modules to remain organized and gated.

    Args:
        profile: High-level analysis profile.
        run_id: Optional stable run identifier.
        random_seed: Random seed used by stochastic stages.
        overwrite: Whether an existing output directory may be reused.
    """

    # Store the high-level analysis profile.
    profile: Literal[
        "standard",
        "publication",
        "regulatory",
        "communication",
        "trajectory",
        "perturbation",
        "full",
    ] = "standard"

    # Store an optional stable run identifier.
    run_id: str | None = None

    # Store the random seed used by stochastic methods.
    random_seed: int = 1337

    # Store whether existing output directories can be reused.
    overwrite: bool = False

    @field_validator("random_seed")
    @classmethod
    def validate_random_seed(cls, value: int) -> int:
        """
        Validate the random seed.

        Args:
            value: Candidate random seed.

        Returns:
            Validated random seed.

        Raises:
            ValueError: If the seed is negative.
        """

        # Reject negative random seeds.
        if value < 0:
            raise ValueError("random_seed must be non-negative.")

        # Return the validated seed.
        return value


class ComputeConfig(StrictBaseModel):
    """
    Store compute and backend preference settings.

    CellQuorum should be easy to run on CPU but should also treat R and GPU
    acceleration as first-class optional backends. These settings describe
    preferences. Actual availability is checked by the backend registry.

    Args:
        backend: Preferred compute backend.
        prefer_gpu: Whether GPU acceleration should be preferred when available.
        fallback_to_cpu: Whether CPU fallback is allowed when GPU is unavailable.
        n_jobs: Number of CPU workers for stages that support parallel execution.
    """

    # Store the preferred compute backend.
    backend: Literal["auto", "cpu", "gpu", "rapids"] = "auto"

    # Store whether GPU acceleration should be preferred when available.
    prefer_gpu: bool = True

    # Store whether CPU fallback is allowed when GPU is unavailable.
    fallback_to_cpu: bool = True

    # Store the number of CPU workers.
    n_jobs: int = Field(default=1, ge=1)


class RConfig(StrictBaseModel):
    """
    Store R backend preferences.

    R support can run through in-process rpy2 or batch-friendly Rscript. This
    config only expresses preference; the backend registry determines what is
    actually available.

    Args:
        enabled: Whether R-backed methods are allowed.
        preferred_backend: Preferred R execution mode.
        fallback_to_rscript: Whether Rscript fallback is allowed.
        rscript_path: Rscript executable name or path.
        timeout_seconds: Timeout for lightweight R checks.
    """

    # Store whether R-backed methods are allowed.
    enabled: bool = True

    # Store the preferred R backend.
    preferred_backend: Literal["auto", "r", "rscript"] = "auto"

    # Store whether Rscript fallback is allowed.
    fallback_to_rscript: bool = True

    # Store the Rscript executable path.
    rscript_path: str = "Rscript"

    # Store timeout for lightweight R backend checks.
    timeout_seconds: int = Field(default=30, ge=1)


class ReportConfig(StrictBaseModel):
    """
    Store final report generation settings.

    The final report is a required part of CellQuorum's publication-grade design.
    These settings control which report formats should be attempted.

    Args:
        enabled: Whether report generation is enabled.
        html: Whether to render an HTML report.
        markdown: Whether to render a Markdown report.
        pdf: Whether to attempt PDF report rendering.
        fail_on_report_error: Whether report failure should mark the run failed.
    """

    # Store whether report generation is enabled.
    enabled: bool = True

    # Store whether an HTML report should be rendered.
    html: bool = True

    # Store whether a Markdown report should be rendered.
    markdown: bool = True

    # Store whether PDF report rendering should be attempted.
    pdf: bool = False

    # Store whether report generation failure should fail the run.
    fail_on_report_error: bool = False


class StageSelectionConfig(StrictBaseModel):
    """
    Store high-level stage enablement flags.

    These flags do not mean that every stage blindly runs on every dataset.
    They mean each stage is allowed to run if its data requirements, method
    assumptions, sample support, and backend requirements are satisfied.

    Args:
        qc: Whether quality control is enabled.
        preprocessing: Whether preprocessing is enabled.
        integration: Whether integration is enabled.
        annotation: Whether annotation is enabled.
        state_scoring: Whether state scoring is enabled.
        discovery: Whether automatic discovery is enabled.
        subclustering: Whether subclustering is enabled.
        composition: Whether composition analysis is enabled.
        differential_expression: Whether differential expression is enabled.
        molecular_inference: Whether molecular inference is enabled.
        cell_cell_communication: Whether communication analysis is enabled.
        network_analysis: Whether network analysis is enabled.
    """

    # Store whether quality control is enabled.
    qc: bool = True

    # Store whether preprocessing is enabled.
    preprocessing: bool = True

    # Store whether integration is enabled.
    integration: bool = True

    # Store whether annotation is enabled.
    annotation: bool = True

    # Store whether state scoring is enabled.
    state_scoring: bool = True

    # Store whether automatic discovery is enabled.
    discovery: bool = True

    # Store whether subclustering is enabled.
    subclustering: bool = True

    # Store whether composition analysis is enabled.
    composition: bool = True

    # Store whether differential expression is enabled.
    differential_expression: bool = True

    # Store whether molecular inference is enabled as a gated capability.
    molecular_inference: bool = True

    # Store whether cell-cell communication analysis is enabled as a gated capability.
    cell_cell_communication: bool = True

    # Store whether network analysis is enabled as a gated capability.
    network_analysis: bool = True


class CellQuorumConfig(StrictBaseModel):
    """
    Store the validated top-level CellQuorum configuration.

    Hydra/OmegaConf should compose flexible user-facing YAML files, but the final
    resolved configuration must become this Pydantic model before execution.
    This object is the authoritative runtime contract for the pipeline.

    Args:
        project: Project-level metadata.
        paths: Input/output path settings.
        input: Input data settings.
        run: Run-level execution settings.
        compute: Compute backend preferences.
        r: R backend preferences.
        report: Final report settings.
        stages: Major stage enablement flags.
        qc: Quality-control settings.
    """

    # Store project-level metadata.
    project: ProjectConfig = Field(default_factory=ProjectConfig)

    # Store input/output path settings.
    paths: PathConfig = Field(default_factory=PathConfig)

    # Store input data settings.
    input: InputConfig = Field(default_factory=InputConfig)

    # Store run-level execution settings.
    run: RunConfig = Field(default_factory=RunConfig)

    # Store compute backend preferences.
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    # Store R backend preferences.
    r: RConfig = Field(default_factory=RConfig)

    # Store final report settings.
    report: ReportConfig = Field(default_factory=ReportConfig)

    # Store major stage enablement flags.
    stages: StageSelectionConfig = Field(default_factory=StageSelectionConfig)

    # Store quality-control settings.
    qc: QCConfig = Field(default_factory=QCConfig)

    # Store preprocessing settings.
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @model_validator(mode="after")
    def validate_backend_fallbacks(self) -> CellQuorumConfig:
        """
        Validate consistency among compute backend settings.

        Returns:
            Validated CellQuorumConfig.

        Raises:
            ValueError: If an unavailable fallback policy is requested.
        """

        # Reject automatic backend selection when CPU fallback is disabled.
        if self.compute.backend == "auto" and not self.compute.fallback_to_cpu:
            raise ValueError(
                "compute.backend='auto' requires compute.fallback_to_cpu=True. "
                "Use compute.backend='gpu' or 'rapids' for explicit GPU-only execution."
            )

        # Return the validated config object.
        return self
