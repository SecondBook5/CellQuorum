"""Validated configuration models for CellQuorum."""

from __future__ import annotations

# Import Path for filesystem-like configuration fields.
from pathlib import Path

# Import Literal for constrained string configuration options.
from typing import Literal

# Import Pydantic primitives for strict runtime validation.
from pydantic import Field, field_validator, model_validator

# Import the adjudication configuration model.
from cellquorum.adjudication.config import AdjudicationConfig

# Import the ambient-correction configuration model.
from cellquorum.ambient_correction.config import AmbientCorrectionConfig

# Import the annotation configuration model.
from cellquorum.annotation.config import AnnotationConfig

# Import the annotation-consensus configuration model.
from cellquorum.annotation_consensus.config import AnnotationConsensusConfig

# Import the annotation-diagnostics configuration model.
from cellquorum.annotation_diagnostics.config import AnnotationDiagnosticsConfig

# Import the cell-cell-communication configuration model.
from cellquorum.cell_cell_communication.config import CellCellCommunicationConfig

# Import the ccc-network (topology + curvature) configuration model.
from cellquorum.cell_cell_communication.network.config import CCCNetworkConfig

# Import the CCC-visualization configuration model.
from cellquorum.cell_cell_communication.viz.config import CccVizConfig

# Import the coexpression configuration model.
from cellquorum.coexpression.config import CoexpressionConfig

# Import the shared strict base model used by CellQuorum configuration models.
from cellquorum.config.base import StrictBaseModel

# Import the central cohort schema (structural keys declared once).
from cellquorum.config.cohort import CohortConfig

# Import the markers, design, and contrasts configuration models.
from cellquorum.config.design import ContrastsConfig, DesignConfig
from cellquorum.config.markers import MarkersConfig

# Import the differential-expression-visualization configuration model.
from cellquorum.de_viz.config import DeVizConfig

# Import the differential-abundance configuration model.
from cellquorum.differential_abundance.config import DifferentialAbundanceConfig

# Import the differential-expression configuration model.
from cellquorum.differential_expression.config import DifferentialExpressionConfig

# Import the embeddings configuration model.
from cellquorum.embeddings.config import EmbeddingsConfig

# Import the enrichment configuration model.
from cellquorum.enrichment.config import EnrichmentConfig

# Import the enrichment-visualization configuration model.
from cellquorum.enrichment.viz.config import EnrichmentVizConfig

# Import the feature-selection configuration model.
from cellquorum.feature_selection.config import FeatureSelectionConfig

# Import the GRN configuration model.
from cellquorum.grn.config import GrnConfig

# Import the integration configuration model.
from cellquorum.integration.config import IntegrationConfig

# Import the integration-benchmark configuration model.
from cellquorum.integration_benchmark.config import IntegrationBenchmarkConfig

# Import the perturbation configuration model.
from cellquorum.perturbation.config import PerturbationConfig

# Import the population-identity configuration model.
from cellquorum.population_identity.config import PopulationIdentityConfig

# Import the preprocessing configuration model.
from cellquorum.preprocessing.config import PreprocessingConfig

# Import the QC configuration model.
from cellquorum.qc.config import QCConfig

# Import the reference-mapping configuration model.
from cellquorum.reference_mapping.config import ReferenceMappingConfig

# Import the subclustering configuration model.
from cellquorum.subclustering.config import SubclusteringConfig

# Import the trajectory configuration model.
from cellquorum.trajectory.config import TrajectoryConfig

# Import the trajectory-visualization configuration model.
from cellquorum.trajectory.viz.config import TrajectoryVizConfig


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


class InputSubsetConfig(StrictBaseModel):
    """
    Restrict the input AnnData to rows whose ``column`` value is in ``values``.

    A per-cell-type (or per-condition) hypothesis analyzes a slice of a shared
    annotated object, never the whole thing. Declaring that slice here makes the
    restriction a first-class, reproducible part of the run instead of a manual
    pre-slice someone has to remember: the loader applies it in backed mode so a
    large global object is never fully materialized (only the matching slice is
    read into memory), and records n_before/n_after in run provenance so the cut
    is never a silent step.

    Args:
        column: obs column to filter on (e.g. ``cell_type``).
        values: keep rows whose ``column`` value is one of these.
    """

    # Store the obs column to filter rows on.
    column: str

    # Store the accepted values; a row is kept when its column value is in here.
    values: list[str] = Field(min_length=1)

    @field_validator("column")
    @classmethod
    def validate_column(cls, value: str) -> str:
        """Reject an empty subset column name."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("input.subset.column cannot be empty.")
        return cleaned

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[str]) -> list[str]:
        """Strip values and reject empty entries."""

        cleaned = [str(item).strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("input.subset.values cannot contain empty strings.")
        return cleaned


class InputConfig(StrictBaseModel):
    """
    Store input data configuration.

    This config describes the primary input data object for a CellQuorum run.
    The first supported input mode is an AnnData h5ad file. File existence is
    checked later by the I/O layer so configs can remain portable across systems.

    Args:
        h5ad: Optional path to an AnnData h5ad file.
        counts_layer: Optional AnnData layer containing raw counts.
        subset: Optional row restriction applied at load time (backed mode).
    """

    # Store an optional AnnData h5ad input path.
    h5ad: Path | None = None

    # Store an optional AnnData layer containing raw counts.
    counts_layer: str | None = None

    # Store an optional row restriction (e.g. cell_type == Fibroblasts).
    subset: InputSubsetConfig | None = None

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
            raise ValueError(f"input.h5ad must point to a '.h5ad' file. Received: {value.name}")

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
        resume: Whether completed stages may be skipped on rerun when their
            input fingerprint matches and their recorded artifacts still exist.
        verbose: Whether to produce runtime output.
        log_level: Output detail level.
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

    # Store whether completed stages may be skipped on rerun (opt-in resume).
    resume: bool = False

    # Store whether to produce runtime output.
    verbose: bool = True

    # Store the output detail level.
    log_level: Literal["quiet", "normal", "verbose"] = "normal"

    # Store whether the final in-memory AnnData is written to disk at the end of
    # a run. Without this, a from-scratch run threads the object through stages
    # in memory and discards it — leaving no annotated deliverable on disk.
    write_final_object: bool = True

    # Store the filename (under the run's objects dir) for the final AnnData.
    final_object_name: str = "final_annotated.h5ad"

    # Store whether execution continues past a failed stage. Default False keeps
    # the fail-fast contract for normal runs. Set True for an unattended canary
    # so one broken optional stage does not halt the whole pipeline: every stage
    # is attempted, all failures are recorded, and a later resume rerun only
    # re-executes the stages that failed.
    continue_on_stage_failure: bool = False

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


class DimensionalityConfig(StrictBaseModel):
    """
    Store dimensionality-reduction (PCA) settings.

    Args:
        enabled: Whether the dimensionality stage may run.
        method: Reduction method name (registry key). Currently "pca".
        input_layer: Layer to run PCA on; must be a log-normalized layer.
            Defaults to the preprocessing normalized output.
        n_pcs: Number of principal components, or "auto" to select via the
            variance-ratio knee.
        max_pcs: Upper bound on components computed and considered for "auto".
        use_highly_variable: Whether to restrict PCA to highly-variable genes.
        random_state: Seed for deterministic PCA.
    """

    # Store whether the dimensionality stage may run.
    enabled: bool = True

    # Store the reduction method registry key.
    method: str = "pca"

    # Store the layer to run PCA on; must be a log-normalized layer.
    # Defaults to the preprocessing normalized output.
    input_layer: str = "cellquorum_normalized"

    # Store the component count, or "auto" for knee-based selection.
    n_pcs: int | str = "auto"

    # Store the upper bound on components for auto selection.
    max_pcs: int = 50

    # Store whether PCA is restricted to highly-variable genes.
    use_highly_variable: bool = False

    # Store the PCA random seed.
    random_state: int = 0


class ClusteringConfig(StrictBaseModel):
    """
    Store clustering (neighbors + Leiden) settings.

    Args:
        enabled: Whether the clustering stage may run.
        method: Clustering method name (registry key). Currently "leiden".
        n_neighbors: Neighborhood size for the kNN graph.
        resolution: Leiden resolution.
        random_state: Seed for deterministic neighbors/Leiden.
        key_added: obs column that receives cluster labels.
        use_rep: Embedding to cluster on (set to the integration output, e.g.
            "X_pca_harmony", when integration runs; defaults to raw PCA).
    """

    # Store whether the clustering stage may run.
    enabled: bool = True

    # Store the clustering method registry key.
    method: str = "leiden"

    # Store the kNN neighborhood size.
    n_neighbors: int = 15

    # Store the Leiden resolution.
    resolution: float = 1.0

    # Store the clustering random seed.
    random_state: int = 0

    # Store the obs column that receives cluster labels.
    key_added: str = "leiden"

    # Embedding to cluster on (set to the integration output, e.g.
    # "X_pca_harmony", when integration runs; defaults to raw PCA).
    use_rep: str = "X_pca"


class StageSelectionConfig(StrictBaseModel):
    """
    Store high-level stage enablement flags.

    These flags do not mean that every stage blindly runs on every dataset.
    They mean each stage is allowed to run if its data requirements, method
    assumptions, sample support, and backend requirements are satisfied.

    Args:
        ambient_correction: Whether ambient correction is enabled.
        qc: Whether quality control is enabled.
        preprocessing: Whether preprocessing is enabled.
        dimensionality: Whether dimensionality reduction is enabled.
        clustering: Whether clustering is enabled.
        integration: Whether integration is enabled.
        annotation: Whether annotation is enabled.
        population_identity: Whether population/state identity evidence is enabled.
        state_scoring: Whether state scoring is enabled.
        discovery: Whether automatic discovery is enabled.
        subclustering: Whether subclustering is enabled.
        adjudication: Whether cluster/state adjudication is enabled.
        composition: Whether composition analysis is enabled.
        differential_expression: Whether differential expression is enabled.
        coexpression: Whether co-expression (hdWGCNA) is enabled.
        grn: Whether GRN (pySCENIC) regulon inference is enabled.
        perturbation: Whether in-silico perturbation (CellOracle) is enabled.
        molecular_inference: Whether molecular inference is enabled.
        cell_cell_communication: Whether communication analysis is enabled.
        network_analysis: Whether network analysis is enabled.
    """

    # Store whether ambient correction is enabled.
    ambient_correction: bool = False

    # Store whether quality control is enabled.
    qc: bool = True

    # Store whether preprocessing is enabled.
    preprocessing: bool = True

    # Store whether the feature-selection (HVG) stage may run.
    feature_selection: bool = True

    # Store whether dimensionality reduction is enabled.
    dimensionality: bool = True

    # Store whether clustering is enabled.
    clustering: bool = True

    # Store whether integration is enabled.
    integration: bool = True

    # Store whether annotation is enabled.
    annotation: bool = True

    # Store whether annotation-diagnostics evaluation is enabled.
    annotation_diagnostics: bool = True

    # Store whether annotation-consensus reconciliation is enabled.
    annotation_consensus: bool = True

    # Store whether reference mapping is enabled.
    reference_mapping: bool = True

    # Store whether integration-benchmark evaluation is enabled.
    integration_benchmark: bool = True

    # Store whether integration-gate filtering is enabled (reserved).
    integration_gate: bool = False

    # Store whether population/state identity evidence output is enabled.
    population_identity: bool = True

    # Store whether state scoring is enabled.
    state_scoring: bool = True

    # Store whether automatic discovery is enabled.
    discovery: bool = True

    # Store whether subclustering is enabled.
    subclustering: bool = True

    # Store whether cluster/state adjudication is enabled.
    adjudication: bool = True

    # Store whether composition analysis is enabled.
    composition: bool = True

    # Store whether differential expression is enabled.
    differential_expression: bool = True

    # Store whether co-expression (hdWGCNA) is enabled.
    coexpression: bool = True

    # Store whether GRN (pySCENIC) regulon inference is enabled.
    grn: bool = True

    # Store whether in-silico perturbation (CellOracle) is enabled.
    perturbation: bool = True

    # Store whether differential abundance is enabled.
    differential_abundance: bool = True

    # Store whether enrichment / pathway-activity is enabled.
    enrichment: bool = True

    # Store whether enrichment visualization is enabled.
    enrichment_viz: bool = True

    # Store whether differential-expression visualization is enabled.
    de_viz: bool = True

    # Store whether CCC visualization is enabled.
    ccc_viz: bool = True

    # Store whether the embeddings stage (UMAP/PHATE/PAGA + overlays) is enabled.
    embeddings: bool = True

    # Store whether the trajectory stage (velocity/RNA-velocity/potency) is enabled.
    trajectory: bool = True

    # Store whether trajectory visualization (figures from producer outputs) is enabled.
    trajectory_viz: bool = True

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
        ambient_correction: Ambient-correction settings.
        qc: Quality-control settings.
        preprocessing: Preprocessing settings.
        dimensionality: Dimensionality-reduction settings.
        clustering: Clustering settings.
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

    # Store ambient-correction settings.
    ambient_correction: AmbientCorrectionConfig = Field(default_factory=AmbientCorrectionConfig)

    # Store quality-control settings.
    qc: QCConfig = Field(default_factory=QCConfig)

    # Store preprocessing settings.
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    # Store feature-selection settings.
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)

    # Store dimensionality-reduction settings.
    dimensionality: DimensionalityConfig = Field(default_factory=DimensionalityConfig)

    # Store clustering settings.
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)

    # Store integration settings.
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)

    # Store annotation settings.
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)

    # Store annotation-diagnostics evaluation settings.
    annotation_diagnostics: AnnotationDiagnosticsConfig = Field(
        default_factory=AnnotationDiagnosticsConfig
    )

    # Store annotation-consensus reconciliation settings.
    annotation_consensus: AnnotationConsensusConfig = Field(
        default_factory=AnnotationConsensusConfig
    )

    # Store integration-benchmark evaluation settings.
    integration_benchmark: IntegrationBenchmarkConfig = Field(
        default_factory=IntegrationBenchmarkConfig
    )

    # Store reference-mapping settings.
    reference_mapping: ReferenceMappingConfig = Field(default_factory=ReferenceMappingConfig)

    # Store population/state identity evidence settings.
    population_identity: PopulationIdentityConfig = Field(default_factory=PopulationIdentityConfig)

    # Store subclustering settings.
    subclustering: SubclusteringConfig = Field(default_factory=SubclusteringConfig)

    # Store adjudication settings.
    adjudication: AdjudicationConfig = Field(default_factory=AdjudicationConfig)

    # Store differential-abundance settings.
    differential_abundance: DifferentialAbundanceConfig = Field(
        default_factory=DifferentialAbundanceConfig
    )

    # Store differential-expression settings.
    differential_expression: DifferentialExpressionConfig = Field(
        default_factory=DifferentialExpressionConfig
    )

    # Store co-expression (hdWGCNA) settings.
    coexpression: CoexpressionConfig = Field(default_factory=CoexpressionConfig)

    # Store GRN (pySCENIC) settings.
    grn: GrnConfig = Field(default_factory=GrnConfig)

    # Store in-silico perturbation (CellOracle) settings.
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)

    # Store enrichment / pathway-activity settings.
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)

    # Store enrichment-visualization settings.
    enrichment_viz: EnrichmentVizConfig = Field(default_factory=EnrichmentVizConfig)

    # Store differential-expression-visualization settings.
    de_viz: DeVizConfig = Field(default_factory=DeVizConfig)

    # Store CCC-visualization settings.
    ccc_viz: CccVizConfig = Field(default_factory=CccVizConfig)

    # Store embeddings settings.
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    # Store trajectory settings.
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)

    # Store trajectory-visualization settings.
    trajectory_viz: TrajectoryVizConfig = Field(default_factory=TrajectoryVizConfig)

    # Store cell-cell-communication settings.
    cell_cell_communication: CellCellCommunicationConfig = Field(
        default_factory=CellCellCommunicationConfig
    )

    # Store ccc_network (topology + curvature) settings.
    ccc_network: CCCNetworkConfig = Field(default_factory=CCCNetworkConfig)

    # Store named marker gene panels.
    markers: MarkersConfig = Field(default_factory=MarkersConfig)

    # Store the central cohort schema (structural obs keys declared once).
    cohort: CohortConfig = Field(default_factory=CohortConfig)

    # Store experimental-design settings.
    design: DesignConfig = Field(default_factory=DesignConfig)

    # Store named case/control contrasts.
    contrasts: ContrastsConfig = Field(default_factory=ContrastsConfig)

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
