"""Validated configuration models for CellQuorum."""

from __future__ import annotations

# Import Counter for detecting repeated contrast names.
from collections import Counter

# Import Path for filesystem-like configuration fields.
from pathlib import Path

# Import Literal for constrained string configuration options.
from typing import Annotated, Literal

# Import Pydantic primitives for strict runtime validation.
from pydantic import Field, field_validator, model_validator

# Import the shared strict base model used by CellQuorum configuration models.
from cellquorum.config.base import StrictBaseModel

# Import the central cohort schema (structural keys declared once).
from cellquorum.config.cohort import CohortConfig

# Import the markers, design, and contrasts configuration models.
from cellquorum.config.design import ContrastsConfig, DesignConfig
from cellquorum.config.markers import MarkersConfig

# Import the ambient-correction configuration model.
from cellquorum.stages.ambient_correction.config import AmbientCorrectionConfig

# Import the adjudication configuration model.
from cellquorum.stages.annotation.adjudication.config import AdjudicationConfig

# Import the annotation configuration model.
from cellquorum.stages.annotation.config import AnnotationConfig

# Import the annotation-consensus configuration model.
from cellquorum.stages.annotation.consensus.config import AnnotationConsensusConfig

# Import the annotation-diagnostics configuration model.
from cellquorum.stages.annotation.diagnostics.config import AnnotationDiagnosticsConfig

# Import the population-identity configuration model.
from cellquorum.stages.annotation.population_identity.config import PopulationIdentityConfig

# Import the reference-mapping configuration model.
from cellquorum.stages.annotation.reference_mapping.config import ReferenceMappingConfig

# Import the cell-cell-communication configuration model.
from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig

# Import the ccc-network (topology + curvature) configuration model.
from cellquorum.stages.cell_cell_communication.network.config import CCCNetworkConfig

# Import the CCC-visualization configuration model.
from cellquorum.stages.cell_cell_communication.viz.config import CccVizConfig

# Import the subclustering configuration model.
from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig

# Import the differential-abundance configuration model.
from cellquorum.stages.comparative.differential_abundance.config import DifferentialAbundanceConfig

# Import the differential-expression configuration model.
from cellquorum.stages.comparative.differential_expression.config import (
    DifferentialExpressionConfig,
)

# Import the differential-expression-visualization configuration model.
from cellquorum.stages.comparative.differential_expression.viz.config import DeVizConfig

# Import the enrichment configuration model.
from cellquorum.stages.comparative.enrichment.config import EnrichmentConfig

# Import the enrichment-visualization configuration model.
from cellquorum.stages.comparative.enrichment.viz.config import EnrichmentVizConfig

# Import the module-remodeling configuration model.
from cellquorum.stages.comparative.module_remodeling.config import ModuleRemodelingConfig

# Import the multicellular-programs configuration model.
from cellquorum.stages.comparative.multicellular_programs.config import MulticellularProgramsConfig

# Import the discovery (consensus-NMF) configuration model.
from cellquorum.stages.discovery.config import DiscoveryConfig

# Import the coexpression configuration model.
from cellquorum.stages.gene_regulation.coexpression.config import CoexpressionConfig

# Import the GRN configuration model.
from cellquorum.stages.gene_regulation.grn.config import GrnConfig

# Import the perturbation configuration model.
from cellquorum.stages.gene_regulation.perturbation.config import PerturbationConfig

# Import the integration-benchmark configuration model.
from cellquorum.stages.integration.benchmark.config import IntegrationBenchmarkConfig

# Import the integration configuration model.
from cellquorum.stages.integration.config import IntegrationConfig

# Import the embeddings configuration model.
from cellquorum.stages.integration.embeddings.config import EmbeddingsConfig

# Import the preprocessing configuration model.
from cellquorum.stages.preprocessing.config import PreprocessingConfig

# Import the feature-selection configuration model.
from cellquorum.stages.preprocessing.feature_selection.config import FeatureSelectionConfig

# Import the QC configuration model.
from cellquorum.stages.qc.config import QCConfig

# Import the state-scoring configuration model.
from cellquorum.stages.state_scoring.config import StateScoringConfig

# Import the trajectory configuration model.
from cellquorum.stages.trajectory.config import TrajectoryConfig

# Import the trajectory-visualization configuration model.
from cellquorum.stages.trajectory.viz.config import TrajectoryVizConfig


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

    Optionally also require a SECOND annotation column to agree. Most annotated
    objects carry more than one label per cell -- a marker/clustering call and a
    reference-mapped call -- and the cells they disagree about are the ones most
    likely to be misassigned. Requiring concordance drops them, which matters most
    when two lineages are being compared: a slice built by agreement and one built
    by a single column are not filtered equally, so a difference between them can
    be filtering rather than biology.

    Args:
        column: obs column to filter on (e.g. ``cell_type``).
        values: keep rows whose ``column`` value is one of these.
        require_agreement: Optional second obs column that must carry the SAME
            value as ``column`` for a row to be kept (e.g. ``ref_state``). The
            loader refuses to apply it when a requested value is missing from that
            column's vocabulary, because "no cell agrees" and "this column has no
            word for this label" are different facts with the same cell count.
    """

    # Store the obs column to filter rows on.
    column: str

    # Store the accepted values; a row is kept when its column value is in here.
    values: list[str] = Field(min_length=1)

    # Store an optional second annotation column that must agree with `column`.
    require_agreement: str | None = None

    @field_validator("column", "require_agreement")
    @classmethod
    def validate_column(cls, value: str | None) -> str | None:
        """Reject an empty subset column name."""

        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("input.subset column names cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_agreement_is_a_second_opinion(self) -> InputSubsetConfig:
        """Reject a column asked to agree with itself, which filters nothing."""

        if self.require_agreement is not None and self.require_agreement == self.column:
            raise ValueError(
                "input.subset.require_agreement must name a DIFFERENT obs column "
                f"than input.subset.column (both are {self.column!r}). A column "
                "always agrees with itself, so this would silently filter nothing."
            )
        return self

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[str]) -> list[str]:
        """Strip values and reject empty entries."""

        cleaned = [str(item).strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("input.subset.values cannot contain empty strings.")
        return cleaned


class InputExcludeConfig(StrictBaseModel):
    """
    Drop the input's rows whose ``column`` value is in ``values``.

    The counterpart to :class:`InputSubsetConfig`, and not expressible with it: an
    inclusion list cannot say "everything except". Dropping one artifact cluster
    from a 39-cluster partition by inclusion means naming the other 38, which is
    unreadable and goes wrong in a specific way -- the list is silently incomplete
    the next time the object gains a cluster.

    The intended use is data artifacts rather than biology: an ambient/debris
    cluster identified by criteria on THIS object, most directly with
    :func:`cellquorum.stats.cluster_artifact_audit`. Because cluster ids belong to
    one clustering run and not to the cells, the loader refuses values that are not
    in the column's vocabulary instead of excluding zero rows -- a mask carried over
    from a different partition names nothing here, and silently removing nothing
    while the config claims a cluster was dropped is the failure this guard exists
    for. The applied counts land in run provenance, so what the exclusion cost is a
    recorded number rather than an assumption.

    Args:
        column: obs column to drop rows on (e.g. ``leiden``).
        values: values of ``column`` whose rows are dropped.
    """

    # Store the obs column to drop rows on.
    column: str

    # Store the values whose rows are dropped.
    values: list[str] = Field(min_length=1)

    @field_validator("column")
    @classmethod
    def validate_column(cls, value: str) -> str:
        """Reject an empty exclusion column name."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("input.exclude.column cannot be empty.")
        return cleaned

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[str]) -> list[str]:
        """Strip values and reject empty entries."""

        cleaned = [str(item).strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("input.exclude.values cannot contain empty strings.")
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
        exclude: Optional row exclusion applied at load time, composable with
            ``subset`` (a lineage slice that also drops an artifact cluster needs
            both).
    """

    # Store an optional AnnData h5ad input path.
    h5ad: Path | None = None

    # Store an optional AnnData layer containing raw counts.
    counts_layer: str | None = None

    # Store an optional row restriction (e.g. cell_type == Fibroblasts).
    subset: InputSubsetConfig | None = None

    # Store an optional row exclusion (e.g. leiden in the audited debris clusters).
    exclude: InputExcludeConfig | None = None

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

    # Write the AnnData after each stage so a later run can start mid-pipeline.
    # OFF by default: a full object per stage is hundreds of GB on a large atlas,
    # a cost production runs must not pay. Enable while developing or verifying a
    # pipeline stage-by-stage. Reuse is guarded by the same input fingerprint the
    # resume path uses, and a mismatch is refused rather than silently loaded.
    checkpoint: bool = False

    # Stages to checkpoint after. Empty/unset means EVERY stage, which is the
    # useful default once checkpointing is on at all — the reason to enable it is
    # to be able to stop anywhere. Narrow it to bound disk on large cohorts.
    checkpoint_after: list[str] = Field(default_factory=list)

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
        n_jobs: Number of CPU workers for steps that support parallel execution,
            or ``"auto"`` to derive one from the machine. Three steps read it, and
            all three are among the slowest things in a run: scDblFinder (one job
            per capture), scVelo's ``recover_dynamics`` / ``velocity_graph``, and
            pySCENIC's GRNBoost2/AUCell. The first two are bit-identical
            regardless of worker count, so for them this is a pure speed knob —
            see ``VelocityConfig.n_jobs`` for the measurements.

            ``"auto"`` is the default because the alternative defaults are both
            wrong: a fixed 1 leaves most of the machine idle on the longest steps,
            and a fixed 8 oversubscribes a 2-core laptop or a CPU-limited
            container 4x. An explicit int always wins over auto, and a stage may
            still pin its own worker count and override both; resolution goes
            through ``core.context.resolve_n_jobs``.
    """

    # Store the preferred compute backend.
    backend: Literal["auto", "cpu", "gpu", "rapids"] = "auto"

    # Store whether GPU acceleration should be preferred when available.
    prefer_gpu: bool = True

    # Store whether CPU fallback is allowed when GPU is unavailable.
    fallback_to_cpu: bool = True

    # Store the number of CPU workers, or "auto" to derive one from the machine.
    # 0 and negatives are still rejected: joblib and dask both read them as "all
    # cores", which is a different request from the one this field expresses.
    n_jobs: Annotated[int, Field(ge=1)] | Literal["auto"] = "auto"


class RConfig(StrictBaseModel):
    """
    Store R backend preferences.

    R support can run through in-process rpy2 or batch-friendly Rscript. This
    config only expresses preference; the backend registry determines what is
    actually available.

    NOTE: ``preferred_backend`` and ``fallback_to_rscript`` are **reserved and
    not yet honored** — every R-backed method currently dispatches through the
    Rscript backend regardless of these values (there is no in-process rpy2
    dispatch for the bundled R scripts). They are documented here so the schema
    is explicit rather than silently ignoring them; wiring rpy2 dual-dispatch is
    tracked separately. ``rscript_path`` and ``enabled``, by contrast, ARE
    honored: ``rscript_path`` is threaded to the Rscript backend and its
    availability check.

    Args:
        enabled: Whether R-backed methods are allowed.
        preferred_backend: Preferred R execution mode (reserved; not yet honored).
        fallback_to_rscript: Whether Rscript fallback is allowed (reserved; not
            yet honored — Rscript is always used).
        rscript_path: Rscript executable name or path (honored).
        timeout_seconds: Timeout for lightweight R checks.
    """

    # Store whether R-backed methods are allowed.
    enabled: bool = True

    # Store the preferred R backend. Reserved: not yet honored (see class NOTE).
    preferred_backend: Literal["auto", "r", "rscript"] = "auto"

    # Store whether Rscript fallback is allowed. Reserved: not yet honored.
    fallback_to_rscript: bool = True

    # Store the Rscript executable path. Honored: threaded to the Rscript backend.
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

    # Store whether ambient correction is enabled. On by default: ambient mRNA is
    # what puts keratin in a fibroblast and haemoglobin in everything, and every
    # later stage inherits that error, so the correct default is to correct it. The
    # stage skips itself with an explicit reason when its inputs are absent (no
    # manifest, no CellRanger raw/filtered h5, no Rscript), so enabling it cannot
    # break a run that has nothing to correct from.
    ambient_correction: bool = True

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

    # Store whether multicellular programs analysis is enabled as a gated capability.
    multicellular_programs: bool = True

    # Store whether module remodeling analysis is enabled as a gated capability.
    module_remodeling: bool = True


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

    # Store cell-state program scoring settings.
    state_scoring: StateScoringConfig = Field(default_factory=StateScoringConfig)

    # Store de-novo program discovery (consensus-NMF) settings.
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)

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

    # Store multicellular-programs settings.
    multicellular_programs: MulticellularProgramsConfig = Field(
        default_factory=MulticellularProgramsConfig
    )

    # Store module-remodeling settings.
    module_remodeling: ModuleRemodelingConfig = Field(default_factory=ModuleRemodelingConfig)

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

    @model_validator(mode="after")
    def validate_declared_contrasts_are_reachable(self) -> CellQuorumConfig:
        """
        Reject a declared contrast the engine will not actually run.

        The comparison that runs comes from ``design.case``/``design.control`` alone:
        the DE stage config declares no case/control/paired fields, so its stage
        wrapper fills them from ``design``, and ``contrasts`` is nowhere in that
        chain. Nothing in the engine reads ``contrasts`` -- multi-contrast DE is not
        wired -- yet the block is serialized into ``provenance/resolved_config.json``,
        where a named contrast with its own case/control reads as the authoritative
        statement of what was compared.

        A contrast that agrees with the design is therefore harmless documentation,
        and one that disagrees is a config that describes a comparison the run never
        performed. That is not a stylistic problem: the natural edit is to change the
        *named* thing ("LE_vs_Normal") and leave ``design`` alone, which would leave
        the run comparing the old levels while provenance advertised the new ones. So
        a divergence halts at load, before any compute.

        Returns:
            Validated CellQuorumConfig.

        Raises:
            ValueError: If a declared contrast diverges from the design, sets a
                ``paired`` value the engine cannot honour, or reuses a name.
        """

        declared = self.contrasts.contrasts
        if not declared:
            return self

        # Duplicate names silently collapse: ContrastsConfig.get returns the first
        # match, so a second contrast under a used name is unreachable by name too.
        counts = Counter(contrast.name for contrast in declared)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"contrasts declares duplicate name(s) {duplicates}. Names address a "
                "contrast, so a repeat makes all but the first unreachable."
            )

        # Without a design comparison, the contrast is the ONLY statement of what to
        # compare -- and it is the one thing not consulted. Fail rather than run nothing.
        if self.design.case is None or self.design.control is None:
            names = [c.name for c in declared]
            raise ValueError(
                f"contrasts declares {names} but design.case/design.control are unset. "
                "The comparison is taken from `design`, never from `contrasts`, so this "
                "config declares a comparison the run cannot perform. Set design.case "
                "and design.control to the levels you mean to compare."
            )

        design_pair = (self.design.case, self.design.control)
        for contrast in declared:
            if (contrast.case, contrast.control) != design_pair:
                raise ValueError(
                    f"contrast '{contrast.name}' compares "
                    f"{contrast.case!r} vs {contrast.control!r}, but the run compares "
                    f"{design_pair[0]!r} vs {design_pair[1]!r} (from `design`). "
                    "Multi-contrast DE is not wired: `contrasts` is recorded in "
                    "provenance but never read, so this contrast would be described "
                    "and not computed. Either make it match `design`, or change "
                    "`design` to the comparison you want."
                )
            # An explicit `paired` on a contrast reads as an override and is not one.
            if contrast.paired is not None and contrast.paired != self.design.paired:
                raise ValueError(
                    f"contrast '{contrast.name}' sets paired={contrast.paired} while "
                    f"design.paired={self.design.paired}. Pairing is resolved from "
                    "`design` (then auto-promoted per cell type when every donor "
                    "contributes both arms); a contrast's `paired` is never read, so "
                    "this would silently not take effect. Set design.paired instead."
                )

        return self
