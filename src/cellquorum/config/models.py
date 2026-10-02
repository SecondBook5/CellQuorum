"""Validated configuration models for CellQuorum."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from cellquorum.config.base import StrictBaseModel
from cellquorum.config.cohort import CohortConfig
from cellquorum.config.design import ContrastsConfig, DesignConfig
from cellquorum.config.markers import MarkersConfig
from cellquorum.stages.ambient_correction.config import AmbientCorrectionConfig
from cellquorum.stages.annotation.adjudication.config import AdjudicationConfig
from cellquorum.stages.annotation.config import AnnotationConfig
from cellquorum.stages.annotation.consensus.config import AnnotationConsensusConfig
from cellquorum.stages.annotation.diagnostics.config import AnnotationDiagnosticsConfig
from cellquorum.stages.annotation.population_identity.config import PopulationIdentityConfig
from cellquorum.stages.annotation.reference_mapping.config import ReferenceMappingConfig
from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig
from cellquorum.stages.cell_cell_communication.network.config import CCCNetworkConfig
from cellquorum.stages.cell_cell_communication.viz.config import CccVizConfig
from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig
from cellquorum.stages.comparative.differential_abundance.config import DifferentialAbundanceConfig
from cellquorum.stages.comparative.differential_expression.config import (
    DifferentialExpressionConfig,
)
from cellquorum.stages.comparative.differential_expression.viz.config import DeVizConfig
from cellquorum.stages.comparative.differential_state.config import DifferentialStateConfig
from cellquorum.stages.comparative.enrichment.config import EnrichmentConfig
from cellquorum.stages.comparative.enrichment.viz.config import EnrichmentVizConfig
from cellquorum.stages.comparative.module_remodeling.config import ModuleRemodelingConfig
from cellquorum.stages.comparative.multicellular_programs.config import MulticellularProgramsConfig
from cellquorum.stages.composition.config import CompositionConfig
from cellquorum.stages.discovery.config import DiscoveryConfig
from cellquorum.stages.gene_regulation.coexpression.config import CoexpressionConfig
from cellquorum.stages.gene_regulation.grn.config import GrnConfig
from cellquorum.stages.gene_regulation.perturbation.config import PerturbationConfig
from cellquorum.stages.integration.benchmark.config import IntegrationBenchmarkConfig
from cellquorum.stages.integration.config import IntegrationConfig
from cellquorum.stages.integration.embeddings.config import EmbeddingsConfig
from cellquorum.stages.metacells.config import MetacellsConfig
from cellquorum.stages.preprocessing.config import PreprocessingConfig
from cellquorum.stages.preprocessing.feature_selection.config import FeatureSelectionConfig
from cellquorum.stages.qc.config import (
    QCConfig,
    QCFinalizationConfig,
    QCSpliceMetricsConfig,
    QueryProjectionConfig,
)
from cellquorum.stages.state_scoring.config import StateScoringConfig
from cellquorum.stages.state_scoring.viz.config import StateScoringVizConfig
from cellquorum.stages.trajectory.config import TrajectoryConfig
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

    name: str = "cellquorum_project"

    description: str | None = None

    organism: str = "human"

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

    data_root: Path | None = None

    run_root: Path | None = None

    scratch_root: Path | None = None

    manifest: Path | None = None

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
        preserve_population_fits: Skip the default stripping of PCA/Harmony/
            neighbors/etc. after this restriction is applied. Only correct
            when ``input.h5ad`` already IS the target population -- its own
            joint fits were computed on exactly these cells -- rather than a
            slice of a larger shared atlas. A downstream-only rerun manifest
            that points straight at a previously exported, already-scoped
            per-lineage object (but still declares this block for provenance
            continuity with the original slicing run) sets this to True.
            Default False keeps the safe behavior: a row restriction normally
            changes the population every joint fit was computed over, so
            those fits are stripped rather than silently kept.
    """

    column: str

    values: list[str] = Field(min_length=1)

    require_agreement: str | None = None

    preserve_population_fits: bool = False

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

    column: str

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

    h5ad: Path | None = None

    counts_layer: str | None = None

    subset: InputSubsetConfig | None = None

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

    profile: Literal[
        "standard",
        "publication",
        "regulatory",
        "communication",
        "trajectory",
        "perturbation",
        "full",
    ] = "standard"

    run_id: str | None = None

    random_seed: int = 1337

    overwrite: bool = False

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

    verbose: bool = True

    log_level: Literal["quiet", "normal", "verbose"] = "normal"

    # Store whether the final in-memory AnnData is written to disk at the end of
    # a run. Without this, a from-scratch run threads the object through stages
    # in memory and discards it — leaving no annotated deliverable on disk.
    write_final_object: bool = True

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

    backend: Literal["auto", "cpu", "gpu", "rapids"] = "auto"

    prefer_gpu: bool = True

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
        max_memory_gb: RLIMIT_AS ceiling for Rscript subprocesses (honored;
            threaded to RscriptBackend). The 12GB default is sized for the
            common case (every script measured needs at most 889MB); a
            whole-atlas-scale call (e.g. scDblFinder across every capture in
            one R session) can legitimately need more, and previously had no
            way to raise it -- it was silently killed and reported identically
            to the R package being unavailable.
    """

    enabled: bool = True

    preferred_backend: Literal["auto", "r", "rscript"] = "auto"

    fallback_to_rscript: bool = True

    rscript_path: str = "Rscript"

    timeout_seconds: int = Field(default=30, ge=1)

    # Store the RLIMIT_AS ceiling for Rscript subprocesses, in GB. None disables it.
    # Raised from 12.0 after a real failure: multinichenet's pseudobulk DE +
    # ligand-receptor scoring across 6+ cell types on a 61,608-cell cohort
    # silently failed under the 12GB cap (confirmed: the identical script and
    # data succeeded with no cap, twice, outside the pipeline).
    max_memory_gb: float | None = 32.0


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

    enabled: bool = True

    html: bool = True

    markdown: bool = True

    pdf: bool = False

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
        use_highly_variable: Whether to restrict PCA to highly-variable genes. None
            follows the feature_selection stage; True requires its output.
        random_state: Seed for deterministic PCA.
        precision: Arithmetic dtype for standard PCA. "input" preserves the layer's
            dtype; float64 increases working memory but can improve numerical stability.
    """

    enabled: bool = True

    method: str = "pca"

    # Store the layer to run PCA on; must be a log-normalized layer.
    # Defaults to the preprocessing normalized output.
    input_layer: str = "cellquorum_normalized"

    # Store the component count, or "auto" for knee-based selection. Constrained to the
    # literal "auto" rather than any string: an unrecognized string (e.g. a typo) used to
    # be accepted here and only failed later as int("atuo") deep inside PCA, with no
    # indication the cause was a misspelled config value.
    n_pcs: int | Literal["auto"] = "auto"

    # Store the upper bound on components for auto selection. A non-positive value used to
    # be accepted and silently forced n_comps to 0 or negative regardless of how many fit
    # cells or HVG genes were available, which then surfaced as a component-budget error
    # blaming the wrong cause entirely.
    max_pcs: int = Field(default=50, ge=1)

    # Store whether PCA is restricted to highly-variable genes. None (the default) means
    # "follow feature_selection": use the HVGs when that stage flagged them, all genes
    # when it did not. Enabling HVG selection and then having to repeat the decision here
    # is what let `stages.feature_selection: true` produce a PCA on all ~33,000 genes.
    # `true` demands the HVGs and fails if they are absent; `false` forces all genes.
    use_highly_variable: bool | None = None

    random_state: int = 0

    precision: Literal["input", "float32", "float64"] = "input"

    @field_validator("n_pcs")
    @classmethod
    def _n_pcs_must_be_positive(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 1:
            raise ValueError("n_pcs must be a positive integer, or the literal 'auto'.")
        return value


class ResolutionDiagnosticConfig(StrictBaseModel):
    """
    Bootstrap cluster-stability diagnostic across a resolution sweep.

    Opt-in and expensive by design: an N-resolution x M-bootstrap sweep reruns
    neighbors+Leiden roughly N*(M+1) times, which is a "run once to choose a
    resolution" cost, not a per-pipeline-run one -- so it defaults off rather than
    silently slowing down every run for users who already know their resolution or
    are fine with the default.

    Method: bootstrap-subsample stability (Patterson-Cross, Levine & Bhaduri 2021,
    "chooseR"; see also scclusteval). At each resolution, a reference partition is
    fitted on every core cell; repeated subsamples are independently reclustered, and
    each reference cluster is scored by the Jaccard overlap to its best-matching
    cluster in the subsample. High, stable Jaccard across bootstraps means the
    partition at that resolution is robust to resampling noise -- the closest
    available proxy to ground truth when no labeled reference exists.
    """

    # Whether the diagnostic sweep runs at all.
    enabled: bool = False

    # Resolutions to sweep. Mirrors the same neighbors graph (use_rep, n_neighbors) the
    # real clustering run uses; only resolution varies.
    resolutions: list[float] = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

    # Independent subsamples per resolution. More bootstraps narrow the stability
    # estimate's own noise, at proportional cost.
    n_bootstraps: int = Field(default=20, ge=1)

    # Two-round allocation (round 1: a small share at every resolution; round 2: the
    # rest weighted toward uncertain/transitioning resolutions) vs. spending
    # n_bootstraps evenly everywhere. See module docstring for why this differs from
    # classic successive-halving.
    adaptive: bool = True

    # Fraction of fit-eligible cells kept per subsample.
    subsample_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)

    # Seed for subsample draws; passed through unchanged to each bootstrap's own
    # neighbors/Leiden call, so bootstrap-to-bootstrap variation reflects the resampled
    # cells, not solver randomness.
    random_state: int = 0

    # Whether to write the two-panel stability figure.
    write_figures: bool = True


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
        resolution_diagnostic: Opt-in bootstrap cluster-stability sweep across
            resolutions; see ResolutionDiagnosticConfig.
    """

    enabled: bool = True

    method: str = "leiden"

    n_neighbors: int = 15

    resolution: float = 1.0

    random_state: int = 0

    key_added: str = "leiden"

    # Embedding to cluster on (set to the integration output, e.g.
    # "X_pca_harmony", when integration runs; defaults to raw PCA).
    use_rep: str = "X_pca"

    # Opt-in bootstrap-stability sweep across resolutions; see ResolutionDiagnosticConfig.
    resolution_diagnostic: ResolutionDiagnosticConfig = ResolutionDiagnosticConfig()


class StageSelectionConfig(StrictBaseModel):
    """
    Store high-level stage enablement flags.

    These flags do not mean that every stage blindly runs on every dataset.
    They mean each stage is allowed to run if its data requirements, method
    assumptions, sample support, and backend requirements are satisfied.

    Args:
        ambient_correction: Whether ambient correction is enabled.
        qc_splice_metrics: Whether the optional splice-QC extraction is enabled.
        qc_evidence: Whether quality control is enabled.
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
        metacells: Whether the metacell (MetaCell2) diagnostic stage is enabled.
        composition: Whether composition analysis is enabled.
        differential_expression: Whether differential expression is enabled.
        differential_state: Whether donor-paired testing of per-cell state/program
            scores between conditions is enabled.
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

    # Store whether the optional splice-QC extraction is enabled (order 15). Opt-in: it
    # only does anything when a manifest names a per-sample velocyto loom, and it never
    # generates one, so enabling it on a run with no looms is a no-op skip.
    qc_splice_metrics: bool = False

    qc_evidence: bool = True

    preprocessing: bool = True

    # Store whether the feature-selection (HVG) stage may run. Opt-in: flagging HVGs
    # changes which genes PCA and scVI see, so it is a stated methodological choice.
    feature_selection: bool = False

    dimensionality: bool = True

    clustering: bool = True

    integration: bool = True

    annotation: bool = True

    # Store whether annotation-diagnostics evaluation is enabled. Opt-in: needs R and
    # the scDiagnostics package.
    annotation_diagnostics: bool = False

    annotation_consensus: bool = True

    # Store whether reference mapping is enabled. Opt-in: requires an atlas h5ad, which
    # has no default.
    reference_mapping: bool = False

    integration_benchmark: bool = False

    integration_gate: bool = False

    population_identity: bool = True

    state_scoring: bool = True

    state_scoring_viz: bool = True

    discovery: bool = True

    subclustering: bool = False

    # Store whether query projection is enabled: place borderline cells on the frozen
    # core manifold (order 105). Opt-in — it is only meaningful once graded QC has
    # produced borderline cells, and it feeds qc_finalization.
    query_projection: bool = False

    # Store whether QC finalization is enabled: per-cell rescue → qc_state_final
    # (order 135). Opt-in, and paired with query_projection: without the projection it
    # can only mark every borderline cell unresolved.
    qc_finalization: bool = False

    adjudication: bool = True

    # Store whether the metacell (MetaCell2) diagnostic stage is enabled. Opt-in:
    # a QC/identity diagnostic, not a required pipeline step.
    metacells: bool = False

    composition: bool = True

    differential_expression: bool = True

    differential_state: bool = True

    coexpression: bool = True

    grn: bool = True

    perturbation: bool = True

    differential_abundance: bool = True

    enrichment: bool = True

    enrichment_viz: bool = True

    de_viz: bool = True

    ccc_viz: bool = True

    embeddings: bool = True

    trajectory: bool = True

    trajectory_viz: bool = True

    molecular_inference: bool = True

    cell_cell_communication: bool = True

    network_analysis: bool = True

    multicellular_programs: bool = True

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
        qc_evidence: Quality-control settings.
        preprocessing: Preprocessing settings.
        dimensionality: Dimensionality-reduction settings.
        clustering: Clustering settings.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)

    paths: PathConfig = Field(default_factory=PathConfig)

    input: InputConfig = Field(default_factory=InputConfig)

    run: RunConfig = Field(default_factory=RunConfig)

    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    r: RConfig = Field(default_factory=RConfig)

    report: ReportConfig = Field(default_factory=ReportConfig)

    stages: StageSelectionConfig = Field(default_factory=StageSelectionConfig)

    ambient_correction: AmbientCorrectionConfig = Field(default_factory=AmbientCorrectionConfig)

    qc_evidence: QCConfig = Field(default_factory=QCConfig)

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)

    dimensionality: DimensionalityConfig = Field(default_factory=DimensionalityConfig)

    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)

    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)

    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)

    annotation_diagnostics: AnnotationDiagnosticsConfig = Field(
        default_factory=AnnotationDiagnosticsConfig
    )

    annotation_consensus: AnnotationConsensusConfig = Field(
        default_factory=AnnotationConsensusConfig
    )

    integration_benchmark: IntegrationBenchmarkConfig = Field(
        default_factory=IntegrationBenchmarkConfig
    )

    reference_mapping: ReferenceMappingConfig = Field(default_factory=ReferenceMappingConfig)

    population_identity: PopulationIdentityConfig = Field(default_factory=PopulationIdentityConfig)

    subclustering: SubclusteringConfig = Field(default_factory=SubclusteringConfig)

    qc_splice_metrics: QCSpliceMetricsConfig = Field(default_factory=QCSpliceMetricsConfig)

    query_projection: QueryProjectionConfig = Field(default_factory=QueryProjectionConfig)

    qc_finalization: QCFinalizationConfig = Field(default_factory=QCFinalizationConfig)

    adjudication: AdjudicationConfig = Field(default_factory=AdjudicationConfig)

    metacells: MetacellsConfig = Field(default_factory=MetacellsConfig)

    differential_abundance: DifferentialAbundanceConfig = Field(
        default_factory=DifferentialAbundanceConfig
    )

    differential_expression: DifferentialExpressionConfig = Field(
        default_factory=DifferentialExpressionConfig
    )

    differential_state: DifferentialStateConfig = Field(default_factory=DifferentialStateConfig)

    coexpression: CoexpressionConfig = Field(default_factory=CoexpressionConfig)

    grn: GrnConfig = Field(default_factory=GrnConfig)

    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)

    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)

    enrichment_viz: EnrichmentVizConfig = Field(default_factory=EnrichmentVizConfig)

    de_viz: DeVizConfig = Field(default_factory=DeVizConfig)

    ccc_viz: CccVizConfig = Field(default_factory=CccVizConfig)

    state_scoring: StateScoringConfig = Field(default_factory=StateScoringConfig)

    state_scoring_viz: StateScoringVizConfig = Field(default_factory=StateScoringVizConfig)

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)

    composition: CompositionConfig = Field(default_factory=CompositionConfig)

    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)

    trajectory_viz: TrajectoryVizConfig = Field(default_factory=TrajectoryVizConfig)

    cell_cell_communication: CellCellCommunicationConfig = Field(
        default_factory=CellCellCommunicationConfig
    )

    ccc_network: CCCNetworkConfig = Field(default_factory=CCCNetworkConfig)

    multicellular_programs: MulticellularProgramsConfig = Field(
        default_factory=MulticellularProgramsConfig
    )

    module_remodeling: ModuleRemodelingConfig = Field(default_factory=ModuleRemodelingConfig)

    markers: MarkersConfig = Field(default_factory=MarkersConfig)

    cohort: CohortConfig = Field(default_factory=CohortConfig)

    design: DesignConfig = Field(default_factory=DesignConfig)

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

        return self

    @model_validator(mode="after")
    def reconcile_stage_switches(self) -> CellQuorumConfig:
        """
        Make ``stages:`` the single switch deciding whether a stage runs.

        Whether a stage runs was spelled in TWO places: the ``stages:`` block, which the
        planner reads, and an ``enabled`` field on the stage's own config block, which the
        run gate at ``stage_base`` reads. Five of them defaulted to opposite values --
        ``feature_selection``, ``annotation_diagnostics``, ``reference_mapping``,
        ``integration_benchmark`` and ``subclustering`` -- so ``stages.feature_selection:
        true`` put the stage in the plan, printed it in the banner, reached it at position
        4 of 36, and then skipped it with "disabled by config" because the block's own
        default said no. Nothing was misconfigured and nothing warned; the HVGs simply were
        never flagged, and PCA and scVI silently ran on all ~33,000 genes.

        So ``stages:`` decides:

        * set in ``stages:`` -- that wins, and the block's ``enabled`` is aligned to it
        * set only on the block -- mirrored up, so the plan matches what will run
        * set in neither -- the ``stages:`` default applies

        Only one direction of disagreement is refused rather than resolved. ``stages.X:
        false`` with ``X.enabled: true`` is switching a stage off and leaving its settings
        behind, which is both unambiguous and what the old two-flag AND already did, so it
        stays silent. ``stages.X: true`` with an explicit ``X.enabled: false`` is the
        dangerous one: it reads as "run this stage", and the thing that actually happens is
        the silent skip described above. That halts at load.

        Component switches nested inside a stage (``qc_evidence.doublets.enabled``,
        ``qc_evidence.graded.enabled``) are untouched: those decide whether a *part* of a
        stage runs, which is a different question with only one place to ask it.

        Returns:
            Validated CellQuorumConfig, with both switches in agreement.

        Raises:
            ValueError: If a stage is requested in ``stages:`` and explicitly disabled in
                its own block.
        """

        for name in type(self.stages).model_fields:
            block = getattr(self, name, None)
            if block is None or not hasattr(block, "enabled"):
                continue

            flag = bool(getattr(self.stages, name))
            flag_given = name in self.stages.model_fields_set
            block_given = "enabled" in block.model_fields_set

            if flag_given and flag and block_given and not block.enabled:
                raise ValueError(
                    f"stages.{name}=True contradicts {name}.enabled=False. Whether a stage "
                    f"runs is declared once, in `stages:`, and this pair is the combination "
                    f"that plans the stage and then skips it. Remove `enabled` from the "
                    f"`{name}:` block, or set `stages.{name}: false`."
                )

            if block_given and not flag_given:
                # Mirror upward so the plan, the banner and the gate agree.
                setattr(self.stages, name, bool(block.enabled))
            else:
                block.enabled = flag

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
