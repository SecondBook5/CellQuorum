"""Pipeline stage executor for CellQuorum."""

from __future__ import annotations

# Import Mapping for immutable-style stage registry inputs.
from collections.abc import Mapping

# Import dataclass and field for structured executor result objects.
from dataclasses import dataclass, field

# Import UTC datetime for stage lifecycle timing.
from datetime import UTC, datetime

# Import adjudication stage.
from cellquorum.adjudication.stage import AdjudicationStage

# Import ambient correction stage.
from cellquorum.ambient_correction.stage import AmbientCorrectionStage

# Import annotation stage.
from cellquorum.annotation.stage import AnnotationStage

# Import annotation-consensus stage.
from cellquorum.annotation_consensus.stage import AnnotationConsensusStage

# Import annotation-diagnostics evaluation stage.
from cellquorum.annotation_diagnostics.stage import AnnotationDiagnosticsStage

# Import ccc-network (topology + curvature) stage.
from cellquorum.cell_cell_communication.network.stage import CCCNetworkStage

# Import cell-cell-communication stage.
from cellquorum.cell_cell_communication.stage import CellCellCommunicationStage

# Import CCC-visualization stage.
from cellquorum.cell_cell_communication.viz.stage import CccVizStage

# Import Phase-2A stages: dimensionality reduction and clustering.
from cellquorum.clustering.stage import ClusteringStage
from cellquorum.coexpression.stage import CoexpressionStage

# Import pipeline context.
from cellquorum.core.context import PipelineContext

# Import deterministic stage input fingerprinting.
from cellquorum.core.fingerprint import compute_input_fingerprint

# Import pipeline planning objects.
from cellquorum.core.planner import PipelinePlan, PlannedStage

# Import runtime progress reporter.
from cellquorum.core.run_reporter import RunReporter

# Import stage execution contracts and lifecycle records.
from cellquorum.core.stage import (
    PipelineStage,
    StageExecutionRecord,
    StageResult,
)

# Import differential expression stage.
from cellquorum.differential_abundance.stage import DifferentialAbundanceStage
from cellquorum.differential_expression.stage import DifferentialExpressionStage

# Import differential-expression-visualization stage.
from cellquorum.differential_expression.viz.stage import DeVizStage

# Import enrichment stage.
from cellquorum.enrichment.stage import EnrichmentStage

# Import enrichment-visualization stage.
from cellquorum.enrichment.viz.stage import EnrichmentVizStage
from cellquorum.grn.stage import GrnStage

# Import integration-benchmark evaluation stage.
from cellquorum.integration.benchmark.stage import IntegrationBenchmarkStage

# Import embeddings stage.
from cellquorum.integration.embeddings.stage import EmbeddingsStage

# Import integration stage.
from cellquorum.integration.stage import IntegrationStage
from cellquorum.perturbation.stage import PerturbationStage

# Import population/state identity evidence stage.
from cellquorum.population_identity.stage import PopulationIdentityStage
from cellquorum.preprocessing.dimensionality.stage import DimensionalityStage

# Import feature selection stage.
from cellquorum.preprocessing.feature_selection.stage import FeatureSelectionStage

# Import the preprocessing stage.
from cellquorum.preprocessing.stage import PreprocessingStage

# Import the first fully implemented scientific stage.
from cellquorum.qc.stage import QCStage

# Import reference-mapping stage.
from cellquorum.reference_mapping.stage import ReferenceMappingStage

# Import subclustering stage.
from cellquorum.subclustering.stage import SubclusteringStage

# Import trajectory stage.
from cellquorum.trajectory.stage import TrajectoryStage

# Import trajectory-visualization stage.
from cellquorum.trajectory.viz.stage import TrajectoryVizStage


@dataclass(frozen=True)
class StageRegistry:
    """
    Store executable pipeline stages by stable stage name.

    The registry is the bridge between a planned stage name, such as ``qc``, and
    the concrete stage implementation that can run it. Stages that are planned
    but not registered are skipped explicitly by the executor.

    Args:
        stages: Mapping from stable stage name to executable stage object.
    """

    # Store the executable stage mapping.
    stages: Mapping[str, PipelineStage] = field(default_factory=dict)

    def get(self, stage_name: str) -> PipelineStage | None:
        """
        Return a registered stage by name.

        Args:
            stage_name: Stable stage name.

        Returns:
            Registered stage object, or None if no implementation is registered.
        """

        # Return the registered stage if present.
        return self.stages.get(stage_name)

    def registered_stage_names(self) -> list[str]:
        """
        Return registered stage names.

        Returns:
            Sorted registered stage names.
        """

        # Return registered names in deterministic order.
        return sorted(self.stages)

    def with_stage(self, stage: PipelineStage) -> StageRegistry:
        """
        Return a new registry with one additional stage.

        Args:
            stage: Executable stage object with a stable ``name`` attribute.

        Returns:
            New StageRegistry containing the added stage.

        Raises:
            TypeError: If the stage does not expose a usable name.
        """

        # Validate that the stage exposes a string name.
        if not isinstance(getattr(stage, "name", None), str):
            raise TypeError("Registered pipeline stages must expose a string 'name' attribute.")

        # Copy the existing mapping.
        updated_stages = dict(self.stages)

        # Add or replace the stage by stable name.
        updated_stages[stage.name] = stage

        # Return a new registry.
        return StageRegistry(stages=updated_stages)


@dataclass(frozen=True)
class PipelineExecutionResult:
    """
    Store the result of executing a pipeline plan.

    Args:
        context: Final pipeline context after executed stages have updated AnnData.
        stage_results: Successful stage results keyed by stage name.
        stage_execution_records: Lifecycle records for every planned stage decision.
    """

    # Store the final pipeline context.
    context: PipelineContext

    # Store successful stage results by stage name.
    stage_results: dict[str, StageResult] = field(default_factory=dict)

    # Store lifecycle records for every planned stage decision.
    stage_execution_records: list[StageExecutionRecord] = field(default_factory=list)

    def succeeded_stage_names(self) -> list[str]:
        """
        Return names of stages that executed successfully.

        Returns:
            Stage names with success records.
        """

        # Return success stage names in execution order.
        return [
            record.stage_name
            for record in self.stage_execution_records
            if record.status == "success"
        ]

    def skipped_stage_names(self) -> list[str]:
        """
        Return names of stages that were skipped.

        Returns:
            Stage names with skipped records.
        """

        # Return skipped stage names in execution order.
        return [
            record.stage_name
            for record in self.stage_execution_records
            if record.status == "skipped"
        ]

    def failed_stage_names(self) -> list[str]:
        """
        Return names of stages that failed.

        Returns:
            Stage names with failed records.
        """

        # Return failed stage names in execution order.
        return [
            record.stage_name
            for record in self.stage_execution_records
            if record.status == "failed"
        ]

    def has_failures(self) -> bool:
        """
        Return whether any stage failed.

        Returns:
            True if at least one failed record exists.
        """

        # Report whether failed stages exist.
        return bool(self.failed_stage_names())


def build_default_stage_registry() -> StageRegistry:
    """
    Build the default executable stage registry.

    Returns:
        StageRegistry containing all currently implemented executable stages.
    """

    # Register fully implemented scientific stages.
    return StageRegistry(
        stages={
            "ambient_correction": AmbientCorrectionStage(),
            "qc": QCStage(),
            "preprocessing": PreprocessingStage(),
            "feature_selection": FeatureSelectionStage(),
            "dimensionality": DimensionalityStage(),
            "clustering": ClusteringStage(),
            "integration": IntegrationStage(),
            "annotation": AnnotationStage(),
            "annotation_diagnostics": AnnotationDiagnosticsStage(),
            "annotation_consensus": AnnotationConsensusStage(),
            "adjudication": AdjudicationStage(),
            "integration_benchmark": IntegrationBenchmarkStage(),
            "population_identity": PopulationIdentityStage(),
            "reference_mapping": ReferenceMappingStage(),
            "subclustering": SubclusteringStage(),
            "differential_expression": DifferentialExpressionStage(),
            "coexpression": CoexpressionStage(),
            "grn": GrnStage(),
            "perturbation": PerturbationStage(),
            "differential_abundance": DifferentialAbundanceStage(),
            "enrichment": EnrichmentStage(),
            "enrichment_viz": EnrichmentVizStage(),
            "de_viz": DeVizStage(),
            "ccc_viz": CccVizStage(),
            "embeddings": EmbeddingsStage(),
            "trajectory": TrajectoryStage(),
            "trajectory_viz": TrajectoryVizStage(),
            "cell_cell_communication": CellCellCommunicationStage(),
            "ccc_network": CCCNetworkStage(),
        }
    )


@dataclass(frozen=True)
class PipelineExecutor:
    """
    Execute a planned CellQuorum pipeline.

    The executor is plan-aware and registry-driven. It runs implemented stages,
    records disabled planned stages as skips, records enabled-but-unimplemented
    stages as skips, updates ``context.adata`` after successful stages, and
    records structured failures.

    Args:
        registry: Executable stage registry.
        stop_on_failure: Whether execution should stop after the first failed stage.
        backend_used: Backend label to record for Python-native stages.
    """

    # Store executable stage implementations.
    registry: StageRegistry = field(default_factory=build_default_stage_registry)

    # Store whether execution should stop after a stage failure.
    stop_on_failure: bool = True

    # Store the backend label for Python-native stage execution.
    backend_used: str = "python"

    def run(
        self,
        *,
        context: PipelineContext,
        plan: PipelinePlan,
        reporter: RunReporter | None = None,
    ) -> PipelineExecutionResult:
        """
        Execute a pipeline plan against a pipeline context.

        Args:
            context: Initialized pipeline context.
            plan: Pipeline plan describing stage order and enablement.
            reporter: Optional RunReporter for progress output. When None,
                defaults to a no-op reporter (verbose=False).

        Returns:
            PipelineExecutionResult with final context, stage results, and records.

        Raises:
            TypeError: If context or plan has the wrong type.
        """

        # Validate executor inputs.
        validate_executor_inputs(context=context, plan=plan)

        # Default to a no-op reporter when none provided (backward compat).
        reporter = reporter or RunReporter(verbose=False)

        # Initialize mutable execution state.
        current_context = context
        stage_results: dict[str, StageResult] = {}
        stage_execution_records: list[StageExecutionRecord] = []

        # Execute or skip every planned stage in order, with progress tracking.
        with reporter.progress(total=len(plan.stages)) as bar:
            for i, planned_stage in enumerate(plan.stages):
                # Announce stage start.
                reporter.stage_start(planned_stage.name, i + 1, len(plan.stages))

                # Execute one planned stage decision.
                stage_result, stage_record = self.execute_planned_stage(
                    context=current_context,
                    planned_stage=planned_stage,
                )

                # Store the lifecycle record for this planned stage.
                stage_execution_records.append(stage_record)

                # Report stage completion.
                reporter.stage_end(stage_record)

                # Advance the progress bar.
                bar.advance()

                # Stop after failures when configured.
                if stage_record.status == "failed":
                    if self.stop_on_failure:
                        break

                    # Continue to the next planned stage when failure stopping is disabled.
                    continue

                # Keep context unchanged for skipped stages.
                if stage_record.status == "skipped":
                    continue

                # Successful records must have a StageResult.
                if stage_result is None:
                    raise RuntimeError(
                        "Executor recorded a successful stage without a StageResult. "
                        f"Stage: {planned_stage.name}"
                    )

                # Store the successful stage result.
                stage_results[planned_stage.name] = stage_result

                # Propagate the updated AnnData object to downstream stages.
                current_context = current_context.with_adata(stage_result.adata)

        # Return the complete execution result.
        return PipelineExecutionResult(
            context=current_context,
            stage_results=stage_results,
            stage_execution_records=stage_execution_records,
        )

    def execute_planned_stage(
        self,
        *,
        context: PipelineContext,
        planned_stage: PlannedStage,
    ) -> tuple[StageResult | None, StageExecutionRecord]:
        """
        Execute or skip one planned stage.

        Args:
            context: Current pipeline context.
            planned_stage: Planned stage row from PipelinePlan.

        Returns:
            Tuple of optional StageResult and StageExecutionRecord.
        """

        # Mark the start of the stage decision.
        started_at = datetime.now(UTC)

        # Skip stages disabled by the plan.
        if not planned_stage.enabled:
            ended_at = datetime.now(UTC)
            return None, StageExecutionRecord.skipped(
                stage_name=planned_stage.name,
                reason=planned_stage.reason,
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                backend_used=None,
                details={
                    "plan_status": planned_stage.status,
                    "implemented": self.registry.get(planned_stage.name) is not None,
                },
            )

        # Look up the executable implementation.
        stage = self.registry.get(planned_stage.name)

        # Skip enabled stages that are not implemented yet.
        if stage is None:
            ended_at = datetime.now(UTC)
            return None, StageExecutionRecord.skipped(
                stage_name=planned_stage.name,
                reason="Stage is planned but no implementation is registered.",
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                backend_used=None,
                details={
                    "plan_status": planned_stage.status,
                    "plan_reason": planned_stage.reason,
                    "implemented": False,
                },
                notes=[
                    "This stage is part of the planned CellQuorum workflow but has not "
                    "been implemented yet."
                ],
            )

        # Compute a deterministic input fingerprint from the stage config and
        # the input AnnData signature, so completed stages can be compared on
        # rerun. Fingerprinting must never break execution, so any failure here
        # degrades to "no fingerprint" rather than aborting the stage.
        input_fingerprint: str | None = None
        try:
            from cellquorum.methods.context_access import resolve_stage_config

            input_fingerprint = compute_input_fingerprint(
                stage_name=planned_stage.name,
                stage_config=resolve_stage_config(context, planned_stage.name),
                adata=getattr(context, "adata", None),
                random_seed=getattr(context, "random_seed", None),
            )
        except Exception:  # pragma: no cover - fingerprinting is best-effort
            input_fingerprint = None

        # Opt-in resume: skip a side-effect-only stage whose prior completion
        # marker matches the current input fingerprint and whose recorded
        # artifacts still exist. Resume must never break a run, so any failure
        # here degrades to normal execution.
        if _is_resume_enabled(context):
            try:
                from cellquorum.core.resume import decide_stage_resume

                decision = decide_stage_resume(
                    stage_name=planned_stage.name,
                    provenance_dir=context.paths.provenance,
                    input_fingerprint=input_fingerprint,
                )
            except Exception:  # pragma: no cover - resume is best-effort
                decision = None

            if decision is not None and decision.resume:
                ended_at = datetime.now(UTC)
                return None, StageExecutionRecord.skipped(
                    stage_name=planned_stage.name,
                    reason=decision.reason,
                    started_at_utc=started_at,
                    ended_at_utc=ended_at,
                    backend_used=None,
                    details={"resumed": True},
                    notes=[f"Resumed: {decision.reason}"],
                    input_fingerprint=input_fingerprint,
                )

        # Execute the registered stage.
        try:
            stage_result = stage.run(context)

        # Convert stage exceptions into structured failed records.
        except Exception as error:
            ended_at = datetime.now(UTC)
            return None, StageExecutionRecord.failed(
                stage_name=planned_stage.name,
                error=error,
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                backend_used=self.backend_used,
            )

        # Mark successful execution end time.
        ended_at = datetime.now(UTC)

        # Stamp the computed input fingerprint onto the result when the stage
        # did not set one itself. Stages may override with a richer fingerprint.
        if stage_result.input_fingerprint is None and input_fingerprint is not None:
            stage_result.input_fingerprint = input_fingerprint

        # Detect skipped stages from the explicit StageResult status. Older
        # stages that still emit metrics["skipped"] are normalized by
        # StageResult.__post_init__.
        if stage_result.status == "skipped":
            skip_reason = stage_result.skip_reason or "skipped by method/config"
            stage_record = StageExecutionRecord.skipped(
                stage_name=planned_stage.name,
                reason=skip_reason,
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                backend_used=stage_result.backend or self.backend_used,
                warnings=stage_result.warnings if stage_result.warnings else None,
                notes=stage_result.notes if stage_result.notes else None,
                details=dict(stage_result.metrics),
                method_version=stage_result.method_version,
                device=stage_result.device,
                input_fingerprint=stage_result.input_fingerprint,
                output_fingerprint=stage_result.output_fingerprint,
                checkpoint_path=stage_result.checkpoint_path,
            )
            # Return (None, record) for skipped stages so executor.run skips
            # storing the result + propagating the unchanged adata.
            return None, stage_record

        # Build a successful stage execution record.
        stage_record = StageExecutionRecord.success(
            stage_name=planned_stage.name,
            result=stage_result,
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            backend_used=self.backend_used,
        )

        # Return the successful result and record.
        return stage_result, stage_record


def _is_resume_enabled(context: PipelineContext) -> bool:
    """Return whether opt-in stage resume is enabled for this run."""

    config = getattr(context, "config", None)
    run = getattr(config, "run", None)
    return bool(getattr(run, "resume", False))


def validate_executor_inputs(*, context: PipelineContext, plan: PipelinePlan) -> None:
    """
    Validate PipelineExecutor.run inputs.

    Args:
        context: Candidate pipeline context.
        plan: Candidate pipeline plan.

    Raises:
        TypeError: If either input has the wrong type.
    """

    # Validate the context type.
    if not isinstance(context, PipelineContext):
        raise TypeError(
            "PipelineExecutor.run expected context to be a PipelineContext. "
            f"Received: {type(context).__name__}"
        )

    # Validate the plan type.
    if not isinstance(plan, PipelinePlan):
        raise TypeError(
            "PipelineExecutor.run expected plan to be a PipelinePlan. "
            f"Received: {type(plan).__name__}"
        )


__all__ = [
    "PipelineExecutionResult",
    "PipelineExecutor",
    "StageRegistry",
    "build_default_stage_registry",
    "validate_executor_inputs",
]
