"""Pipeline planning utilities for CellQuorum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cellquorum.backends.registry import BackendRegistry, build_default_backend_registry
from cellquorum.config.models import CellQuorumConfig

StagePlanStatus = Literal["enabled", "disabled"]


@dataclass(frozen=True)
class PlannedStage:
    """
    Store the planned status of one pipeline stage.

    The planner does not decide all method-level details yet. Its first job is
    to convert validated configuration into a clear stage-level execution plan
    that can be shown to the user before any heavy analysis begins.

    Args:
        name: Stable stage name.
        enabled: Whether the stage is allowed by configuration.
        status: Human-readable status category.
        reason: Explanation of why the stage is enabled or disabled.
    """

    # Store the stable stage name.
    name: str

    # Store whether the stage is allowed by configuration.
    enabled: bool

    # Store the stage planning status.
    status: StagePlanStatus

    # Store the planning explanation.
    reason: str


@dataclass
class PipelinePlan:
    """
    Store a complete CellQuorum pipeline plan.

    The plan summarizes selected stages and backend availability before pipeline
    execution. This gives users a clear audit trail of what CellQuorum intends
    to do and helps prevent hidden behavior in advanced workflows.

    Args:
        profile: Selected run profile.
        stages: Planned stages.
        backend_status_table: JSON-serializable backend status rows.
        warnings: Planner-level warnings.
    """

    # Store the selected run profile.
    profile: str

    # Store planned stages.
    stages: list[PlannedStage] = field(default_factory=list)

    # Store backend status rows.
    backend_status_table: list[dict[str, object]] = field(default_factory=list)

    # Store planner-level warnings.
    warnings: list[str] = field(default_factory=list)

    def enabled_stage_names(self) -> list[str]:
        """
        Return names of stages enabled by the plan.

        Returns:
            Ordered list of enabled stage names.
        """

        # Return enabled stage names in plan order.
        return [stage.name for stage in self.stages if stage.enabled]

    def disabled_stage_names(self) -> list[str]:
        """
        Return names of stages disabled by the plan.

        Returns:
            Ordered list of disabled stage names.
        """

        # Return disabled stage names in plan order.
        return [stage.name for stage in self.stages if not stage.enabled]

    def to_dict(self) -> dict[str, object]:
        """
        Convert the pipeline plan to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the plan.
        """

        # Convert planned stages to dictionaries.
        stage_rows = [
            {
                "name": stage.name,
                "enabled": stage.enabled,
                "status": stage.status,
                "reason": stage.reason,
            }
            for stage in self.stages
        ]

        # Return the full plan dictionary.
        return {
            "profile": self.profile,
            "stages": stage_rows,
            "backend_status_table": self.backend_status_table,
            "warnings": list(self.warnings),
        }


class PipelinePlanner:
    """
    Build a stage-level execution plan from validated configuration.

    The planner is intentionally separate from the pipeline runner. It lets users
    inspect what CellQuorum intends to do before any expensive scRNA-seq analysis
    runs. Later, this class will also apply method gates based on manifest
    metadata, available layers, sample support, and backend availability.

    Args:
        config: Validated CellQuorum configuration.
        backend_registry: Backend registry used for availability reporting.
    """

    def __init__(
        self,
        config: CellQuorumConfig,
        backend_registry: BackendRegistry | None = None,
    ) -> None:
        """
        Initialize the pipeline planner.

        Args:
            config: Validated CellQuorum configuration.
            backend_registry: Optional backend registry. If omitted, the default
                CellQuorum backend registry is created.
        """

        # Store the validated configuration.
        self.config = config

        # Store the backend registry, creating the default one when omitted.
        self.backend_registry = backend_registry or build_default_backend_registry()

    def build_plan(self) -> PipelinePlan:
        """
        Build a pipeline plan.

        Returns:
            PipelinePlan describing configured stages and backend status.
        """

        # Build planned stages from stage configuration.
        stages = self._build_stage_plan()

        # Build backend status rows for planner/report output.
        backend_status_table = self.backend_registry.to_status_table()

        # Build planner warnings.
        warnings = self._build_warnings()

        # Return the complete pipeline plan.
        return PipelinePlan(
            profile=self.config.run.profile,
            stages=stages,
            backend_status_table=backend_status_table,
            warnings=warnings,
        )

    def _build_stage_plan(self) -> list[PlannedStage]:
        """
        Build the ordered stage plan from configuration flags.

        Returns:
            Ordered list of planned stages.
        """

        # Define stage names and their enabled flags in canonical run order.
        stage_flags = [
            ("ambient_correction", self.config.stages.ambient_correction),
            ("qc", self.config.stages.qc),
            ("preprocessing", self.config.stages.preprocessing),
            ("feature_selection", self.config.stages.feature_selection),
            ("dimensionality", self.config.stages.dimensionality),
            ("integration", self.config.stages.integration),
            # integration_gate sits here (after integration, before clustering) to
            # rank embeddings BEFORE committing expensive clustering+annotation.
            ("integration_gate", self.config.stages.integration_gate),
            ("clustering", self.config.stages.clustering),
            ("annotation", self.config.stages.annotation),
            ("subclustering", self.config.stages.subclustering),
            ("adjudication", self.config.stages.adjudication),
            ("reference_mapping", self.config.stages.reference_mapping),
            ("annotation_consensus", self.config.stages.annotation_consensus),
            # Annotation diagnostics must run after reference mapping so
            # transferred labels such as ref_state can be audited.
            ("annotation_diagnostics", self.config.stages.annotation_diagnostics),
            # Population identity is evidence-driven: it uses reference labels
            # when present, otherwise annotation labels or native clusters.
            ("population_identity", self.config.stages.population_identity),
            ("integration_benchmark", self.config.stages.integration_benchmark),
            ("state_scoring", self.config.stages.state_scoring),
            ("discovery", self.config.stages.discovery),
            ("composition", self.config.stages.composition),
            ("embeddings", self.config.stages.embeddings),
            ("differential_expression", self.config.stages.differential_expression),
            ("differential_abundance", self.config.stages.differential_abundance),
            ("enrichment", self.config.stages.enrichment),
            ("enrichment_viz", self.config.stages.enrichment_viz),
            ("molecular_inference", self.config.stages.molecular_inference),
            # Trajectory/potency runs with the tail-end discovery tracks: it only
            # needs embeddings (integration rep + 2D coords) to have already run,
            # which happens mid-backbone, so it slots in after molecular_inference
            # and before the CCC chain.
            ("trajectory", self.config.stages.trajectory),
            # CCC chain runs producer-before-consumer: the communication stage
            # writes the LR tables, ccc_network derives topology+curvature from
            # them, and ccc_viz renders figures from both. ccc_viz MUST come
            # last or it finds no inputs and every method MethodSkips. The
            # topology stage is registered as "ccc_network"; the enabling toggle
            # is still stages.network_analysis.
            ("cell_cell_communication", self.config.stages.cell_cell_communication),
            ("ccc_network", self.config.stages.network_analysis),
            ("ccc_viz", self.config.stages.ccc_viz),
        ]

        # Initialize the planned stage list.
        planned_stages: list[PlannedStage] = []

        # Convert each stage flag into a PlannedStage object.
        for stage_name, enabled in stage_flags:
            # Set the stage status from the enabled flag.
            status: StagePlanStatus = "enabled" if enabled else "disabled"

            # Create the explanatory reason.
            reason = (
                "Enabled by configuration. Method gates will decide whether this stage "
                "actually runs on the dataset."
                if enabled
                else "Disabled by configuration."
            )

            # Add the planned stage.
            planned_stages.append(
                PlannedStage(
                    name=stage_name,
                    enabled=enabled,
                    status=status,
                    reason=reason,
                )
            )

        # Return the ordered stage plan.
        return planned_stages

    def _build_warnings(self) -> list[str]:
        """
        Build planner-level warnings from configuration and backend availability.

        Returns:
            List of warning messages.
        """

        # Initialize planner warnings.
        warnings: list[str] = []

        # Warn if GPU is preferred but neither GPU nor RAPIDS is available.
        if self.config.compute.prefer_gpu:
            # Check generic GPU and RAPIDS backend availability.
            gpu_available = self.backend_registry.available("gpu")
            rapids_available = self.backend_registry.available("rapids")

            # Add a warning when GPU was preferred but no GPU backend is currently available.
            if not gpu_available and not rapids_available:
                warnings.append(
                    "GPU acceleration is preferred, but no GPU/RAPIDS backend is currently "
                    "available. CPU fallback will be used if allowed."
                )

        # Warn if R is enabled but neither rpy2 nor Rscript is available.
        if self.config.r.enabled:
            # Check rpy2 and Rscript backend availability.
            r_available = self.backend_registry.available("r")
            rscript_available = self.backend_registry.available("rscript")

            # Add a warning when R was enabled but no R backend is currently available.
            if not r_available and not rscript_available:
                warnings.append(
                    "R-backed methods are enabled, but neither rpy2 nor Rscript is currently "
                    "available. R-backed stages will be skipped or require installation."
                )

        # Return planner warnings.
        return warnings


def build_pipeline_plan(
    config: CellQuorumConfig,
    backend_registry: BackendRegistry | None = None,
) -> PipelinePlan:
    """
    Build a CellQuorum pipeline plan from validated configuration.

    Args:
        config: Validated CellQuorum configuration.
        backend_registry: Optional backend registry for tests or custom execution.

    Returns:
        PipelinePlan describing configured stages and backend status.
    """

    # Create a planner and build the plan.
    return PipelinePlanner(config=config, backend_registry=backend_registry).build_plan()
