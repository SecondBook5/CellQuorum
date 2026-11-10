"""Pipeline planning utilities for CellQuorum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cellquorum.backends.registry import (
    BackendRegistry,
    BackendStatusRow,
    build_default_backend_registry,
)
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.exceptions import CellQuorumConfigError

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

    name: str

    enabled: bool

    status: StagePlanStatus

    reason: str
    methods: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()


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

    profile: str

    stages: list[PlannedStage] = field(default_factory=list)

    backend_status_table: list[BackendStatusRow] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def enabled_stage_names(self) -> list[str]:
        """
        Return names of stages enabled by the plan.

        Returns:
            Ordered list of enabled stage names.
        """

        return [stage.name for stage in self.stages if stage.enabled]

    def disabled_stage_names(self) -> list[str]:
        """
        Return names of stages disabled by the plan.

        Returns:
            Ordered list of disabled stage names.
        """

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
                "methods": list(stage.methods),
                "expected_outputs": list(stage.expected_outputs),
                "required_inputs": list(stage.required_inputs),
            }
            for stage in self.stages
        ]

        return {
            "profile": self.profile,
            "stages": stage_rows,
            "backend_status_table": self.backend_status_table,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
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
        # Thread the configured Rscript path so R availability reflects where R
        # actually lives (e.g. a non-default container/HPC path).
        self.backend_registry = backend_registry or build_default_backend_registry(
            rscript_path=config.r.rscript_path,
            max_memory_gb=config.r.max_memory_gb,
            sctenifoldknk_python=config.perturbation.sctenifoldknk.python_executable,
        )

    def build_plan(self) -> PipelinePlan:
        """
        Build a pipeline plan.

        Returns:
            PipelinePlan describing configured stages and backend status.
        """

        # Build planned stages from stage configuration.
        stages = self._build_stage_plan()
        unavailable = self.config.run.required_outputs.keys() - {
            s.name for s in stages if s.enabled
        }
        if unavailable:
            raise CellQuorumConfigError(
                f"run.required_outputs names unknown or disabled stages: {sorted(unavailable)}"
            )

        # Build backend status rows for planner/report output.
        backend_status_table = self.backend_registry.to_status_table()

        warnings = self._build_warnings()

        blockers = []
        if self.config.run.require_selected_methods:
            from cellquorum.core.study_preflight import study_blockers

            blockers = study_blockers(
                self.config, {s.name: list(s.methods) for s in stages if s.enabled}
            )
        return PipelinePlan(
            blockers=blockers,
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

        from cellquorum.core.stages import all_stage_specs

        specs = all_stage_specs()
        reasons = self.config.run.disabled_stage_reasons
        unknown = reasons.keys() - {spec.name for spec in specs}
        if unknown:
            raise CellQuorumConfigError(
                f"Unknown stages in run.disabled_stage_reasons: {sorted(unknown)}"
            )

        planned_stages: list[PlannedStage] = []

        for spec in specs:
            stage_name = spec.name
            enabled = getattr(self.config.stages, spec.config_flag)
            status: StagePlanStatus = "enabled" if enabled else "disabled"

            reason = (
                "Enabled by configuration. Method gates will decide whether this stage "
                "actually runs on the dataset."
                if enabled
                else reasons.get(stage_name) or "Disabled by configuration."
            )

            if enabled and self.config.run.study_questions:
                families = {
                    "cell_cell_communication": "Communication across populations",
                    "ccc_viz": "Communication figures",
                    "grn": "Transcriptional regulation",
                    "trajectory": "Lineage ordering and fate",
                    "trajectory_viz": "Lineage figures",
                    "differential_expression": "Replicate-level changes within populations",
                    "differential_abundance": "Replicate-level population proportions",
                    "enrichment": "Gene-program and regulatory activity",
                    "integration_benchmark": "Batch mixing and cell-type preservation",
                    "population_identity": "Population identity and supporting evidence",
                }
                reason = families.get(stage_name, "Study preparation or supporting analysis") + "."
                if stage_name == "cell_cell_communication":
                    for method, omission in self.config.run.method_omissions.items():
                        reason += f" {method} omitted. {omission}"

            methods: tuple[str, ...] = ()
            if enabled and spec.config_field:
                from cellquorum.methods.stage_base import MethodDispatchStage

                settings = getattr(self.config, spec.config_field)
                stage = spec.factory() if spec.factory is not None else None
                if isinstance(stage, MethodDispatchStage):
                    methods = tuple(stage.planned_methods(self.config))
                elif getattr(settings, "method", None):
                    methods = (str(settings.method),)
            required_inputs = []
            stage_inputs = {
                "qc_evidence": "Raw counts with gene identifiers and library metadata",
                "differential_expression": "Raw counts and a replicated case/control design",
                "differential_abundance": "Population labels and biological replicate IDs",
                "grn": "Count layer, organism TF list, motif annotations and cisTarget rankings",
                "reference_mapping": "Labeled reference atlas and count-bearing query",
                "integration_benchmark": (
                    "Baseline and integrated embeddings, batch and biological labels"
                ),
            }
            method_inputs = {
                "liana": "Population labels, library IDs and expression",
                "tensor_c2c": "Per-library LIANA scores and sufficient eligible contexts",
                "nichenet": "Named receiver, receiver differential expression and NicheNet priors",
                "multinichenet": (
                    "Replicated contrast, raw counts and NicheNet ligand and receptor priors"
                ),
                "activity": ("Log expression, population labels and a weighted prior"),
                "dpt": "Neighbor representation and a biological root",
                "velocity": ("Splice counts, looms, or annotated Cell Ranger BAMs"),
                "palantir": "A diffusion representation and a biological root",
            }
            if enabled:
                if stage_name in stage_inputs:
                    required_inputs.append(stage_inputs[stage_name])
                required_inputs.extend(method_inputs[m] for m in methods if m in method_inputs)
            planned_stages.append(
                PlannedStage(
                    name=stage_name,
                    enabled=enabled,
                    status=status,
                    reason=reason,
                    methods=methods,
                    required_inputs=tuple(required_inputs),
                    expected_outputs=tuple(self.config.run.required_outputs.get(stage_name, [])),
                )
            )

        return planned_stages

    def _build_warnings(self) -> list[str]:
        """
        Build planner-level warnings from configuration and backend availability.

        Returns:
            List of warning messages.
        """

        warnings: list[str] = []
        registered = set(self.backend_registry.names())

        # Warn if GPU is preferred but neither GPU nor RAPIDS is available.
        if self.config.compute.prefer_gpu:
            # Check generic GPU and RAPIDS backend availability.
            gpu_available = "gpu" in registered and self.backend_registry.available("gpu")
            rapids_available = "rapids" in registered and self.backend_registry.available("rapids")

            # Add a warning when GPU was preferred but no GPU backend is currently available.
            if not gpu_available and not rapids_available:
                warnings.append(
                    "GPU acceleration is preferred, but no GPU/RAPIDS backend is currently "
                    "available. CPU fallback will be used if allowed."
                )

        # Warn if R is enabled but neither rpy2 nor Rscript is available.
        if self.config.r.enabled:
            # Check rpy2 and Rscript backend availability.
            r_available = "r" in registered and self.backend_registry.available("r")
            rscript_available = "rscript" in registered and self.backend_registry.available(
                "rscript"
            )

            # Add a warning when R was enabled but no R backend is currently available.
            if not r_available and not rscript_available:
                warnings.append(
                    "R-backed methods are enabled, but neither rpy2 nor Rscript is currently "
                    "available. R-backed stages will be skipped or require installation."
                )

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

    return PipelinePlanner(config=config, backend_registry=backend_registry).build_plan()
