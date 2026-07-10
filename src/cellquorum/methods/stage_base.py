"""Abstract Stage that dispatches to a config-selected AnalysisMethod.

Concrete stages (AmbientCorrectionStage, CellCellCommunicationStage, ...) inherit
this. They only declare their ``stage_category`` and how to read the method name
from config; the base handles registry lookup, execution, and turning a
``MethodSkip`` into a recorded (non-silent) skipped StageResult. This is the
class that satisfies the existing ``PipelineStage`` Protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cellquorum.core.stage import StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.registry import METHOD_REGISTRY, MethodRegistry


class MethodDispatchStage(ABC):
    """
    Abstract PipelineStage that runs whichever method the config selects.
    """

    # Stable stage name (set by subclasses); satisfies PipelineStage.name.
    name: str

    # Stage category used for registry lookup (usually equals name).
    stage_category: str

    def __init__(self, registry: MethodRegistry | None = None) -> None:
        """
        Initialize the stage.

        Args:
            registry: Method registry to resolve against. Defaults to the
                module-level METHOD_REGISTRY singleton.
        """

        # Store the registry (dependency-injected for tests).
        self._registry = registry or METHOD_REGISTRY

    @abstractmethod
    def _select_method_name(self, config: dict) -> str:
        """Return the configured method name for this stage from the run config."""

    def run(self, context: object) -> StageResult:
        """
        Resolve and execute the configured method.

        Args:
            context: Pipeline context exposing ``require_adata()``, ``config``,
                and optionally ``donor_col``.

        Returns:
            A StageResult — either the method's result, or a recorded skipped
            result carrying the skip reason as a warning.
        """

        # Pull the active AnnData off the context.
        adata = context.require_adata()

        # Resolve this stage's config sub-block (handles both dict and pydantic).
        from cellquorum.methods.context_access import resolve_stage_config

        stage_config = resolve_stage_config(context, self.name)

        # Honor the enabled flag: if disabled, return a recorded skip.
        if not stage_config.get("enabled", True):
            return StageResult(
                adata=adata,
                warnings=[f"{self.name} disabled by config"],
                metrics={"skipped": True, "reason": "disabled by config"},
            )

        # Resolve the selected method name and look it up in the registry.
        method_name = self._select_method_name(stage_config)
        method_cls = self._registry.get(self.stage_category, method_name)
        method = method_cls()

        # Execute with the donor column (if the context provides one).
        donor_col = getattr(context, "donor_col", None)
        outcome = method.run(adata, stage_config, context, donor_col=donor_col)

        # Convert a MethodSkip into a recorded skipped StageResult. NOTE: because
        # the PipelineStage Protocol only permits returning a StageResult, a method
        # skip surfaces to the executor as a SUCCESSFUL record carrying a warning and
        # metrics["skipped"]=True — it does NOT populate core.stage's StageSkipReason
        # machinery. This is non-silent by design; downstream reporting must key on
        # metrics["skipped"], not on record.status, to distinguish skipped methods.
        if isinstance(outcome, MethodSkip):
            return StageResult(
                adata=adata,
                warnings=[outcome.reason],
                metrics={"skipped": True, **outcome.details},
            )

        # Call the overridable validation hook before returning.
        self._validate_output(outcome)
        return outcome

    def _validate_output(self, result: StageResult) -> None:  # noqa: B027
        """
        Hook for subclass-specific output validation.

        Override this to validate stage-specific postconditions. The default is a no-op.

        Args:
            result: The StageResult returned by the method.
        """


__all__ = ["MethodDispatchStage"]
