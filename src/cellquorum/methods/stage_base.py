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

        # Overlay the central cohort schema so structural obs keys are declared
        # once and every dispatched method sees the resolved values.
        stage_config = _apply_cohort_overlay(context, stage_config)

        # Honor the enabled flag: if disabled, return a recorded skip.
        if not stage_config.get("enabled", True):
            return StageResult.skipped(
                adata=adata,
                reason="disabled by config",
                warnings=[f"{self.name} disabled by config"],
            )

        # Resolve the selected method name and look it up in the registry.
        method_name = self._select_method_name(stage_config)
        method_cls = self._registry.get(self.stage_category, method_name)
        method = method_cls()

        # Execute with the donor column (if the context provides one).
        donor_col = getattr(context, "donor_col", None)
        outcome = method.run(adata, stage_config, context, donor_col=donor_col)

        # Convert a MethodSkip into an explicit skipped StageResult.
        if isinstance(outcome, MethodSkip):
            return StageResult.skipped(
                adata=adata,
                reason=outcome.reason,
                warnings=[outcome.reason],
                metrics=outcome.details,
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


# Cohort attributes that overlay onto same-named stage config keys. These are
# the structural obs columns a dataset declares once (config.cohort). When a
# cohort field is set it takes precedence, so the biological structure is
# declared in one place; when unset, the stage's own key is left untouched.
_COHORT_OVERLAY_KEYS: tuple[str, ...] = (
    "batch_key",
    "sample_key",
    "donor_key",
    "condition_key",
)


def _apply_cohort_overlay(context: object, stage_config: dict) -> dict:
    """
    Return a copy of ``stage_config`` with cohort-declared keys overlaid.

    Only keys the cohort block actually sets are overlaid, and only onto config
    keys of the same name. When there is no cohort block (or it is empty), the
    stage config is returned unchanged, so existing configs behave identically.

    Args:
        context: Pipeline context exposing ``config`` (may lack a cohort block).
        stage_config: The resolved per-stage config dict.

    Returns:
        A (possibly updated) config dict.
    """

    config = getattr(context, "config", None)
    cohort = getattr(config, "cohort", None)
    if cohort is None:
        return stage_config

    overlaid = dict(stage_config)
    for key in _COHORT_OVERLAY_KEYS:
        cohort_value = getattr(cohort, key, None)
        # Overlay only same-named keys the stage already understands, so we
        # never inject a key a method does not expect.
        if cohort_value and key in overlaid:
            overlaid[key] = cohort_value
    return overlaid


__all__ = ["MethodDispatchStage"]
