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
        Resolve and execute the configured method, or a list of methods.

        Single-method path (config has a scalar ``method``) is unchanged. When
        the resolved config carries a ``methods`` list, each entry is a full
        method sub-config run in order against the same AnnData; a per-method
        MethodSkip is recorded as a warning and does not abort the others.

        Args:
            context: Pipeline context exposing ``require_adata()``, ``config``,
                and optionally ``donor_col``.

        Returns:
            A StageResult — the (final) method's result, or a recorded skip.
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

        # Multi-method path: run a list of method sub-configs in order.
        methods = stage_config.get("methods")
        if methods:
            return self._run_methods_list(adata, stage_config, methods, context)

        # Single-method path (unchanged behavior).
        outcome = self._run_single_method(adata, stage_config, context)
        if isinstance(outcome, MethodSkip):
            return StageResult.skipped(
                adata=adata,
                reason=outcome.reason,
                warnings=[outcome.reason],
                metrics=outcome.details,
            )
        self._validate_output(outcome)
        return outcome

    def _run_single_method(
        self,
        adata: object,
        method_config: dict,
        context: object,
    ) -> StageResult | MethodSkip:
        """
        Resolve and execute one method from its config sub-block.

        Args:
            adata: Active AnnData threaded through the stage.
            method_config: Config dict carrying at least a ``method`` name plus
                that method's own keys.
            context: Pipeline context (passed through to the method).

        Returns:
            The method's StageResult, or a MethodSkip when a guard trips.
        """

        # Resolve the selected method name and look it up in the registry.
        method_name = self._select_method_name(method_config)
        method_cls = self._registry.get(self.stage_category, method_name)
        method = method_cls()

        # Execute with the donor column (if the context provides one).
        donor_col = getattr(context, "donor_col", None)
        return method.run(adata, method_config, context, donor_col=donor_col)

    def _run_methods_list(
        self,
        adata: object,
        stage_config: dict,
        methods: list[dict],
        context: object,
    ) -> StageResult:
        """
        Run a list of method sub-configs in order against the same AnnData.

        Each entry inherits shared stage-level keys (everything except the
        ``methods`` list itself) so cohort overlays and common settings apply,
        then overlays the entry's own keys. A per-method MethodSkip is recorded
        as a warning and does not abort the remaining methods.

        Args:
            adata: Active AnnData threaded through every method.
            stage_config: The resolved stage config (provides shared keys).
            methods: List of per-method config dicts.
            context: Pipeline context.

        Returns:
            A single StageResult aggregating notes/warnings/metrics; its
            ``.adata`` is the object after every method has run.
        """

        # Shared keys apply to every method (drop the list key itself).
        shared = {k: v for k, v in stage_config.items() if k != "methods"}

        notes: list[str] = []
        warnings: list[str] = []
        per_method_metrics: list[dict] = []
        current = adata

        for entry in methods:
            # Build this method's effective config: shared keys then entry keys.
            method_config = {**shared, **dict(entry)}
            outcome = self._run_single_method(current, method_config, context)

            if isinstance(outcome, MethodSkip):
                # Record the skip; keep going so one missing method (e.g. no
                # CellTypist model) does not lose the others' output.
                warnings.append(
                    f"{self.name}: method "
                    f"'{self._select_method_name(method_config)}' skipped: "
                    f"{outcome.reason}"
                )
                per_method_metrics.append({"skipped": True, "reason": outcome.reason})
                continue

            # Validate and thread this method's AnnData into the next method.
            self._validate_output(outcome)
            current = outcome.adata
            notes.extend(outcome.notes)
            warnings.extend(outcome.warnings)
            per_method_metrics.append(dict(outcome.metrics))

        return StageResult(
            adata=current,
            notes=notes,
            warnings=warnings,
            metrics={
                "n_methods": len(methods),
                "per_method": per_method_metrics,
            },
        )

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
