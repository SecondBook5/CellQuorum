"""Dimensionality stage: dispatch to the configured reduction method.

Subclasses the Phase-1 MethodDispatchStage. Selects the reduction method from
config (default "pca"), runs it via the method registry, and validates that the
output carries an ``X_pca`` embedding before handing the AnnData downstream.
"""

from __future__ import annotations

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.methods.stage_base import MethodDispatchStage


class DimensionalityStage(MethodDispatchStage):
    """Config-driven dimensionality-reduction stage."""

    # Stage identity (matches the config sub-block + registry category).
    name = "dimensionality"
    stage_category = "dimensionality"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured reduction method (default 'pca')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "pca")

    def run(self, context: object) -> StageResult:
        """
        Run the configured reduction method and validate the output embedding.

        Args:
            context: Pipeline context.

        Returns:
            StageResult whose AnnData carries obsm["X_pca"].
        """

        # Resolve this stage's config from either a pydantic config or a dict.
        stage_config = resolve_stage_config(context, self.name)

        # Dispatch to the configured method via the registry (base-class logic),
        # but feed it the resolved sub-block by shimming context.config access.
        result = self._dispatch(context, stage_config)

        # Validate the reduction produced an embedding (skips pass through).
        if not result.metrics.get("skipped"):
            DataContract(required_obsm=["X_pca"]).validate(result.adata)
        return result

    def _dispatch(self, context: object, stage_config: dict) -> StageResult:
        """
        Look up and run the configured method with the resolved config.

        Args:
            context: Pipeline context (provides adata + donor_col).
            stage_config: Resolved config sub-block for this stage.

        Returns:
            StageResult from the method (or a recorded skip).
        """

        # Resolve the method name and fetch its class from the registry.
        method_name = self._select_method_name(stage_config)
        method = self._registry.get(self.stage_category, method_name)()

        # Execute with the resolved sub-block and the context's donor column.
        adata = context.require_adata()
        donor_col = getattr(context, "donor_col", None)
        from cellquorum.methods.base import MethodSkip

        outcome = method.run(adata, stage_config, context, donor_col=donor_col)
        if isinstance(outcome, MethodSkip):
            return StageResult(
                adata=adata,
                warnings=[outcome.reason],
                metrics={"skipped": True, **outcome.details},
            )
        return outcome


__all__ = ["DimensionalityStage"]
