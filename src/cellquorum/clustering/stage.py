"""Clustering stage: dispatch to the configured clustering method.

Mirrors DimensionalityStage: resolves its config sub-block from a pydantic-or-dict
context, dispatches to the configured method via the registry, and validates that
cluster labels landed in obs before handing the AnnData downstream.
"""

from __future__ import annotations

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.methods.stage_base import MethodDispatchStage


class ClusteringStage(MethodDispatchStage):
    """Config-driven clustering stage."""

    # Stage identity.
    name = "clustering"
    stage_category = "clustering"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured clustering method (default 'leiden')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "leiden")

    def run(self, context: object) -> StageResult:
        """
        Run the configured clustering method and validate the labels landed.

        Args:
            context: Pipeline context.

        Returns:
            StageResult whose AnnData carries the cluster-label obs column.
        """

        # Resolve this stage's config from either a pydantic config or a dict.
        stage_config = resolve_stage_config(context, self.name)

        # Dispatch to the configured method with the resolved sub-block.
        method_name = self._select_method_name(stage_config)
        method = self._registry.get(self.stage_category, method_name)()
        adata = context.require_adata()
        donor_col = getattr(context, "donor_col", None)
        outcome = method.run(adata, stage_config, context, donor_col=donor_col)

        # Convert a skip into a recorded StageResult.
        if isinstance(outcome, MethodSkip):
            return StageResult(
                adata=adata,
                warnings=[outcome.reason],
                metrics={"skipped": True, **outcome.details},
            )

        # Validate the cluster labels landed in obs.
        key_added = stage_config.get("key_added", "leiden")
        DataContract(required_obs=[key_added]).validate(outcome.adata)
        return outcome


__all__ = ["ClusteringStage"]
