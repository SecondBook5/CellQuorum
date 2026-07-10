"""Clustering stage: dispatch to the configured clustering method.

Mirrors DimensionalityStage: resolves its config sub-block from a pydantic-or-dict
context, dispatches to the configured method via the registry, and validates that
cluster labels landed in obs before handing the AnnData downstream.
"""

from __future__ import annotations

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.registry import MethodRegistry
from cellquorum.methods.stage_base import MethodDispatchStage


class ClusteringStage(MethodDispatchStage):
    """Config-driven clustering stage."""

    # Stage identity.
    name = "clustering"
    stage_category = "clustering"

    def __init__(self, registry: MethodRegistry | None = None) -> None:
        super().__init__(registry)
        # Store the key_added from config so _validate_output can access it.
        self._key_added = None

    def _select_method_name(self, config: dict) -> str:
        """Return the configured clustering method (default 'leiden')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "leiden")

    def run(self, context: object) -> StageResult:
        """Override run to store key_added for validation."""
        # Resolve config to extract key_added before calling base run.
        from cellquorum.methods.context_access import resolve_stage_config

        stage_config = resolve_stage_config(context, self.name)
        self._key_added = stage_config.get("key_added", "leiden")
        # Now call base run which will handle enabled check, dispatch, and validation.
        return super().run(context)

    def _validate_output(self, result: StageResult) -> None:
        """Validate that cluster labels landed in the configured obs column."""

        # Skipped results pass through without validation.
        if not result.metrics.get("skipped"):
            key_added = self._key_added or "leiden"
            DataContract(required_obs=[key_added]).validate(result.adata)


__all__ = ["ClusteringStage"]
