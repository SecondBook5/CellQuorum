"""Dimensionality stage: dispatch to the configured reduction method.

Subclasses the Phase-1 MethodDispatchStage. Selects the reduction method from
config (default "pca"), runs it via the method registry, and validates that the
output carries an ``X_pca`` embedding before handing the AnnData downstream.
"""

from __future__ import annotations

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
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

    def _validate_output(self, result: StageResult) -> None:
        """Validate that the reduction produced an X_pca embedding."""

        # Skipped results pass through without validation.
        if not result.metrics.get("skipped"):
            DataContract(required_obsm=["X_pca"]).validate(result.adata)


__all__ = ["DimensionalityStage"]
