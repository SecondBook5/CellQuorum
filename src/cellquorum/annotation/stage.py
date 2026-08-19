"""Annotation stage: dispatch to the configured annotation method."""

from __future__ import annotations

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class AnnotationStage(MethodDispatchStage):
    """Config-driven annotation stage."""

    # Stage identity.
    name = "annotation"
    stage_category = "annotation"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured annotation method (default 'marker_vote')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "marker_vote")

    def _validate_output(self, result: StageResult) -> None:
        """Validate the cell-type label landed in obs (non-skip only)."""

        # Skipped results pass through without validation.
        if result.metrics.get("skipped"):
            return

        # The key_added column must be present. Re-resolve it is not available
        # here (no context); read it from metrics recorded by the method, else
        # fall back to the default.
        key_added = result.metrics.get("key_added", "cell_type")
        DataContract(required_obs=[key_added]).validate(result.adata)


__all__ = ["AnnotationStage"]
