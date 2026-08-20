"""Integration stage: dispatch to the configured batch-correction method.

Subclasses the generalized MethodDispatchStage (config resolution, enabled-gate,
dispatch, skip conversion are inherited). Validates that the corrected embedding
was written — the guard that makes a silently-failed integration loud.
"""

from __future__ import annotations

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="integration",
    order=60,
    config_flag="integration",
    config_field="integration",
    category="integration",
)
class IntegrationStage(MethodDispatchStage):
    """Config-driven batch-integration stage."""

    def _select_method_name(self, config: dict) -> str:
        """Return the configured integration method (default 'harmony')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "harmony")

    def _validate_output(self, result: StageResult) -> None:
        """Validate that the corrected embedding was written (non-skip only)."""

        # Skipped results pass through without validation.
        if result.metrics.get("skipped"):
            return

        # Resolve the expected output embedding key from provenance/metrics and
        # assert it exists — turns a silent Harmony fallback into a loud failure.
        output_rep = result.metrics.get("output_rep", "X_pca_harmony")
        DataContract(required_obsm=[output_rep]).validate(result.adata)


__all__ = ["IntegrationStage"]
