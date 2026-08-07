"""Differential-expression stage: dispatches to the configured DE method."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.differential_expression  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class DifferentialExpressionStage(MethodDispatchStage):
    """Run the configured pseudobulk differential-expression method."""

    name = "differential_expression"
    stage_category = "differential_expression"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured DE method name."""

        # Default to the donor-blocked pseudobulk edgeR method.
        return config.get("method", "pseudobulk_edger")

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition on obs/var; DE writes a table artifact."""
