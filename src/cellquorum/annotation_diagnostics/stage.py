"""Annotation-diagnostics stage: dispatch to scDiagnostics method."""

from __future__ import annotations

from cellquorum.contracts import CellQuorumContractError
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class AnnotationDiagnosticsStage(MethodDispatchStage):
    """Config-driven annotation-confidence diagnostic stage.

    Computes annotation confidence metrics (anomaly detection, kNN probabilities,
    categorization entropy) via scDiagnostics. READ-ONLY: adds scdiag_* obs
    columns but never modifies cell_type or embeddings.
    """

    name = "annotation_diagnostics"
    stage_category = "annotation_diagnostics"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured diagnostic method (default 'scdiagnostics')."""
        return config.get("method", "scdiagnostics")

    def _validate_output(self, result: StageResult) -> None:
        """Assert diagnostic metrics were recorded (non-skip only)."""
        if result.metrics.get("skipped"):
            return
        # The diagnostics_computed key must exist in metrics.
        if "diagnostics_computed" not in result.metrics:
            raise CellQuorumContractError(
                "annotation_diagnostics did not produce 'diagnostics_computed' in " "metrics."
            )


__all__ = ["AnnotationDiagnosticsStage"]
