# Pipeline step (order=40): feature_selection — select highly variable genes.
"""Feature-selection stage: dispatch to the configured HVG method."""

from __future__ import annotations

from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import CellScope, CellScopePolicy, register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="feature_selection",
    order=40,
    config_flag="feature_selection",
    config_field="feature_selection",
    category="feature_selection",
    # Fits HVG means, variances and dispersions across cells, so damaged cells could
    # otherwise shape the biological reference every later stage is measured against.
    cell_scope=CellScopePolicy(fit_scope=CellScope.CORE),
)
class FeatureSelectionStage(MethodDispatchStage):
    """Config-driven highly-variable-gene selection stage."""

    def _select_method_name(self, config: dict) -> str:
        """Return the configured HVG method (default 'seurat')."""

        return config.get("method", "seurat")

    def _validate_output(self, result: StageResult) -> None:
        """Assert HVGs were flagged (non-skip only)."""

        if result.metrics.get("skipped"):
            return
        # The highly_variable column must exist and flag at least one gene.
        if "highly_variable" not in result.adata.var.columns:
            raise CellQuorumContractError(
                "feature_selection did not produce var['highly_variable'] column."
            )
        if int(result.adata.var["highly_variable"].sum()) < 1:
            raise CellQuorumContractError("feature_selection produced zero highly-variable genes.")


__all__ = ["FeatureSelectionStage"]
