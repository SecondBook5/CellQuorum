"""State-scoring stage: dispatches to the configured state-scoring method(s)."""

from __future__ import annotations

# Import the package so the methods register themselves as a side effect.
import cellquorum.state_scoring  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="state_scoring",
    order=170,
    config_flag="state_scoring",
    config_field="state_scoring",
    category="state_scoring",
)
class StateScoringStage(MethodDispatchStage):
    """Score cell-state programs with the configured method(s)."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "score_genes")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """Default to running both methods (obs scores + obsm AUCs) when unset."""
        augmented = dict(stage_config)
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "score_genes"}, {"method": "aucell"}]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: state scoring writes obs/obsm + a table, no strict postcondition."""


__all__ = ["StateScoringStage"]
