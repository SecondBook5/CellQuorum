# Pipeline step (order=175): state_scoring_viz — quantitative companion to state_score_umap.
"""State-scoring visualization stage: dispatches the configured viz method(s).

Runs right after state_scoring (170), before discovery (180) -- unlike
state_score_umap (which lives in the embeddings stage because it needs a
computed UMAP/PHATE basis), this stage only needs state_scoring's own
obsm/uns output plus obs cluster/condition columns, so it doesn't need to
wait for embeddings (200).
"""

from __future__ import annotations

import cellquorum.stages.state_scoring.viz  # noqa: F401  (side-effect: registers methods)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

_VIZ_CONFIG_KEYS = (
    "enabled",
    "score_source",
    "programs",
    "cluster_key",
    "cluster_labels",
    "condition_col",
    "case",
    "control",
    "figure_formats",
    "dpi",
)
_DEFAULT_METHODS = ["state_score_violin"]


@register_stage(
    name="state_scoring_viz",
    order=175,
    config_flag="state_scoring_viz",
    config_field="state_scoring_viz",
    category="state_scoring_viz",
)
class StateScoringVizStage(MethodDispatchStage):
    """Render quantitative (violin + p-value) figures from state_scoring's output."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "state_score_violin")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        viz_cfg = getattr(config, "state_scoring_viz", None)
        if viz_cfg is not None:
            for key in _VIZ_CONFIG_KEYS:
                if key not in augmented:
                    value = getattr(viz_cfg, key, None)
                    if value is not None:
                        augmented[key] = value
        design = getattr(config, "design", None)
        if design is not None:
            if not augmented.get("case"):
                augmented["case"] = getattr(design, "case", None)
            if not augmented.get("control"):
                augmented["control"] = getattr(design, "control", None)
            if not augmented.get("condition_col"):
                augmented["condition_col"] = getattr(design, "condition_col", None)
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": m} for m in _DEFAULT_METHODS]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: this stage writes figures, no obs/var postcondition."""


__all__ = ["StateScoringVizStage"]
