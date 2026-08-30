"""Trajectory-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

import cellquorum.stages.trajectory.viz  # noqa: F401  (side-effect: method registration)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

_VIZ_CONFIG_KEYS = (
    "enabled",
    "figure_formats",
    "dpi",
    "top_k",
    "embedding_basis",
    "pseudotime_keys",
    "lineages",
    "genes",
    "cluster_key",
    "heatmap_genes",
    "heatmap_score_key",
    "heatmap_state_key",
    "heatmap_n_bins",
    "heatmap_max_genes",
    "heatmap_corr_cut",
    "heatmap_expr_cmap",
)

_DEFAULT_METHODS = [
    {"method": "pseudotime_viz"},
    {"method": "fate_viz"},
    {"method": "driver_viz"},
    {"method": "gene_trend_viz"},
    {"method": "macrostate_viz"},
    {"method": "velocity_viz"},
    {"method": "pseudotime_heatmap"},
]


@register_stage(
    name="trajectory_viz",
    order=310,
    config_flag="trajectory_viz",
    config_field="trajectory_viz",
    category="trajectory_viz",
)
class TrajectoryVizStage(MethodDispatchStage):
    """Render publication figures from the trajectory producers' outputs."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "pseudotime_viz")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        viz_cfg = getattr(config, "trajectory_viz", None)
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
                augmented["condition_col"] = getattr(design, "condition_col", "condition")
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [dict(m) for m in _DEFAULT_METHODS]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: this stage writes figures, no obs/var postcondition."""


__all__ = ["TrajectoryVizStage"]
