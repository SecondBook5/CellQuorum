"""Trajectory-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

import cellquorum.trajectory_viz  # noqa: F401  (side-effect: method registration)
from cellquorum.core.stage import StageResult
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
)

_DEFAULT_METHODS = [
    {"method": "pseudotime_viz"},
    {"method": "fate_viz"},
    {"method": "driver_viz"},
    {"method": "gene_trend_viz"},
    {"method": "macrostate_viz"},
    {"method": "velocity_viz"},
]


class TrajectoryVizStage(MethodDispatchStage):
    """Render publication figures from the trajectory producers' outputs."""

    name = "trajectory_viz"
    stage_category = "trajectory_viz"

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
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [dict(m) for m in _DEFAULT_METHODS]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: this stage writes figures, no obs/var postcondition."""


__all__ = ["TrajectoryVizStage"]
