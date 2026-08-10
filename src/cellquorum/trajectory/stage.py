"""Trajectory stage: dispatches the configured method(s) (spec #1: velocity)."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.trajectory  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage

# Keys flattened from config.velocity.* into each method's config dict.
_VELOCITY_KEYS = (
    "enabled",
    "grouping_col",
    "sample_col",
    "loom_path_col",
    "groups",
    "use_rep",
    "use_rep_fallback",
    "mode",
    "min_shared_counts",
    "n_top_genes",
    "n_pcs",
    "n_neighbors",
    "min_cells",
    "n_jobs",
    "seed",
    "generation",
)


class TrajectoryStage(MethodDispatchStage):
    """Run the configured trajectory method(s). Spec #1 registers 'velocity'."""

    name = "trajectory"
    stage_category = "trajectory"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "velocity")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)

        # Flatten config.trajectory.velocity.* into the stage config.
        config = getattr(context, "config", None)
        traj = getattr(config, "trajectory", None) if config is not None else None
        velocity = getattr(traj, "velocity", None) if traj is not None else None
        if velocity is not None:
            velocity_dict = (
                velocity.model_dump() if hasattr(velocity, "model_dump") else dict(velocity)
            )
            for key in _VELOCITY_KEYS:
                if key in velocity_dict and key not in augmented:
                    augmented[key] = velocity_dict[key]

        # Default to the single velocity method when nothing was specified.
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "velocity"}]

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes h5ad artifacts + uns keys, no obs/var postcondition."""


__all__ = ["TrajectoryStage"]
