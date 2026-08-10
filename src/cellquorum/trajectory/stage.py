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

# Keys flattened from config.cellrank.* into each method's config dict.
_CELLRANK_KEYS = (
    "enabled",
    "cluster_key",
    "pseudotime_key",
    "cytotrace_key",
    "use_rep",
    "use_rep_fallback",
    "n_neighbors",
    "weight_connectivities",
    "n_components",
    "n_states",
    "n_terminal_states",
    "terminal_method",
    "predict_initial_states",
    "n_initial_states",
    "max_cells",
    "seed",
)

# Keys flattened from config.dpt.* into each method's config dict.
_DPT_KEYS = (
    "enabled",
    "use_rep",
    "use_rep_fallback",
    "n_neighbors",
    "n_comps",
    "n_dcs",
    "n_branchings",
    "root_key",
    "root_group",
    "root_marker_score_key",
    "exclude_outliers",
    "outlier_mad",
    "orient_by_score_key",
    "seed",
)

# Keys flattened from config.palantir.* into each method's config dict.
_PALANTIR_KEYS = (
    "enabled",
    "use_rep",
    "use_rep_fallback",
    "n_components",
    "knn",
    "n_eigs",
    "num_waypoints",
    "root_key",
    "root_group",
    "root_marker_score_key",
    "max_cells",
    "seed",
)

# Keys flattened from config.cytotrace.* into each method's config dict.
_CYTOTRACE_KEYS = (
    "enabled",
    "species",
    "counts_layer",
    "batch_size",
    "smooth_batch_size",
    "disable_parallelization",
    "seed",
)


class TrajectoryStage(MethodDispatchStage):
    """Run the configured trajectory method(s). Spec #1 registers 'velocity'."""

    name = "trajectory"
    stage_category = "trajectory"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "velocity")

    # method name -> (config.trajectory.<attr>, flatten keys) for that block.
    _METHOD_BLOCKS = {
        "velocity": ("velocity", _VELOCITY_KEYS),
        "cellrank": ("cellrank", _CELLRANK_KEYS),
        "dpt": ("dpt", _DPT_KEYS),
        "palantir": ("palantir", _PALANTIR_KEYS),
        "cytotrace": ("cytotrace", _CYTOTRACE_KEYS),
    }

    def _flatten_block(self, traj: object, name: str, target: dict) -> None:
        """Overlay config.trajectory.<name>.* onto ``target`` (existing keys win)."""
        block_info = self._METHOD_BLOCKS.get(name)
        if block_info is None or traj is None:
            return
        attr, keys = block_info
        block = getattr(traj, attr, None)
        if block is None:
            return
        block_dict = block.model_dump() if hasattr(block, "model_dump") else dict(block)
        for key in keys:
            if key in block_dict and key not in target:
                target[key] = block_dict[key]

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)

        # Determine which method blocks are actually selected, so we only flatten
        # the relevant config and never leak one method's values into another
        # (e.g. velocity's seed/n_neighbors bleeding into a cellrank-only run).
        methods_list = stage_config.get("methods")
        if methods_list:
            selected = {m.get("method") for m in methods_list if isinstance(m, dict)}
        elif "method" in stage_config:
            selected = {stage_config["method"]}
        else:
            selected = {"velocity"}  # matches the default applied below

        config = getattr(context, "config", None)
        traj = getattr(config, "trajectory", None) if config is not None else None

        # Multi-method chain: flatten each block into ITS OWN methods-list entry
        # rather than a shared top level. Keys that share a NAME across methods
        # but differ in MEANING (n_components = diffmap comps in palantir vs
        # Schur vectors in cellrank; also n_neighbors, max_cells, seed) would
        # otherwise collide first-wins and silently corrupt a downstream method.
        # _run_methods_list overlays each entry over the shared (cohort) keys, so
        # per-entry values win without leaking sideways.
        if methods_list and len(selected) >= 2:
            new_methods = []
            for entry in methods_list:
                if not isinstance(entry, dict):
                    new_methods.append(entry)
                    continue
                merged = dict(entry)
                self._flatten_block(traj, entry.get("method"), merged)
                new_methods.append(merged)
            augmented["methods"] = new_methods
            return augmented

        # Single-method path: exactly one method name is selected here, so flatten
        # that one block to the top level (unchanged behavior — _run_methods_list
        # applies shared keys to it). Stop after the first match to make the
        # single-block invariant explicit and immune to a future gate change.
        for name in self._METHOD_BLOCKS:
            if name in selected:
                self._flatten_block(traj, name, augmented)
                break

        # Default to the single velocity method when nothing was specified.
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "velocity"}]

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes h5ad artifacts + uns keys, no obs/var postcondition."""


__all__ = ["TrajectoryStage"]
