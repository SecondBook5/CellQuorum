"""Registration + stage-config-flattening tests for CellRank."""

from __future__ import annotations

import types

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.config import CellRankConfig, TrajectoryConfig
from cellquorum.trajectory.stage import TrajectoryStage


def test_cellrank_registered():
    import cellquorum.trajectory  # noqa: F401 — trigger registration

    assert METHOD_REGISTRY.has("trajectory", "cellrank")


def test_stage_flattens_cellrank_keys():
    # Build a context whose config carries a trajectory block with a cellrank
    # sub-config and a methods list selecting cellrank.
    traj = TrajectoryConfig(
        methods=[{"method": "cellrank"}],
        cellrank=CellRankConfig(cluster_key="my_labels", n_states=5),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)

    stage = TrajectoryStage()
    # stage_config mirrors the real flow (resolve_stage_config → traj.model_dump),
    # which carries the methods list that selects which block to flatten.
    augmented = stage._augment_config(
        context, {"methods": [{"method": "cellrank"}], "cluster_key": None}
    )
    # Flattened cellrank keys are present.
    assert augmented["n_states"] == 5
    # A pre-existing key in the stage config is not overwritten.
    assert augmented["cluster_key"] is None


def test_cellrank_only_run_does_not_inherit_velocity_shared_keys():
    # A cellrank-only run must receive cellrank's OWN values for keys the two
    # method configs share (seed/n_neighbors/use_rep/...), not velocity's.
    from cellquorum.trajectory.config import VelocityConfig

    traj = TrajectoryConfig(
        methods=[{"method": "cellrank"}],
        velocity=VelocityConfig(seed=111, n_neighbors=42),
        cellrank=CellRankConfig(seed=999, n_neighbors=7),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)

    stage = TrajectoryStage()
    augmented = stage._augment_config(context, {"methods": [{"method": "cellrank"}]})
    # cellrank's values win; velocity's do not leak in.
    assert augmented["seed"] == 999
    assert augmented["n_neighbors"] == 7
