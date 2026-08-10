"""Registration + stage-config-flatten tests for dpt/palantir methods."""

from __future__ import annotations

import types

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.config import DptConfig, PalantirConfig, TrajectoryConfig
from cellquorum.trajectory.stage import TrajectoryStage


def test_dpt_and_palantir_registered():
    import cellquorum.trajectory  # noqa: F401 — trigger registration

    assert METHOD_REGISTRY.has("trajectory", "dpt")
    assert METHOD_REGISTRY.has("trajectory", "palantir")


def test_dpt_only_run_does_not_inherit_velocity_shared_keys():
    from cellquorum.trajectory.config import VelocityConfig

    traj = TrajectoryConfig(
        methods=[{"method": "dpt"}],
        velocity=VelocityConfig(seed=111, n_neighbors=42),
        dpt=DptConfig(seed=999, n_neighbors=7, root_marker_score_key="stem_score"),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)
    stage = TrajectoryStage()
    augmented = stage._augment_config(context, {"methods": [{"method": "dpt"}]})
    assert augmented["seed"] == 999
    assert augmented["n_neighbors"] == 7


def test_palantir_only_flattens_its_keys():
    traj = TrajectoryConfig(
        methods=[{"method": "palantir"}],
        palantir=PalantirConfig(num_waypoints=77, knn=11),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)
    stage = TrajectoryStage()
    augmented = stage._augment_config(context, {"methods": [{"method": "palantir"}]})
    assert augmented["num_waypoints"] == 77
    assert augmented["knn"] == 11
