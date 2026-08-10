"""Registration + stage-config-flattening tests for CytoTRACE 2."""

from __future__ import annotations

import types

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.config import CytoTraceConfig, TrajectoryConfig
from cellquorum.trajectory.stage import TrajectoryStage


def test_cytotrace_registered():
    import cellquorum.trajectory  # noqa: F401 — trigger registration

    assert METHOD_REGISTRY.has("trajectory", "cytotrace")


def test_stage_flattens_cytotrace_keys():
    traj = TrajectoryConfig(
        methods=[{"method": "cytotrace"}],
        cytotrace=CytoTraceConfig(species="mouse", seed=7),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)

    stage = TrajectoryStage()
    augmented = stage._augment_config(context, {"methods": [{"method": "cytotrace"}]})
    assert augmented["species"] == "mouse"
    assert augmented["seed"] == 7


def test_cytotrace_only_run_does_not_inherit_velocity_shared_keys():
    from cellquorum.trajectory.config import VelocityConfig

    traj = TrajectoryConfig(
        methods=[{"method": "cytotrace"}],
        velocity=VelocityConfig(seed=111),
        cytotrace=CytoTraceConfig(seed=999),
    )
    config = types.SimpleNamespace(trajectory=traj, cohort=None)
    context = types.SimpleNamespace(config=config)

    stage = TrajectoryStage()
    augmented = stage._augment_config(context, {"methods": [{"method": "cytotrace"}]})
    # cytotrace's own seed wins; velocity's does not leak in.
    assert augmented["seed"] == 999
