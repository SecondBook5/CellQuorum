"""Tests for trajectory stage wiring into CellQuorum engine integration points."""

from __future__ import annotations


def test_config_has_trajectory():
    """Verify trajectory config field and toggle are present in CellQuorumConfig."""
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.stages.trajectory.config import TrajectoryConfig

    cfg = CellQuorumConfig()
    assert isinstance(cfg.trajectory, TrajectoryConfig)
    assert cfg.stages.trajectory is True


def test_trajectory_stage_registered():
    """Verify trajectory stage is registered in default stage registry."""
    from cellquorum.core.executor import build_default_stage_registry
    from cellquorum.stages.trajectory.stage import TrajectoryStage

    reg = build_default_stage_registry()
    assert "trajectory" in reg.stages
    assert isinstance(reg.stages["trajectory"], TrajectoryStage)


def test_planner_orders_trajectory_after_molecular_before_ccc():
    """Verify trajectory is ordered after molecular_inference and before cell_cell_communication."""
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.planner import build_pipeline_plan

    plan = build_pipeline_plan(CellQuorumConfig())
    order = plan.enabled_stage_names()
    assert "trajectory" in order
    assert order.index("molecular_inference") < order.index("trajectory")
    assert order.index("trajectory") < order.index("cell_cell_communication")


def test_manifest_accepts_loom_path_column():
    """Verify manifest accepts and exposes optional loom_path column."""
    import pandas as pd

    from cellquorum.io.manifest import validate_manifest_dataframe

    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "path": ["/data/s1.h5ad"],
            "loom_path": ["/data/s1.loom"],
        }
    )
    manifest = validate_manifest_dataframe(df)
    assert manifest.has_column("loom_path")
