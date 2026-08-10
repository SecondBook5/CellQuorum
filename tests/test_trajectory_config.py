# tests/test_trajectory_config.py
from __future__ import annotations


def test_trajectory_defaults():
    from cellquorum.trajectory.config import TrajectoryConfig

    c = TrajectoryConfig()
    assert c.enabled is True
    assert c.methods == []
    assert c.velocity.enabled is True
    assert c.velocity.grouping_col == "cell_type"
    assert c.velocity.sample_col == "sample_id"
    assert c.velocity.loom_path_col == "loom_path"
    assert c.velocity.groups is None
    assert c.velocity.use_rep is None
    assert c.velocity.use_rep_fallback == ["X_scANVI", "X_scVI", "X_pca"]
    assert c.velocity.mode == "dynamical"
    assert c.velocity.min_shared_counts == 20
    assert c.velocity.n_top_genes == 2000
    assert c.velocity.n_pcs == 30
    assert c.velocity.n_neighbors == 30
    assert c.velocity.min_cells == 30
    assert c.velocity.n_jobs == 1
    assert c.velocity.seed == 1337
    assert c.velocity.generation.generate_missing is False
    assert c.velocity.generation.bam_dir is None
    assert c.velocity.generation.gtf_path is None
    assert c.velocity.generation.repeat_mask is None
    assert c.velocity.generation.threads == 8
    assert c.velocity.generation.samtools_memory == 2000


def test_trajectory_strict_rejects_unknown_field():
    import pytest
    from pydantic import ValidationError

    from cellquorum.trajectory.config import TrajectoryConfig

    with pytest.raises(ValidationError):
        TrajectoryConfig(not_a_field=1)


def test_velocity_config_strict_nested():
    import pytest
    from pydantic import ValidationError

    from cellquorum.trajectory.config import VelocityConfig

    with pytest.raises(ValidationError):
        VelocityConfig(nope=1)
