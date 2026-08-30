"""Config tests for the DPT + Palantir pseudotime methods."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.stages.trajectory.config import DptConfig, PalantirConfig, TrajectoryConfig


def test_dpt_config_defaults():
    c = DptConfig()
    assert c.enabled is True
    assert c.use_rep is None
    assert c.use_rep_fallback == ["X_scANVI", "X_scVI", "X_pca"]
    assert c.n_neighbors == 15
    assert c.n_comps == 15
    assert c.n_dcs == 10
    assert c.n_branchings == 0
    assert c.root_key is None
    assert c.root_group is None
    assert c.root_marker_score_key is None
    assert c.exclude_outliers is False
    assert c.outlier_mad == 5.0
    assert c.orient_by_score_key is None
    assert c.seed == 1337


def test_palantir_config_defaults():
    c = PalantirConfig()
    assert c.enabled is True
    assert c.use_rep is None
    assert c.use_rep_fallback == ["X_scANVI", "X_scVI", "X_pca"]
    assert c.n_components == 10
    assert c.knn == 30
    assert c.n_eigs == 10
    assert c.num_waypoints == 1200
    assert c.root_key is None
    assert c.root_group is None
    assert c.root_marker_score_key is None
    assert c.max_cells is None
    assert c.seed == 1337


def test_configs_forbid_extra():
    with pytest.raises(ValidationError):
        DptConfig(bogus=1)
    with pytest.raises(ValidationError):
        PalantirConfig(bogus=1)


def test_trajectory_config_has_pseudotime_fields():
    t = TrajectoryConfig()
    assert isinstance(t.dpt, DptConfig)
    assert isinstance(t.palantir, PalantirConfig)
