"""CellRankConfig schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.trajectory.config import CellRankConfig, TrajectoryConfig


def test_cellrank_config_defaults():
    cfg = CellRankConfig()
    assert cfg.enabled is True
    assert cfg.cluster_key == "cell_type"
    assert cfg.pseudotime_key is None
    assert cfg.cytotrace_key is None
    assert cfg.use_rep is None
    assert cfg.use_rep_fallback == ["X_scANVI", "X_scVI", "X_pca"]
    assert cfg.n_neighbors == 30
    assert cfg.weight_connectivities == pytest.approx(0.2)
    assert cfg.n_components == 20
    assert cfg.n_states == 8
    assert cfg.n_terminal_states is None
    assert cfg.terminal_method == "stability"
    assert cfg.predict_initial_states is False
    assert cfg.n_initial_states == 1
    assert cfg.max_cells is None
    assert cfg.n_jobs == 1
    assert cfg.seed == 1337


def test_trajectory_config_has_cellrank():
    tc = TrajectoryConfig()
    assert isinstance(tc.cellrank, CellRankConfig)


def test_cellrank_config_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        CellRankConfig(bogus_field=123)
