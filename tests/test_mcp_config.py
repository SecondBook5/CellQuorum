"""Config tests for the multicellular_programs (DIALOGUE) stage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.stages.comparative.multicellular_programs.config import MulticellularProgramsConfig
from cellquorum.config.models import CellQuorumConfig


def test_defaults():
    cfg = MulticellularProgramsConfig()
    assert cfg.enabled is True
    assert cfg.method == "dialogue"
    assert cfg.use_rep == "X_pca"
    assert cfg.n_programs == 5
    assert cfg.n_program_genes == 200
    assert cfg.min_cell_types == 2
    assert cfg.min_samples == 4
    assert cfg.stability_resamples == 5
    assert cfg.donor_support_min == 2
    assert cfg.timeout_seconds == 7200
    assert cfg.confounders == []


def test_extra_forbidden():
    with pytest.raises(ValidationError):
        MulticellularProgramsConfig(not_a_field=1)


def test_wired_into_root_config():
    root = CellQuorumConfig()
    assert root.stages.multicellular_programs is True
    assert isinstance(root.multicellular_programs, MulticellularProgramsConfig)
    assert root.multicellular_programs.method == "dialogue"
