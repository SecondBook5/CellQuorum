"""Tests for MetacellsConfig defaults and strictness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.stages.metacells.config import MetacellsConfig


def test_defaults() -> None:
    cfg = MetacellsConfig()
    assert cfg.enabled is False
    assert cfg.method == "metacells"
    assert cfg.rep_key == "X_pca_harmony"
    assert cfg.layer == "counts"
    assert cfg.group_by == "ref_cell_type_granular_coarse"
    assert cfg.target_metacell_size == 96
    assert cfg.min_cells == 5000
    assert cfg.marker_genes == []
    assert cfg.seed == 0
    assert cfg.env_name == "metacells_env"
    assert cfg.launcher == "micromamba"
    assert cfg.timeout_seconds == 3600


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MetacellsConfig(not_a_real_field=1)


def test_marker_genes_overridable() -> None:
    cfg = MetacellsConfig(marker_genes=["PECAM1", "PDGFRB"])
    assert cfg.marker_genes == ["PECAM1", "PDGFRB"]
