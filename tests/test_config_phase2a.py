"""Tests for the Phase-2A config additions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import (
    CellQuorumConfig,
    ClusteringConfig,
    DimensionalityConfig,
)


def test_dimensionality_defaults():
    c = DimensionalityConfig()
    assert c.enabled is True
    assert c.method == "pca"
    assert c.n_pcs == "auto"
    assert c.max_pcs == 50


def test_dimensionality_accepts_int_n_pcs():
    assert DimensionalityConfig(n_pcs=30).n_pcs == 30


def test_dimensionality_rejects_non_positive_max_pcs():
    """max_pcs=0 used to be accepted, then produced a component-budget error deep in
    PCA that blamed too-few-fit-cells/too-few-HVG-genes -- the wrong cause entirely."""
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            DimensionalityConfig(max_pcs=bad)


def test_dimensionality_rejects_unrecognized_n_pcs_string():
    """Only "auto" is a valid string; anything else (e.g. a typo) used to be accepted
    and only failed later with int("atuo"), a confusing crash with no config context."""
    with pytest.raises(ValidationError):
        DimensionalityConfig(n_pcs="atuo")


def test_dimensionality_rejects_non_positive_int_n_pcs():
    with pytest.raises(ValidationError):
        DimensionalityConfig(n_pcs=0)


def test_clustering_defaults():
    c = ClusteringConfig()
    assert c.method == "leiden"
    assert c.n_neighbors == 15
    assert c.resolution == 1.0
    assert c.key_added == "leiden"


def test_strict_rejects_unknown_field():
    with pytest.raises(ValidationError):
        DimensionalityConfig(bogus=1)


def test_top_level_config_has_new_stage_blocks_and_flags():
    cfg = CellQuorumConfig()
    assert cfg.stages.dimensionality is True
    assert cfg.stages.clustering is True
    assert isinstance(cfg.dimensionality, DimensionalityConfig)
    assert isinstance(cfg.clustering, ClusteringConfig)
