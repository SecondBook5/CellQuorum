"""Tests for the markers + design/contrasts config surface."""

from __future__ import annotations

import pytest

from cellquorum.config.design import Contrast, ContrastsConfig, DesignConfig
from cellquorum.config.markers import MarkersConfig
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.exceptions import CellQuorumConfigError


def test_markers_panel_lookup():
    m = MarkersConfig(panels={"interest": ["IL33", "KRT14"], "mast": ["TPSAB1"]})
    assert m.panel("interest") == ["IL33", "KRT14"]
    assert set(m.names()) == {"interest", "mast"}


def test_markers_unknown_panel_raises():
    m = MarkersConfig(panels={"interest": ["IL33"]})
    with pytest.raises(CellQuorumConfigError, match="stress"):
        m.panel("stress")


def test_design_defaults_and_fields():
    d = DesignConfig(case="Lymphedema", control="Normal", paired=True)
    assert d.donor_col == "patient_id"
    assert d.condition_col == "condition"
    assert d.case == "Lymphedema"
    assert d.paired is True


def test_contrasts_lookup():
    c = ContrastsConfig(
        contrasts=[
            Contrast(
                name="le_vs_normal", case="Lymphedema", control="Normal", paired=True, min_donors=8
            )
        ]
    )
    got = c.get("le_vs_normal")
    assert got.case == "Lymphedema" and got.min_donors == 8


def test_contrasts_unknown_raises():
    c = ContrastsConfig(contrasts=[])
    with pytest.raises(CellQuorumConfigError, match="nope"):
        c.get("nope")


def test_top_level_config_has_new_blocks():
    cfg = CellQuorumConfig()
    assert isinstance(cfg.markers, MarkersConfig)
    assert isinstance(cfg.design, DesignConfig)
    assert isinstance(cfg.contrasts, ContrastsConfig)
