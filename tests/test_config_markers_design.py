"""Tests for the markers + design/contrasts config surface."""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.config.design import (
    Contrast,
    ContrastsConfig,
    DesignConfig,
    validate_design_against_obs,
)
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


def test_validate_design_against_obs_accepts_replicated_unpaired_contrast():
    obs = pd.DataFrame(
        {
            "patient_id": ["d1", "d2", "d3", "d4"],
            "condition": ["case", "case", "control", "control"],
        }
    )
    design = DesignConfig(case="case", control="control")

    result = validate_design_against_obs(obs, design=design)

    assert result.case_donors == {"d1", "d2"}
    assert result.control_donors == {"d3", "d4"}
    assert result.paired is False


def test_validate_design_against_obs_rejects_missing_metadata_column():
    obs = pd.DataFrame({"condition": ["case", "control"]})
    design = DesignConfig(case="case", control="control")

    with pytest.raises(CellQuorumConfigError, match="patient_id"):
        validate_design_against_obs(obs, design=design)


def test_validate_design_against_obs_rejects_under_replicated_arm():
    obs = pd.DataFrame(
        {
            "patient_id": ["d1", "d2", "d3"],
            "condition": ["case", "control", "control"],
        }
    )
    design = DesignConfig(case="case", control="control")

    with pytest.raises(CellQuorumConfigError, match="donor replication"):
        validate_design_against_obs(obs, design=design)


def test_validate_design_against_obs_rejects_incomplete_paired_contrast():
    obs = pd.DataFrame(
        {
            "patient_id": ["d1", "d1", "d2", "d3"],
            "condition": ["case", "control", "case", "control"],
        }
    )
    design = DesignConfig(case="case", control="control", paired=True)

    with pytest.raises(CellQuorumConfigError, match="incomplete donor pairs"):
        validate_design_against_obs(obs, design=design, min_donors_per_arm=1)
