"""Tests for IntegrationConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig
from cellquorum.integration.config import IntegrationConfig


def test_integration_defaults():
    c = IntegrationConfig()
    assert c.method == "harmony"
    assert c.batch_key == "patient_id"
    assert c.input_rep == "X_pca"
    assert c.output_rep == "X_pca_harmony"


def test_integration_strict():
    with pytest.raises(ValidationError):
        IntegrationConfig(bogus=1)


def test_top_level_has_integration():
    assert isinstance(CellQuorumConfig().integration, IntegrationConfig)
