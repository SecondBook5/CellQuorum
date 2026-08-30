"""Tests for IntegrationConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig
from cellquorum.stages.integration.config import IntegrationConfig


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


def test_integration_accepts_methods_list():
    """Test that IntegrationConfig accepts a methods list via pydantic validation."""
    config = CellQuorumConfig.model_validate(
        {
            "integration": {
                "methods": [
                    {"method": "harmony", "output_rep": "X_pca_harmony"},
                    {"method": "scvi", "output_rep": "X_scvi"},
                ]
            }
        }
    )
    assert config.integration.methods == [
        {"method": "harmony", "output_rep": "X_pca_harmony"},
        {"method": "scvi", "output_rep": "X_scvi"},
    ]


def test_integration_scalar_method_has_empty_methods_list():
    """Test that scalar method configs have an empty methods list by default."""
    config = CellQuorumConfig.model_validate({"integration": {"method": "harmony"}})
    assert config.integration.method == "harmony"
    assert config.integration.methods == []
