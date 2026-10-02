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


def test_integration_rejects_non_positive_max_iter_harmony():
    """max_iter_harmony=0 used to be accepted, and harmonypy does not raise for it --
    it runs the KMeans initialization, stops before any correction iteration, and still
    returns a result, which reads as a successfully integrated embedding that never
    actually corrected anything."""
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            IntegrationConfig(max_iter_harmony=bad)


def test_integration_rejects_non_positive_n_latent():
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            IntegrationConfig(n_latent=bad)


def test_integration_rejects_non_positive_max_epochs():
    with pytest.raises(ValidationError):
        IntegrationConfig(max_epochs=0)
    with pytest.raises(ValidationError):
        IntegrationConfig(max_epochs=-5)


def test_integration_max_epochs_none_is_still_allowed():
    assert IntegrationConfig(max_epochs=None).max_epochs is None


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
