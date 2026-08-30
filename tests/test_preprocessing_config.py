"""Tests for preprocessing configuration models."""

import pytest

from cellquorum.config.models import CellQuorumConfig
from cellquorum.stages.preprocessing.config import (
    NormalizationConfig,
    PreprocessingConfig,
    validate_preprocessing_config_dict,
)


def test_normalization_config_defaults():
    """Test default NormalizationConfig values."""
    config = NormalizationConfig()

    assert config.enabled is True
    assert config.recipe == "cellquorum_pf_log1p_pf_v1"
    assert config.input_layer is None
    assert config.output_layer == "cellquorum_normalized"
    assert config.preserve_counts_layer == "counts"
    assert config.target_sum == 10000.0
    assert config.pseudocount == 1.0
    assert config.overwrite is False


def test_normalization_config_valid_recipes():
    """Test that all documented recipes are accepted."""
    valid_recipes = [
        "none",
        "cellquorum_pf_v1",
        "cellquorum_log1p_cp10k_v1",
        "cellquorum_log1p_pf_v1",
        "cellquorum_pf_log1p_pf_v1",
    ]

    for recipe in valid_recipes:
        config = NormalizationConfig(recipe=recipe)
        assert config.recipe == recipe


def test_normalization_config_invalid_recipe():
    """Test that invalid recipes are rejected."""
    with pytest.raises(ValueError, match="Input should be"):
        NormalizationConfig(recipe="unknown_recipe")


def test_normalization_config_invalid_target_sum():
    """Test that invalid target_sum values are rejected."""
    # Reject zero.
    with pytest.raises(ValueError, match="target_sum must be positive"):
        NormalizationConfig(target_sum=0.0)

    # Reject negative values.
    with pytest.raises(ValueError, match="target_sum must be positive"):
        NormalizationConfig(target_sum=-1.0)


def test_normalization_config_invalid_pseudocount():
    """Test that invalid pseudocount values are rejected."""
    # Reject zero.
    with pytest.raises(ValueError, match="pseudocount must be positive"):
        NormalizationConfig(pseudocount=0.0)

    # Reject negative values.
    with pytest.raises(ValueError, match="pseudocount must be positive"):
        NormalizationConfig(pseudocount=-1.0)


def test_normalization_config_empty_layer_names():
    """Test that empty layer names are rejected."""
    # Reject empty output_layer.
    with pytest.raises(ValueError, match="Layer names cannot be empty"):
        NormalizationConfig(output_layer="")

    # Reject empty preserve_counts_layer.
    with pytest.raises(ValueError, match="Layer names cannot be empty"):
        NormalizationConfig(preserve_counts_layer="")

    # Reject whitespace-only output_layer.
    with pytest.raises(ValueError, match="Layer names cannot be empty"):
        NormalizationConfig(output_layer="   ")


def test_preprocessing_config_defaults():
    """Test default PreprocessingConfig values."""
    config = PreprocessingConfig()

    assert config.enabled is True
    assert isinstance(config.normalization, NormalizationConfig)


def test_preprocessing_config_custom_normalization():
    """Test PreprocessingConfig with custom NormalizationConfig."""
    norm_config = NormalizationConfig(recipe="cellquorum_log1p_cp10k_v1", target_sum=5000.0)
    config = PreprocessingConfig(normalization=norm_config)

    assert config.normalization.recipe == "cellquorum_log1p_cp10k_v1"
    assert config.normalization.target_sum == 5000.0


def test_validate_preprocessing_config_dict():
    """Test validate_preprocessing_config_dict with valid dictionary."""
    config_dict = {
        "enabled": True,
        "normalization": {
            "recipe": "cellquorum_pf_v1",
            "output_layer": "normalized_pf",
        },
    }

    config = validate_preprocessing_config_dict(config_dict)

    assert config.enabled is True
    assert config.normalization.recipe == "cellquorum_pf_v1"
    assert config.normalization.output_layer == "normalized_pf"


def test_validate_preprocessing_config_dict_invalid():
    """Test validate_preprocessing_config_dict with invalid dictionary."""
    config_dict = {
        "enabled": True,
        "normalization": {
            "recipe": "unknown_recipe",
        },
    }

    with pytest.raises(ValueError):
        validate_preprocessing_config_dict(config_dict)


def test_validate_preprocessing_config_dict_unknown_key():
    """Test that unknown keys are rejected by strict config."""
    config_dict = {
        "enabled": True,
        "unknown_field": "should_fail",
    }

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_preprocessing_config_dict(config_dict)


def test_cellquorum_config_includes_preprocessing():
    """Test that top-level CellQuorumConfig includes preprocessing field."""
    config = CellQuorumConfig()

    assert hasattr(config, "preprocessing")
    assert isinstance(config.preprocessing, PreprocessingConfig)


def test_cellquorum_config_preprocessing_from_dict():
    """Test CellQuorumConfig.preprocessing from dictionary."""
    config_dict = {
        "project": {"name": "test_project"},
        "preprocessing": {
            "enabled": True,
            "normalization": {
                "recipe": "cellquorum_log1p_pf_v1",
            },
        },
    }

    config = CellQuorumConfig.model_validate(config_dict)

    assert config.preprocessing.enabled is True
    assert config.preprocessing.normalization.recipe == "cellquorum_log1p_pf_v1"


def test_cellquorum_config_preprocessing_unknown_key_rejection():
    """Test that unknown keys in preprocessing are rejected."""
    config_dict = {
        "project": {"name": "test_project"},
        "preprocessing": {
            "enabled": True,
            "unknown_key": "should_fail",
        },
    }

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CellQuorumConfig.model_validate(config_dict)
