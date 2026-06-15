"""Validated configuration models for preprocessing stages."""

from __future__ import annotations

# Import Literal for constrained recipe options.
from typing import Literal

# Import Pydantic primitives for strict runtime validation.
from pydantic import Field, field_validator

# Import the shared strict base model used by CellQuorum configuration models.
from cellquorum.config.base import StrictBaseModel


class NormalizationConfig(StrictBaseModel):
    """
    Store normalization method configuration.

    Normalization converts raw count matrices into comparable expression values.
    Recipe names are versioned because future method changes should be explicit.

    Args:
        enabled: Whether normalization should run.
        recipe: Normalization transformation recipe.
        input_layer: Optional input layer. None means X.
        output_layer: Layer where normalized values are written.
        preserve_counts_layer: Layer where raw counts are preserved.
        target_sum: Target total count per cell for scaling recipes.
        pseudocount: Pseudocount for log-family recipes.
        overwrite: Whether existing output layers may be overwritten.
    """

    # Store whether normalization is enabled.
    enabled: bool = True

    # Store the normalization transformation recipe.
    recipe: Literal[
        "none",
        "cellquorum_pf_v1",
        "cellquorum_log1p_cp10k_v1",
        "cellquorum_log1p_pf_v1",
        "cellquorum_pf_log1p_pf_v1",
    ] = "cellquorum_pf_log1p_pf_v1"

    # Store the optional input layer name.
    input_layer: str | None = None

    # Store the output layer name.
    output_layer: str = "cellquorum_normalized"

    # Store the raw-counts preservation layer name.
    preserve_counts_layer: str = "counts"

    # Store the target sum for scaling recipes.
    target_sum: float = 10000.0

    # Store the pseudocount for log-family recipes.
    pseudocount: float = 1.0

    # Store whether existing output layers may be overwritten.
    overwrite: bool = False

    @field_validator("output_layer", "preserve_counts_layer")
    @classmethod
    def validate_layer_names(cls, value: str) -> str:
        """
        Validate layer name strings.

        Args:
            value: Candidate layer name.

        Returns:
            Cleaned layer name.

        Raises:
            ValueError: If the layer name is empty.
        """

        # Strip surrounding whitespace.
        cleaned = value.strip()

        # Reject empty layer names.
        if not cleaned:
            raise ValueError("Layer names cannot be empty.")

        # Return the cleaned layer name.
        return cleaned

    @field_validator("target_sum")
    @classmethod
    def validate_target_sum(cls, value: float) -> float:
        """
        Validate the target sum parameter.

        Args:
            value: Candidate target sum.

        Returns:
            Validated target sum.

        Raises:
            ValueError: If the target sum is not positive.
        """

        # Reject non-positive target sums.
        if value <= 0:
            raise ValueError("target_sum must be positive.")

        # Return the validated target sum.
        return value

    @field_validator("pseudocount")
    @classmethod
    def validate_pseudocount(cls, value: float) -> float:
        """
        Validate the pseudocount parameter.

        Args:
            value: Candidate pseudocount.

        Returns:
            Validated pseudocount.

        Raises:
            ValueError: If the pseudocount is not positive.
        """

        # Reject non-positive pseudocounts.
        if value <= 0:
            raise ValueError("pseudocount must be positive.")

        # Return the validated pseudocount.
        return value


class PreprocessingConfig(StrictBaseModel):
    """
    Store preprocessing stage configuration.

    Preprocessing transforms raw count matrices into analysis-ready expression
    values. It includes normalization and will grow to include scaling, feature
    selection, and doublet detection as those are implemented.

    Args:
        enabled: Whether preprocessing should run.
        normalization: Normalization method configuration.
    """

    # Store whether preprocessing is enabled.
    enabled: bool = True

    # Store normalization configuration.
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


def validate_preprocessing_config_dict(config_dict: dict) -> PreprocessingConfig:
    """
    Validate a preprocessing configuration dictionary.

    Args:
        config_dict: Raw preprocessing configuration dictionary.

    Returns:
        Validated PreprocessingConfig.

    Raises:
        ValueError: If the dictionary contains invalid values.
    """

    # Validate through Pydantic coercion.
    return PreprocessingConfig.model_validate(config_dict)


__all__ = [
    "NormalizationConfig",
    "PreprocessingConfig",
    "validate_preprocessing_config_dict",
]
