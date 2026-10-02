"""Validated configuration models for preprocessing stages."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

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
        scclr_target: PFlog1pPF depth target passed to scclr for the
            ``cellquorum_pf_log1p_pf_v1`` recipe: ``auto`` (estimate the NB
            overdispersion alpha), ``mean``/``median`` depth, or a fixed numeric
            ``K`` (as a string).
        overwrite: Whether existing output layers may be overwritten.
    """

    enabled: bool = True

    recipe: Literal[
        "none",
        "cellquorum_pf_v1",
        "cellquorum_log1p_cp10k_v1",
        "cellquorum_log1p_pf_v1",
        "cellquorum_pf_log1p_pf_v1",
    ] = "cellquorum_pf_log1p_pf_v1"

    input_layer: str | None = None

    output_layer: str = "cellquorum_normalized"

    preserve_counts_layer: str = "counts"

    target_sum: float = 10000.0

    pseudocount: float = 1.0

    scclr_target: str = "auto"

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

        if not cleaned:
            raise ValueError("Layer names cannot be empty.")

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
        write_figures: Whether to write the normalization diagnostic figures.
        figure_format: Figure file format (png, pdf, svg).
        figure_dpi: Figure resolution in dots per inch.
    """

    enabled: bool = True

    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)

    write_figures: bool = True

    figure_format: str = "png"

    figure_dpi: int = 300


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
