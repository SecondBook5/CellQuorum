"""Feature-family annotation utilities for CellQuorum QC."""

from __future__ import annotations

# Import regular expressions for hemoglobin and custom feature matching.
import re

# Import dataclass for structured feature-mask summaries.
from dataclasses import dataclass

# Import AnnData for runtime input validation.
import anndata as ad

# Import pandas for feature-mask tables.
import pandas as pd

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import QC feature-pattern configuration.
from cellquorum.stages.qc.config import QCFeaturePatternConfig

# Store the CellQuorum mitochondrial feature-mask column name.
MITO_COLUMN = "cellquorum_is_mito"

# Store the CellQuorum ribosomal feature-mask column name.
RIBO_COLUMN = "cellquorum_is_ribo"

# Store the CellQuorum hemoglobin feature-mask column name.
HEMOGLOBIN_COLUMN = "cellquorum_is_hemoglobin"

# Store the CellQuorum custom-exclude feature-mask column name.
CUSTOM_EXCLUDE_COLUMN = "cellquorum_is_custom_exclude"


class QCFeatureAnnotationError(CellQuorumDataError):
    """
    Report feature-family annotation failures.

    Feature-family annotation is the bridge between gene naming conventions and
    QC metric calculation. Errors here should be explicit because incorrect
    mitochondrial, ribosomal, hemoglobin, or custom feature masks would directly
    corrupt percentage-based QC metrics.
    """


@dataclass(frozen=True)
class QCFeatureMaskSummary:
    """
    Store a structured summary of QC feature masks.

    The summary is useful for tests, stage metrics, reports, and provenance. It
    records how many variables were assigned to each QC feature family.

    Args:
        n_vars: Number of variables in the feature mask table.
        n_mito: Number of mitochondrial features.
        n_ribo: Number of ribosomal features.
        n_hemoglobin: Number of hemoglobin features.
        n_custom_exclude: Number of custom-excluded features.
    """

    # Store the number of variables represented in the mask table.
    n_vars: int

    # Store the number of mitochondrial features.
    n_mito: int

    # Store the number of ribosomal features.
    n_ribo: int

    # Store the number of hemoglobin features.
    n_hemoglobin: int

    # Store the number of custom-excluded features.
    n_custom_exclude: int

    def to_dict(self) -> dict[str, int]:
        """
        Convert the feature-mask summary to a JSON-friendly dictionary.

        Returns:
            Dictionary representation of feature-mask counts.
        """

        # Return a JSON-friendly dictionary.
        return {
            "n_vars": self.n_vars,
            "n_mito": self.n_mito,
            "n_ribo": self.n_ribo,
            "n_hemoglobin": self.n_hemoglobin,
            "n_custom_exclude": self.n_custom_exclude,
        }


def build_feature_masks(
    adata: ad.AnnData,
    config: QCFeaturePatternConfig | None = None,
) -> pd.DataFrame:
    """
    Build QC feature-family masks from AnnData variable names.

    This function does not mutate the AnnData object. It returns a DataFrame
    indexed by `adata.var_names` with boolean columns for mitochondrial,
    ribosomal, hemoglobin, and custom-excluded features. Human mitochondrial
    genes are typically matched by `MT-`, while mouse mitochondrial genes can be
    matched by configuring `mitochondrial_prefixes=["mt-"]`.

    Args:
        adata: AnnData object containing variables to annotate.
        config: Optional feature-pattern configuration. Defaults to
            QCFeaturePatternConfig().

    Returns:
        DataFrame of boolean QC feature masks.

    Raises:
        QCFeatureAnnotationError: If the AnnData object or feature-pattern config
            is invalid.
    """

    # Resolve the feature-pattern configuration.
    feature_config = QCFeaturePatternConfig() if config is None else config

    # Validate the feature-pattern configuration type.
    if not isinstance(feature_config, QCFeaturePatternConfig):
        raise QCFeatureAnnotationError(
            "build_feature_masks expected config to be a QCFeaturePatternConfig object. "
            f"Received: {type(feature_config).__name__}."
        )

    # Validate the AnnData input type.
    if not isinstance(adata, ad.AnnData):
        raise QCFeatureAnnotationError(
            "build_feature_masks expected an AnnData object. " f"Received: {type(adata).__name__}."
        )

    # Validate that the AnnData object contains variables.
    if adata.n_vars <= 0:
        raise QCFeatureAnnotationError("Cannot build QC feature masks for zero variables.")

    # Convert variable names to a pandas Index for vectorized string matching.
    feature_names = pd.Index(adata.var_names.astype(str))

    # Build the mitochondrial feature mask.
    mito_mask = _match_prefixes(feature_names, feature_config.mitochondrial_prefixes)

    # Build the ribosomal feature mask.
    ribo_mask = _match_prefixes(feature_names, feature_config.ribosomal_prefixes)

    # Build the hemoglobin feature mask.
    hemoglobin_mask = _match_regexes(feature_names, feature_config.hemoglobin_regexes)

    # Build the custom-exclude feature mask.
    custom_exclude_mask = _match_prefixes(
        feature_names,
        feature_config.custom_exclude_prefixes,
    )

    # Build the feature-mask DataFrame.
    masks = pd.DataFrame(
        {
            MITO_COLUMN: mito_mask.to_numpy(dtype=bool),
            RIBO_COLUMN: ribo_mask.to_numpy(dtype=bool),
            HEMOGLOBIN_COLUMN: hemoglobin_mask.to_numpy(dtype=bool),
            CUSTOM_EXCLUDE_COLUMN: custom_exclude_mask.to_numpy(dtype=bool),
        },
        index=adata.var_names,
    )

    # Return the feature-mask DataFrame.
    return masks


def annotate_qc_feature_masks(
    adata: ad.AnnData,
    config: QCFeaturePatternConfig | None = None,
    *,
    copy: bool = False,
) -> ad.AnnData:
    """
    Annotate AnnData.var with CellQuorum QC feature masks.

    Args:
        adata: AnnData object to annotate.
        config: Optional feature-pattern configuration.
        copy: Whether to annotate a copy instead of mutating the input object.

    Returns:
        AnnData object containing QC feature-mask columns in `.var`.

    Raises:
        QCFeatureAnnotationError: If feature-mask construction fails.
    """

    # Create a copy when requested.
    target = adata.copy() if copy else adata

    # Build QC feature masks for the target object.
    masks = build_feature_masks(target, config)

    # Assign each feature-mask column into AnnData.var.
    for column in masks.columns:
        # Store the feature-mask column as a boolean vector.
        target.var[column] = masks[column].to_numpy(dtype=bool)

    # Return the annotated AnnData object.
    return target


def summarize_feature_masks(masks: pd.DataFrame) -> QCFeatureMaskSummary:
    """
    Summarize a QC feature-mask table.

    Args:
        masks: Feature-mask DataFrame produced by build_feature_masks.

    Returns:
        Structured feature-mask summary.

    Raises:
        QCFeatureAnnotationError: If required mask columns are missing.
    """

    # Validate the mask table type.
    if not isinstance(masks, pd.DataFrame):
        raise QCFeatureAnnotationError(
            "summarize_feature_masks expected a pandas DataFrame. "
            f"Received: {type(masks).__name__}."
        )

    # Define required feature-mask columns.
    required_columns = [
        MITO_COLUMN,
        RIBO_COLUMN,
        HEMOGLOBIN_COLUMN,
        CUSTOM_EXCLUDE_COLUMN,
    ]

    # Identify missing feature-mask columns.
    missing_columns = [column for column in required_columns if column not in masks.columns]

    # Raise a clear error if mask columns are missing.
    if missing_columns:
        raise QCFeatureAnnotationError(
            "Feature-mask table is missing required column(s): " f"{', '.join(missing_columns)}."
        )

    # Return the structured mask summary.
    return QCFeatureMaskSummary(
        n_vars=int(masks.shape[0]),
        n_mito=int(masks[MITO_COLUMN].sum()),
        n_ribo=int(masks[RIBO_COLUMN].sum()),
        n_hemoglobin=int(masks[HEMOGLOBIN_COLUMN].sum()),
        n_custom_exclude=int(masks[CUSTOM_EXCLUDE_COLUMN].sum()),
    )


def _match_prefixes(feature_names: pd.Index, prefixes: list[str]) -> pd.Series:
    """
    Match feature names against one or more prefixes.

    Args:
        feature_names: Feature names to match.
        prefixes: Prefixes used for startswith matching.

    Returns:
        Boolean Series indexed like feature_names.
    """

    # Return all-False masks when no prefixes are configured.
    if not prefixes:
        return pd.Series(False, index=feature_names, dtype=bool)

    # Convert prefixes to a tuple for pandas startswith.
    prefix_tuple = tuple(prefixes)

    # Return the prefix match mask.
    return feature_names.to_series(index=feature_names).str.startswith(prefix_tuple)


def _match_regexes(feature_names: pd.Index, regexes: list[str]) -> pd.Series:
    """
    Match feature names against one or more regular expressions.

    Args:
        feature_names: Feature names to match.
        regexes: Regular expressions used for feature-family matching.

    Returns:
        Boolean Series indexed like feature_names.

    Raises:
        QCFeatureAnnotationError: If a regex is invalid.
    """

    # Return all-False masks when no regexes are configured.
    if not regexes:
        return pd.Series(False, index=feature_names, dtype=bool)

    # Initialize the combined regex mask.
    combined_mask = pd.Series(False, index=feature_names, dtype=bool)

    # Convert feature names to a Series once for repeated matching.
    feature_series = feature_names.to_series(index=feature_names)

    # Iterate over configured regexes.
    for regex in regexes:
        # Compile the regex so invalid patterns fail clearly.
        try:
            # Compile the regex pattern.
            re.compile(regex)

        # Convert regex syntax errors into CellQuorum errors.
        except re.error as error:
            raise QCFeatureAnnotationError(
                f"Invalid QC feature regex '{regex}': {error}"
            ) from error

        # Update the combined mask with this regex match.
        combined_mask = combined_mask | feature_series.str.contains(regex, regex=True, na=False)

    # Return the combined regex mask.
    return combined_mask


__all__ = [
    "CUSTOM_EXCLUDE_COLUMN",
    "HEMOGLOBIN_COLUMN",
    "MITO_COLUMN",
    "RIBO_COLUMN",
    "QCFeatureAnnotationError",
    "QCFeatureMaskSummary",
    "annotate_qc_feature_masks",
    "build_feature_masks",
    "summarize_feature_masks",
]
