"""Feature-family annotation utilities for CellQuorum QC."""

from __future__ import annotations

import re
from dataclasses import dataclass

import anndata as ad
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.config import QCFeaturePatternConfig

MITO_COLUMN = "cellquorum_is_mito"


RIBO_COLUMN = "cellquorum_is_ribo"


HEMOGLOBIN_COLUMN = "cellquorum_is_hemoglobin"


CUSTOM_EXCLUDE_COLUMN = "cellquorum_is_custom_exclude"


class QCFeatureAnnotationError(CellQuorumDataError):
    """Report feature-family annotation failures."""


@dataclass(frozen=True)
class QCFeatureMaskSummary:
    """Store a structured summary of QC feature masks.

    Args:
        n_vars: Number of variables in the feature mask table.
        n_mito: Number of mitochondrial features.
        n_ribo: Number of ribosomal features.
        n_hemoglobin: Number of hemoglobin features.
        n_custom_exclude: Number of custom-excluded features.
    """

    n_vars: int
    n_mito: int
    n_ribo: int
    n_hemoglobin: int
    n_custom_exclude: int

    def to_dict(self) -> dict[str, int]:
        """Convert the feature-mask summary to a JSON-friendly dictionary.

        Returns:
            Dictionary representation of feature-mask counts.
        """

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
    """Build QC feature-family masks from AnnData variable names.

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

    feature_config = QCFeaturePatternConfig() if config is None else config

    if not isinstance(feature_config, QCFeaturePatternConfig):
        raise QCFeatureAnnotationError(
            "build_feature_masks expected config to be a QCFeaturePatternConfig object. "
            f"Received: {type(feature_config).__name__}."
        )

    if not isinstance(adata, ad.AnnData):
        raise QCFeatureAnnotationError(
            "build_feature_masks expected an AnnData object. " f"Received: {type(adata).__name__}."
        )

    if adata.n_vars <= 0:
        raise QCFeatureAnnotationError("Cannot build QC feature masks for zero variables.")

    feature_names = pd.Index(adata.var_names.astype(str))

    mito_mask = _match_prefixes(feature_names, feature_config.mitochondrial_prefixes)

    ribo_mask = _match_prefixes(feature_names, feature_config.ribosomal_prefixes)

    hemoglobin_mask = _match_regexes(feature_names, feature_config.hemoglobin_regexes)

    custom_exclude_mask = _match_prefixes(
        feature_names,
        feature_config.custom_exclude_prefixes,
    )

    masks = pd.DataFrame(
        {
            MITO_COLUMN: mito_mask.to_numpy(dtype=bool),
            RIBO_COLUMN: ribo_mask.to_numpy(dtype=bool),
            HEMOGLOBIN_COLUMN: hemoglobin_mask.to_numpy(dtype=bool),
            CUSTOM_EXCLUDE_COLUMN: custom_exclude_mask.to_numpy(dtype=bool),
        },
        index=adata.var_names,
    )

    return masks


def annotate_qc_feature_masks(
    adata: ad.AnnData,
    config: QCFeaturePatternConfig | None = None,
    *,
    copy: bool = False,
) -> ad.AnnData:
    """Annotate AnnData.var with CellQuorum QC feature masks.

    Args:
        adata: AnnData object to annotate.
        config: Optional feature-pattern configuration.
        copy: Whether to annotate a copy instead of mutating the input object.

    Returns:
        AnnData object containing QC feature-mask columns in `.var`.

    Raises:
        QCFeatureAnnotationError: If feature-mask construction fails.
    """

    target = adata.copy() if copy else adata

    masks = build_feature_masks(target, config)

    for column in masks.columns:
        target.var[column] = masks[column].to_numpy(dtype=bool)

    return target


def summarize_feature_masks(masks: pd.DataFrame) -> QCFeatureMaskSummary:
    """Summarize a QC feature-mask table.

    Args:
        masks: Feature-mask DataFrame produced by build_feature_masks.

    Returns:
        Structured feature-mask summary.

    Raises:
        QCFeatureAnnotationError: If required mask columns are missing.
    """

    if not isinstance(masks, pd.DataFrame):
        raise QCFeatureAnnotationError(
            "summarize_feature_masks expected a pandas DataFrame. "
            f"Received: {type(masks).__name__}."
        )

    required_columns = [
        MITO_COLUMN,
        RIBO_COLUMN,
        HEMOGLOBIN_COLUMN,
        CUSTOM_EXCLUDE_COLUMN,
    ]

    missing_columns = [column for column in required_columns if column not in masks.columns]

    if missing_columns:
        raise QCFeatureAnnotationError(
            "Feature-mask table is missing required column(s): " f"{', '.join(missing_columns)}."
        )

    return QCFeatureMaskSummary(
        n_vars=int(masks.shape[0]),
        n_mito=int(masks[MITO_COLUMN].sum()),
        n_ribo=int(masks[RIBO_COLUMN].sum()),
        n_hemoglobin=int(masks[HEMOGLOBIN_COLUMN].sum()),
        n_custom_exclude=int(masks[CUSTOM_EXCLUDE_COLUMN].sum()),
    )


def _match_prefixes(feature_names: pd.Index, prefixes: list[str]) -> pd.Series:
    """Match feature names against one or more prefixes.

    Args:
        feature_names: Feature names to match.
        prefixes: Prefixes used for startswith matching.

    Returns:
        Boolean Series indexed like feature_names.
    """

    if not prefixes:
        return pd.Series(False, index=feature_names, dtype=bool)

    prefix_tuple = tuple(prefixes)

    return feature_names.to_series(index=feature_names).str.startswith(prefix_tuple)


def _match_regexes(feature_names: pd.Index, regexes: list[str]) -> pd.Series:
    """Match feature names against one or more regular expressions.

    Args:
        feature_names: Feature names to match.
        regexes: Regular expressions used for feature-family matching.

    Returns:
        Boolean Series indexed like feature_names.

    Raises:
        QCFeatureAnnotationError: If a regex is invalid.
    """

    if not regexes:
        return pd.Series(False, index=feature_names, dtype=bool)

    combined_mask = pd.Series(False, index=feature_names, dtype=bool)

    feature_series = feature_names.to_series(index=feature_names)

    for regex in regexes:
        try:
            re.compile(regex)

        except re.error as error:
            raise QCFeatureAnnotationError(
                f"Invalid QC feature regex '{regex}': {error}"
            ) from error

        combined_mask = combined_mask | feature_series.str.contains(regex, regex=True, na=False)

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
