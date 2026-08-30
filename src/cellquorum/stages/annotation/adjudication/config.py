"""Configuration for CellQuorum adjudication."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AdjudicationConfig(StrictBaseModel):
    """
    Store adjudication-stage settings.

    Args:
        enabled: Whether adjudication may run.
        cluster_key: obs column containing candidate cluster labels.
        donor_key: obs column containing donor identifiers. None uses design.donor_col.
        condition_key: obs column containing condition labels. None uses design.condition_col.
        marker_support_key: Optional obs column with per-cell marker/annotation support.
        technical_score_key: Optional obs column with per-cell technical artifact score.
        technical_flag_key: Optional obs column with boolean technical flags.
        output_prefix: Prefix used for output artifact filenames.
    """

    # Store whether adjudication may run.
    enabled: bool = True

    # Store the cluster obs column.
    cluster_key: str = "leiden"

    # Store donor/condition columns. None means inherit from design config.
    donor_key: str | None = None
    condition_key: str | None = None

    # Store optional evidence columns.
    marker_support_key: str | None = None
    technical_score_key: str | None = None
    technical_flag_key: str | None = "predicted_doublet"

    # Store output filename prefix.
    output_prefix: str = "adjudication"


__all__ = ["AdjudicationConfig"]
