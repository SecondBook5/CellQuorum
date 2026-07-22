"""Configuration for the annotation_consensus stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AnnotationConsensusConfig(StrictBaseModel):
    """Settings for reconciling per-method label columns into one label."""

    # Whether the annotation_consensus stage may run.
    enabled: bool = True

    # obs columns holding per-method labels to reconcile (row-aligned).
    method_label_keys: list[str] = []

    # Alias map: raw per-method label -> canonical backbone label. Applied to
    # every vote before counting so method vocabularies are comparable. This is
    # where dataset-specific vocabulary lives (kept out of the engine).
    backbone_aliases: dict[str, str] = {}

    # obs column that receives the final consensus label.
    key_added: str = "cell_type"

    # obs column that receives the confidence tier (high|medium|low).
    confidence_key: str = "annotation_confidence"

    # obs column (bool) flagged True where confidence is low.
    needs_review_key: str = "needs_review"

    # Optional obs column (e.g. 'ref_state') whose value is copied into
    # 'cell_type_granular' for cells whose consensus is high-confidence.
    granular_source_key: str | None = None

    # Fraction of non-missing votes that must agree to call a majority.
    min_agree_fraction: float = 0.5

    # Whether unanimous agreement across all non-missing votes is 'high'.
    high_confidence_all: bool = True


__all__ = ["AnnotationConsensusConfig"]
