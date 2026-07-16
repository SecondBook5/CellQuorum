"""Cohort schema — the dataset's structural keys, declared once.

Many stages need the same handful of ``obs`` column names: which column is the
sample, the donor, the condition, the batch, and which lineage is the analysis
focus. Historically each stage carried its own copy of these keys (le_kc.yaml
repeats ``sample_key``/``donor_key``/``condition_key``/``batch_key`` across three
blocks), and domain values like the condition labels ``Normal``/``Lymphedema``
were baked into code. ``CohortConfig`` centralizes them so a dataset declares
its structure once and every stage resolves from it.

All fields are optional. When a cohort field is unset, stages fall back to their
own per-stage key (see :func:`resolve_cohort_key`), so adding a cohort block is
purely additive and existing configs behave identically.
"""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class FocusLineageConfig(StrictBaseModel):
    """
    The analysis focus lineage, declared generically (not KC-specific).

    Args:
        label_key: obs column holding lineage/cell-type labels.
        labels: labels that define the focus lineage (empty = all cells).
    """

    label_key: str | None = None
    labels: list[str] = []


class CohortConfig(StrictBaseModel):
    """
    Structural cohort keys shared across stages.

    Args:
        sample_key: obs column identifying a library/sample.
        donor_key: obs column identifying a donor/patient/biological replicate.
        condition_key: obs column holding the condition/treatment label.
        batch_key: obs column identifying the technical batch to integrate over.
        condition_levels: the condition labels present in the cohort, in the
            order they should be reported (e.g. control first). Empty = infer.
        focus: the analysis focus lineage (generic replacement for the
            KC-specific "keratinocyte" focus).
    """

    sample_key: str | None = None
    donor_key: str | None = None
    condition_key: str | None = None
    batch_key: str | None = None
    condition_levels: list[str] = []
    focus: FocusLineageConfig = FocusLineageConfig()


def resolve_cohort_key(
    config: object,
    *,
    attr: str,
    stage_value: str | None,
) -> str | None:
    """
    Resolve a structural key, preferring the cohort block over a per-stage value.

    This lets stages read a dataset-wide key without forcing every config to
    declare a cohort block: cohort value wins when set, otherwise the stage's
    own value is used.

    Args:
        config: A CellQuorumConfig (or anything exposing ``.cohort``), or None.
        attr: CohortConfig attribute name (e.g. ``batch_key``, ``donor_key``).
        stage_value: The stage's own value for this key (the fallback).

    Returns:
        The cohort value when set, else ``stage_value``.
    """

    cohort = getattr(config, "cohort", None)
    if cohort is not None:
        cohort_value = getattr(cohort, attr, None)
        if cohort_value:
            return cohort_value
    return stage_value


def validate_cohort_against_obs(obs_columns: list[str], *, cohort: CohortConfig) -> list[str]:
    """
    Return warnings for cohort keys that are declared but absent from obs.

    This is intentionally warn-not-raise: a cohort block may declare keys used
    only by some stages, and a given dataset need not carry every one. Stages
    that hard-require a key still validate it themselves.

    Args:
        obs_columns: The AnnData ``obs`` column names.
        cohort: The cohort configuration.

    Returns:
        Human-readable warnings for each declared-but-missing key.
    """

    present = set(obs_columns)
    warnings: list[str] = []
    for attr in ("sample_key", "donor_key", "condition_key", "batch_key"):
        value = getattr(cohort, attr)
        if value and value not in present:
            warnings.append(f"cohort.{attr} '{value}' is not present in obs columns.")
    if cohort.focus.label_key and cohort.focus.label_key not in present:
        warnings.append(
            f"cohort.focus.label_key '{cohort.focus.label_key}' is not present in obs columns."
        )
    return warnings


__all__ = [
    "CohortConfig",
    "FocusLineageConfig",
    "resolve_cohort_key",
    "validate_cohort_against_obs",
]
