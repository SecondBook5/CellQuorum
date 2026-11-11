"""Resolve study and stage settings for analysis methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


def resolve_stage_config(context: PipelineContext, stage_name: str) -> dict:
    """
    Return a stage's config sub-block as a plain dict.

    Args:
        context: Pipeline context exposing ``config`` (a CellQuorumConfig
            pydantic object, a dict, or None).
        stage_name: Stage/attribute name whose sub-block to read.

    Returns:
        The stage's config as a dict, or an empty dict when absent.
    """

    # Pull the config off the context; missing config means empty settings.
    config = getattr(context, "config", None)
    if config is None:
        return {}

    # Dict-style context (unit tests): index by stage name.
    if isinstance(config, dict):
        sub = config.get(stage_name, {})
        return dict(sub) if isinstance(sub, dict) else {}

    # Pydantic-style config: read the attribute and dump to a dict.
    sub = getattr(config, stage_name, None)
    if sub is None:
        return {}
    if hasattr(sub, "model_dump"):
        return sub.model_dump()
    if isinstance(sub, dict):
        return dict(sub)
    return {}


def with_study_contrast(context: PipelineContext, stage_config: dict) -> dict:
    """Fill missing comparison settings from the cohort and primary study design."""
    from cellquorum.config.cohort import resolve_design_columns

    settings = dict(stage_config)
    config = getattr(context, "config", None)
    design = getattr(config, "design", None)
    if design is None:
        return settings

    donor, condition = resolve_design_columns(config, donor_key=None, condition_key=None)
    for key, value in (
        ("donor_col", donor),
        ("condition_col", condition),
        ("case", getattr(design, "case", None)),
        ("control", getattr(design, "control", None)),
    ):
        if not settings.get(key):
            settings[key] = value
    settings.setdefault("paired", bool(getattr(design, "paired", False)))
    return settings


__all__ = ["resolve_stage_config", "with_study_contrast"]
