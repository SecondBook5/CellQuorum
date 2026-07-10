"""Resolve a stage's config sub-block from a pipeline context.

The real ``PipelineContext.config`` is a ``CellQuorumConfig`` pydantic object, so
a stage reads its own settings via ``getattr(config, stage_name).model_dump()``.
Unit tests, however, often pass a plain dict context. This helper accepts either
and always returns a plain dict, so ``MethodDispatchStage`` subclasses work in
both settings without branching.
"""

from __future__ import annotations


def resolve_stage_config(context: object, stage_name: str) -> dict:
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


__all__ = ["resolve_stage_config"]
