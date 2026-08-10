"""Configuration for the trajectory-visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class TrajectoryVizConfig(StrictBaseModel):
    """Publication figures rendered from the trajectory producers' outputs.

    All biology comes from producer-written keys/files; this config carries only
    rendering controls and optional name filters (user-supplied, never defaulted
    to any biological value).
    """

    enabled: bool = True
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    top_k: int = 15
    embedding_basis: str | None = None
    pseudotime_keys: list[str] | None = None
    lineages: list[str] | None = None
    genes: list[str] | None = None
    cluster_key: str | None = None


__all__ = ["TrajectoryVizConfig"]
