"""Configuration for the CCC-visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class CccVizConfig(StrictBaseModel):
    """Publication figures rendered from the CCC stages' CSV/uns outputs.

    All biology comes from the input CSVs/uns; this config carries only rendering
    controls and optional name filters (supplied by the user — never a biological default).

    Attributes:
        enabled: Whether the stage runs.
        top_k: Top-N items kept per figure (edges, LR pairs, nodes).
        figure_formats: File formats written per figure.
        dpi: Raster resolution.
        sources: If set, only render these canonical LR sources by name; None -> all present.
        levels: If set, only render these network levels (cci/gci) by name; None -> all present.
    """

    enabled: bool = True
    top_k: int = 15
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    sources: list[str] | None = None
    levels: list[str] | None = None


__all__ = ["CccVizConfig"]
