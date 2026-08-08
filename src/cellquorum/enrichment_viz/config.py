"""Configuration for the enrichment-visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class EnrichmentVizConfig(StrictBaseModel):
    """Publication figures rendered from the enrichment stage's CSV outputs.

    All biology comes from the input CSVs; this config carries only rendering
    controls and optional collection/resource filters (by name, supplied by the
    user — never defaulted to any biological value).

    Attributes:
        enabled: Whether the stage runs.
        top_k: Top-K up + top-K down sources per diverging figure.
        figure_formats: File formats written per figure.
        dpi: Raster resolution.
        collections: If set, only render these GSEA/ORA/GSVA collections; None → all present.
        resources: If set, only render these activity resources; None → all present.
    """

    enabled: bool = True
    top_k: int = 12
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    collections: list[str] | None = None
    resources: list[str] | None = None


__all__ = ["EnrichmentVizConfig"]
