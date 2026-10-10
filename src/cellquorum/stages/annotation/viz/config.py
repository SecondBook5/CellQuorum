"""Configuration for the annotation figure stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class AnnotationVizConfig(StrictBaseModel):
    """Figures that show whether the annotation holds up.

    The marker panel, the label column and the expression layer are the annotation stage's own
    (``annotation.marker_panels``, ``annotation.key_added``, ``annotation.score_layer``), so the
    panel that scored the labels is the panel that audits them and is spelled in one place. The
    fields below only override or style.

    Attributes:
        enabled: Whether the stage runs.
        group_by: obs column to put on the dotplot's columns; None reads ``annotation.key_added``.
        highlight_expected: Put a cream band behind the block each group is expected to light up
            (a group is expected to light up the panel block that carries its name).
        vmax_percentile: Percentile of the plotted means that caps the colour scale.
        title: Optional figure title.
        figure_formats: Output formats.
        dpi: Raster DPI for PNG output.
    """

    enabled: bool = True
    group_by: str | None = None
    highlight_expected: bool = True
    vmax_percentile: float = 95.0
    title: str | None = None
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    methods: list[dict] = []


__all__ = ["AnnotationVizConfig"]
