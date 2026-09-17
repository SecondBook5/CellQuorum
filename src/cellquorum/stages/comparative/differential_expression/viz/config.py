"""Configuration for the differential-expression visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DeVizConfig(StrictBaseModel):
    """DE-visualization stage configuration: volcano(s) and/or the cross-cluster heatmap.

    All biology comes from the DE CSVs and the ``config.design`` bridge; this
    config carries only rendering controls, thresholds, and method selection.

    Args:
        methods: DE-viz method registry entries (registry keys under
            stage_category 'de_viz', e.g. volcano_viz, de_heatmap). Empty
            (the default) falls back to volcano_viz alone, for backward
            compatibility with configs written before de_heatmap existed.
    """

    enabled: bool = True
    fc_cut: float = 1.0  # log2 fold-change cut for significance
    fdr_cut: float = 0.05
    top_n_labels: int = 40
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    x_label: str | None = None  # default derived from case/control at render time
    case_color: str | None = None  # None → figstyle.LE_RED
    control_color: str | None = None  # None → figstyle.NORMAL_BLUE
    methods: list[dict] = []


__all__ = ["DeVizConfig"]
