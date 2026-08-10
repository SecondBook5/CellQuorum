"""Configuration for the differential-expression visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class DeVizConfig(StrictBaseModel):
    """Pseudobulk volcano rendered from the DE stage's CSV output.

    All biology comes from the DE CSV and the ``config.design`` bridge; this
    config carries only rendering controls and thresholds.
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


__all__ = ["DeVizConfig"]
