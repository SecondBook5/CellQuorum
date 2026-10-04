"""Configuration for the state-scoring visualization stage."""

from __future__ import annotations

from cellquorum.config.base import StrictBaseModel


class StateScoringVizConfig(StrictBaseModel):
    """Quantitative companion to state_score_umap: violin/box + a donor-level test
    per cluster (donors and pairing from config.design), for each scored program,
    control vs case.

    All biology comes from state_scoring's obsm output and config.design;
    this config carries only rendering controls and program/method selection.

    Attributes:
        enabled: Whether the stage runs.
        score_source: Which scoring method's output to read ("aucell" or "ucell").
        programs: Subset of scored program names to render (empty -> all scored).
        cluster_key: obs column with cluster/cell-type labels.
        cluster_labels: Optional display-only {raw code -> name} override.
        condition_col: obs column with the case/control condition label
            (falls back to config.design.condition_col when unset).
        case: Case condition token (falls back to config.design.case).
        control: Control condition token (falls back to config.design.control).
        figure_formats: Output formats to save each figure in.
        dpi: Raster DPI for PNG output.
    """

    enabled: bool = True
    score_source: str = "aucell"
    programs: list[str] = []
    cluster_key: str | None = None
    cluster_labels: dict[str, str] | None = None
    condition_col: str | None = None
    case: str | None = None
    control: str | None = None
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    methods: list[dict] = []


__all__ = ["StateScoringVizConfig"]
