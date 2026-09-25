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
        cluster_labels: Optional {raw cluster code -> display name} override for every
            figure's cluster axis/node/segment labels (display only -- grouping and
            color still key off the raw code). A code with no entry stays bare;
            unset, every figure shows raw codes. Never defaulted, since the mapping
            is study-specific.
        mnn_key_clusters: Clusters to render a focused MultiNicheNet circos for
            (multinichenet_viz); None -> every cluster seen as a sender or
            receiver in mnn_prioritization.csv, never a biology-driven guess.
        mnn_circos_top_k: Partners kept per focused circos (ribbons around the
            focus cluster) -- deliberately smaller than top_k, since the point
            of focusing is enough headroom to label each ribbon's LR pair.
    """

    enabled: bool = True
    top_k: int = 15
    figure_formats: list[str] = ["pdf", "png"]
    dpi: int = 300
    sources: list[str] | None = None
    levels: list[str] | None = None
    cluster_labels: dict[str, str] | None = None
    mnn_key_clusters: list[str] | None = None
    mnn_circos_top_k: int = 8


__all__ = ["CccVizConfig"]
