"""CellQuorum visualization module."""

from __future__ import annotations

from cellquorum.visualization.style import (
    BACKGROUND_BLUE,
    BACKGROUND_GRAY,
    BACKGROUND_RED,
    CELLQUORUM_BLUE,
    CELLQUORUM_CLUSTER_COLORS,
    CELLQUORUM_FIGSIZE_SMALL,
    CELLQUORUM_FIGSIZE_SQUARE,
    CELLQUORUM_FIGSIZE_WIDE,
    CELLQUORUM_GRAY,
    CELLQUORUM_RED,
    add_dashed_reference_lines,
    add_directional_arrows,
    add_statistical_annotation_box,
    add_volcano_background_panels,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    get_cellquorum_colors,
    save_cellquorum_figure,
)

__all__ = [
    "CELLQUORUM_BLUE",
    "CELLQUORUM_RED",
    "CELLQUORUM_GRAY",
    "BACKGROUND_BLUE",
    "BACKGROUND_GRAY",
    "BACKGROUND_RED",
    "CELLQUORUM_CLUSTER_COLORS",
    "CELLQUORUM_FIGSIZE_SMALL",
    "CELLQUORUM_FIGSIZE_WIDE",
    "CELLQUORUM_FIGSIZE_SQUARE",
    "apply_cellquorum_theme",
    "apply_cellquorum_axis_style",
    "save_cellquorum_figure",
    "get_cellquorum_colors",
    "add_volcano_background_panels",
    "add_dashed_reference_lines",
    "add_directional_arrows",
    "add_statistical_annotation_box",
]
