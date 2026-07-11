"""
CellQuorum visualization style and theme.

This module encodes the distinctive CellQuorum figure aesthetic: clean, professional,
publication-quality diagnostic figures with consistent visual language.

The style is inspired by AJ Book's signature scRNA-seq figure aesthetic:
- Blue-gray-red directional color palette
- Subtle background panels for significance regions
- Dashed reference lines (not solid)
- Direct labeling, minimal decoration
- Transparent scatter points with proper alpha
- Clean sans-serif typography
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# =============================================================================
# CellQuorum Color Palette
# =============================================================================

# Signature directional colors (from your volcano plots)
CELLQUORUM_BLUE = "#5A8BC4"  # Directional low / downregulated
CELLQUORUM_RED = "#C45A5A"  # Directional high / upregulated
CELLQUORUM_GRAY = "#BDBDBD"  # Neutral / non-significant

# Background panel colors (very subtle, used in volcano plot regions)
BACKGROUND_BLUE = "#E8F0F8"  # Light blue for downregulated region
BACKGROUND_GRAY = "#F5F5F5"  # Light gray for neutral region
BACKGROUND_RED = "#F8E8E8"  # Light red for upregulated region

# Cluster colors (from your UMAP plots - distinct, harmonious palette)
CELLQUORUM_CLUSTER_COLORS = [
    "#E57373",  # Coral red
    "#FFB74D",  # Orange
    "#81C784",  # Green
    "#64B5F6",  # Blue
    "#BA68C8",  # Purple
    "#FFD54F",  # Yellow
    "#4DB6AC",  # Teal
    "#F06292",  # Pink
    "#AED581",  # Light green
    "#4FC3F7",  # Light blue
    "#9575CD",  # Light purple
    "#FFB74D",  # Amber
    "#A1887F",  # Brown
    "#90A4AE",  # Blue gray
    "#EF9A9A",  # Light red
    "#C5E1A5",  # Lime
    "#80DEEA",  # Cyan
    "#CE93D8",  # Light purple
    "#FFCC80",  # Light orange
    "#BCAAA4",  # Light brown
]

# =============================================================================
# Figure Sizing Constants
# =============================================================================

CELLQUORUM_FIGSIZE_SMALL = (6, 4)  # Single panel diagnostic
CELLQUORUM_FIGSIZE_WIDE = (10, 4)  # Wide single row
CELLQUORUM_FIGSIZE_SQUARE = (6, 6)  # Square plots (UMAP, etc)
CELLQUORUM_FIGSIZE_LARGE = (12, 8)  # Multi-panel figure

# =============================================================================
# Theme Setup
# =============================================================================


def apply_cellquorum_theme() -> None:
    """
    Apply the CellQuorum seaborn + matplotlib theme.

    This sets up the base aesthetic that all CellQuorum figures share:
    - Clean white background
    - Minimal grid
    - Sans-serif fonts
    - Thin gray axes
    - High DPI rendering
    """
    # Set seaborn style as base
    sns.set_style(
        "ticks",
        {
            "axes.edgecolor": CELLQUORUM_GRAY,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.color": "#EEEEEE",
            "grid.linewidth": 0.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
        },
    )

    # Set context for scaling
    sns.set_context(
        "notebook",
        font_scale=1.0,
        rc={
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
        },
    )

    # Matplotlib overrides for publication quality
    mpl.rcParams.update(
        {
            "figure.dpi": 100,  # Screen DPI
            "savefig.dpi": 300,  # Save DPI (publication quality)
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "figure.titleweight": "bold",
            "pdf.fonttype": 42,  # TrueType fonts in PDF
            "ps.fonttype": 42,  # TrueType fonts in PS
        }
    )


def apply_cellquorum_axis_style(ax: Axes, remove_top_right: bool = True) -> None:
    """
    Apply CellQuorum styling to a single axis.

    This is the final polish applied after plotting: removes unnecessary spines,
    adjusts tick parameters, and ensures consistency.

    Args:
        ax: Matplotlib axis to style.
        remove_top_right: Whether to remove top and right spines (Tufte style).
    """
    # Remove top and right spines (Tufte-style)
    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Style remaining spines
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_color(CELLQUORUM_GRAY)
            spine.set_linewidth(0.8)

    # Tick styling
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=0.8,
        color=CELLQUORUM_GRAY,
        labelsize=9,
    )

    # Minor ticks if present
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2,
        width=0.6,
        color=CELLQUORUM_GRAY,
    )


def get_cellquorum_colors(n: int | None = None) -> list[str]:
    """
    Get CellQuorum cluster colors.

    Args:
        n: Number of colors to return. If None, returns all colors.
           If n exceeds available colors, cycles through the palette.

    Returns:
        List of hex color codes.
    """
    if n is None:
        return CELLQUORUM_CLUSTER_COLORS.copy()

    # Cycle through colors if n exceeds palette size
    return [CELLQUORUM_CLUSTER_COLORS[i % len(CELLQUORUM_CLUSTER_COLORS)] for i in range(n)]


def save_cellquorum_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = 300,
    tight: bool = True,
    **kwargs: Any,
) -> Path:
    """
    Save a CellQuorum figure with consistent quality settings.

    Args:
        fig: Matplotlib figure to save.
        path: Output file path.
        dpi: Resolution in dots per inch (default: 300 for publication).
        tight: Whether to use tight_layout and bbox_inches='tight'.
        **kwargs: Additional arguments passed to fig.savefig().

    Returns:
        Path to saved figure.
    """
    path = Path(path)

    # Apply tight layout if requested
    if tight:
        fig.tight_layout()

    # Default save kwargs
    save_kwargs = {
        "dpi": dpi,
        "bbox_inches": "tight" if tight else None,
        "facecolor": "white",
        "edgecolor": "none",
    }
    save_kwargs.update(kwargs)

    # Save figure
    fig.savefig(path, **save_kwargs)

    return path


# =============================================================================
# Specialized Plot Styling Functions
# =============================================================================


def add_volcano_background_panels(
    ax: Axes,
    fc_threshold: float = 0.5,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Add subtle background color panels to volcano plot.

    This creates the signature three-region background from your volcano plots:
    - Blue panel for downregulated genes (left)
    - Gray panel for non-significant genes (center)
    - Red panel for upregulated genes (right)

    Args:
        ax: Matplotlib axis for volcano plot.
        fc_threshold: Fold-change threshold for significance regions.
        xlim: Optional x-axis limits. If None, uses current axis limits.
    """
    if xlim is None:
        xlim = ax.get_xlim()

    ymin, ymax = ax.get_ylim()

    # Left panel (downregulated) - blue
    ax.axvspan(xlim[0], -fc_threshold, facecolor=BACKGROUND_BLUE, alpha=0.3, zorder=0)

    # Center panel (non-significant) - gray
    ax.axvspan(-fc_threshold, fc_threshold, facecolor=BACKGROUND_GRAY, alpha=0.3, zorder=0)

    # Right panel (upregulated) - red
    ax.axvspan(fc_threshold, xlim[1], facecolor=BACKGROUND_RED, alpha=0.3, zorder=0)


def add_dashed_reference_lines(
    ax: Axes,
    fc_threshold: float = 0.5,
    p_threshold: float = 1.3,
    color: str | None = None,
    linewidth: float = 0.8,
) -> None:
    """
    Add dashed reference lines to volcano plot.

    Args:
        ax: Matplotlib axis for volcano plot.
        fc_threshold: Fold-change threshold (vertical lines at ±fc_threshold).
        p_threshold: -log10(p) threshold (horizontal line).
        color: Line color (default: CELLQUORUM_GRAY).
        linewidth: Line width.
    """
    if color is None:
        color = CELLQUORUM_GRAY

    # Vertical lines at fold-change thresholds
    ax.axvline(-fc_threshold, color=color, linestyle="--", linewidth=linewidth, zorder=1)
    ax.axvline(fc_threshold, color=color, linestyle="--", linewidth=linewidth, zorder=1)

    # Horizontal line at significance threshold
    ax.axhline(p_threshold, color=color, linestyle="--", linewidth=linewidth, zorder=1)


def add_directional_arrows(
    ax: Axes,
    left_label: str = "Higher in Control",
    right_label: str = "Higher in Treatment",
    y_position: float | None = None,
    fontsize: int = 10,
    color: str = "#333333",
) -> None:
    """
    Add directional arrow labels to top of volcano plot.

    This adds the characteristic "Higher in X ← → Higher in Y" labels.

    Args:
        ax: Matplotlib axis for volcano plot.
        left_label: Label for left (downregulated) side.
        right_label: Label for right (upregulated) side.
        y_position: Vertical position as fraction of axis (default: 0.95).
        fontsize: Font size for labels.
        color: Text color.
    """
    if y_position is None:
        y_position = 0.95

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_pos = ylim[0] + y_position * (ylim[1] - ylim[0])

    # Left arrow and label
    ax.annotate(
        left_label,
        xy=(xlim[0] + 0.1 * (xlim[1] - xlim[0]), y_pos),
        xytext=(xlim[0] + 0.25 * (xlim[1] - xlim[0]), y_pos),
        fontsize=fontsize,
        color=CELLQUORUM_BLUE,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "<-", "color": CELLQUORUM_BLUE, "lw": 1.5},
    )

    # Right arrow and label
    ax.annotate(
        right_label,
        xy=(xlim[1] - 0.1 * (xlim[1] - xlim[0]), y_pos),
        xytext=(xlim[1] - 0.25 * (xlim[1] - xlim[0]), y_pos),
        fontsize=fontsize,
        color=CELLQUORUM_RED,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": CELLQUORUM_RED, "lw": 1.5},
    )


def add_statistical_annotation_box(
    ax: Axes,
    stats: dict[str, Any],
    position: str = "lower left",
    fontsize: int = 8,
    box_alpha: float = 0.8,
) -> None:
    """
    Add a clean statistical annotation box to a plot.

    This creates the small, minimal statistical summary boxes seen in your figures.

    Args:
        ax: Matplotlib axis.
        stats: Dictionary of statistics to display (e.g., {'p<0.05': 123, 'log2FC>0.5': 456}).
        position: Box position ('lower left', 'lower right', 'upper left', 'upper right').
        fontsize: Font size for text.
        box_alpha: Box background alpha.
    """
    # Build annotation text
    lines = [f"{key}: {value}" for key, value in stats.items()]
    text = "\n".join(lines)

    # Position mapping
    position_map = {
        "lower left": {"x": 0.05, "y": 0.05, "ha": "left", "va": "bottom"},
        "lower right": {"x": 0.95, "y": 0.05, "ha": "right", "va": "bottom"},
        "upper left": {"x": 0.05, "y": 0.95, "ha": "left", "va": "top"},
        "upper right": {"x": 0.95, "y": 0.95, "ha": "right", "va": "top"},
    }

    pos = position_map.get(position, position_map["lower left"])

    # Add text box
    ax.text(
        pos["x"],
        pos["y"],
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=pos["ha"],
        va=pos["va"],
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": CELLQUORUM_GRAY,
            "linewidth": 0.5,
            "alpha": box_alpha,
        },
    )


def add_panel_letter(ax: Axes, letter: str, *, size: int = 22) -> None:
    """Draw a bold publication panel letter at the axes' upper-left corner."""

    # Match the lekc house placement: axes-fraction (-0.08, 0.98), bold, top-left.
    ax.text(
        -0.08,
        0.98,
        letter,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="top",
        ha="right",
    )


def get_group_palette(groups: list[str]) -> dict[str, str]:
    """Map group values to house-palette colors, deterministically by sorted order."""

    # Sort for determinism so the same groups always map to the same colors.
    ordered = sorted({str(g) for g in groups})
    colors = get_cellquorum_colors(len(ordered))
    return {group: colors[i] for i, group in enumerate(ordered)}


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
    "CELLQUORUM_FIGSIZE_LARGE",
    "apply_cellquorum_theme",
    "apply_cellquorum_axis_style",
    "get_cellquorum_colors",
    "save_cellquorum_figure",
    "add_volcano_background_panels",
    "add_dashed_reference_lines",
    "add_directional_arrows",
    "add_statistical_annotation_box",
    "add_panel_letter",
    "get_group_palette",
]
