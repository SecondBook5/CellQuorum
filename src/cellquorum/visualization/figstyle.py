"""Canonical publication figure-style contract for cellquorum.

Single source of truth for the house aesthetic extracted from AJ's published
figures: editable vector text, sans fonts, no top/right spines, a fixed
colorblind-safe categorical palette, condition→red/blue mapping, a diverging
norm centered at 0, a significance-star ladder, dual PNG+PDF saving, and
publication panel letters.

Biology-free: every study-specific value (gene names, condition labels) is a
function argument, never a module constant.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import anndata as ad

TEXT = "#202428"
NORMAL_BLUE = "#1B4F8A"
LE_RED = "#C41E3A"

# ---------------------------------------------------------------------------
# Directional colors (signature volcano blue/gray/red). These are the exact
# hex values from the legacy style module so existing visual tests still pass.
# ---------------------------------------------------------------------------
CELLQUORUM_BLUE = "#5A8BC4"  # Directional low / downregulated
CELLQUORUM_RED = "#C45A5A"  # Directional high / upregulated
CELLQUORUM_GRAY = "#BDBDBD"  # Neutral / non-significant

# ---------------------------------------------------------------------------
# QC-semantic / generic colors (NOT disease). These encode QC status and
# generic axis text, never a specific biology/condition.
# ---------------------------------------------------------------------------
TEXT_COLOR = "#25292C"
AXIS_COLOR = "#2C2C2C"
NORMAL_COLOR = "#24608F"
QC_FAIL_COLOR = "#7E858B"
DOUBLET_COLOR = "#D1495B"

# Single named continuous colormap knob (one source of truth). All continuous
# heatmap/scatter colormaps default to this constant.
SEQUENTIAL_CMAP: str = "viridis"

# Background panel colors (very subtle, used in volcano plot regions).
BACKGROUND_BLUE = "#E8F0F8"  # Light blue for downregulated region
BACKGROUND_GRAY = "#F5F5F5"  # Light gray for neutral region
BACKGROUND_RED = "#F8E8E8"  # Light red for upregulated region

# Figure sizing constants.
CELLQUORUM_FIGSIZE_SMALL = (6, 4)  # Single panel diagnostic
CELLQUORUM_FIGSIZE_WIDE = (10, 4)  # Wide single row
CELLQUORUM_FIGSIZE_SQUARE = (6, 6)  # Square plots (UMAP, etc)
CELLQUORUM_FIGSIZE_LARGE = (12, 8)  # Multi-panel figure

# Cluster colors (distinct, harmonious palette for cluster/group diagnostics).
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

# Named figure sizes and font sizes for the compact publication grammar.
FIGSIZE = {
    "single": (3.4, 3.2),
    "single_tall": (3.4, 4.0),
    "double": (6.85, 3.2),
    "double_tall": (6.85, 5.5),
    "triple": (6.85, 2.8),
    "embedding": (3.2, 3.0),
    "embedding_pair": (6.5, 3.0),
    "embedding_2x2": (6.5, 6.0),
    "dotplot": (7.5, 4.5),
    "violin": (3.4, 3.8),
    "violin_wide": (5.5, 3.8),
    "volcano": (3.8, 4.2),
    "heatmap": (6.85, 5.0),
}

FONTSIZE = {
    "panel_label": 11,
    "title": 9,
    "axis_title": 8,
    "tick": 7,
    "legend": 7,
    "annotation": 6.5,
}

# 18-hue colorblind-safe ordered categorical palette.
#
# Slots 1-8 are the validated dataviz reference categorical theme, in its
# CVD-safe order (each adjacent pair clears the colorblind-separation and
# normal-vision gates on the light chart surface `#fcfcfb`). Slots 9-18 are an
# overflow tier for high-cardinality categorical use (e.g. many cell types /
# trajectory states); they are ordered so no two look-alike hues sit adjacent
# and the whole 18 clears the hard gates (lightness band, chroma floor, CVD
# separation, normal-vision floor) on the light surface.
#
# The ordering is NOT cosmetic: it was chosen by running the dataviz palette
# validator (`skills/dataviz/scripts/validate_palette.py`) over candidate
# orders and keeping one that passes. Do NOT reorder or add hues without
# re-running that validator — categorical identity beyond ~8 series relies on
# secondary encoding (direct labels / legend / position), which every figure in
# this engine provides. cellquorum renders only on the light (white) surface,
# so the palette is validated light-mode only.
CATEGORICAL_PALETTE: list[str] = [
    "#2a78d6",  # blue      (dataviz core 1)
    "#eb6834",  # orange    (dataviz core 2)
    "#1baf7a",  # aqua      (dataviz core 3)
    "#eda100",  # yellow    (dataviz core 4)
    "#e87ba4",  # magenta   (dataviz core 5)
    "#008300",  # green     (dataviz core 6)
    "#4a3aa7",  # violet    (dataviz core 7)
    "#e34948",  # red       (dataviz core 8)
    "#17becf",  # overflow: cyan
    "#7b2ff7",  # overflow: purple
    "#0aa86b",  # overflow: emerald
    "#c1121f",  # overflow: crimson
    "#118ab2",  # overflow: steel blue
    "#e07a00",  # overflow: amber
    "#b5179e",  # overflow: fuchsia
    "#00a1b0",  # overflow: teal
    "#ef476f",  # overflow: rose
    "#d05ce3",  # overflow: orchid
]


def distinct_palette(n: int) -> list[str]:
    """Return ``n`` maximally-distinct hex colors spanning jewel + vivid-bright tones.

    Two things make the ``n`` colors stay apart far past where a fixed list would
    blur: (1) hue advances by the golden angle rather than an even step, which
    spreads hues without clustering for *any* ``n``; (2) each successive color
    also steps through a small set of ``(saturation, value)`` shade tiers — deep
    jewel through bright vivid — so neighbours differ in depth as well as hue.
    Unlike cycling a fixed list, this never repeats a color when categories
    outnumber it (the cycling failure mode — several clusters sharing one hue —
    is what makes a many-group embedding look muddy). Deterministic in ``n``;
    biology-free (no fixed category meaning).

    Even so, no palette can make (say) 40 groups *easily* separable — that many
    distinct colors is a signal to reduce the cluster count (principled
    clustering) rather than a palette defect.
    """
    if n <= 0:
        return []
    # (saturation, value) shade tiers: deep jewel → bright vivid → soft-bright →
    # deep-vivid. Cycling tiers as hue advances means adjacent colors differ in
    # depth, not just hue — the extra separation "vivid bright" buys us.
    shades = ((0.70, 0.58), (0.88, 0.90), (0.58, 0.80), (0.95, 0.68))
    golden = 0.6180339887498949  # golden-angle hue step: even hue spread for any n
    start = 0.60  # begin at a sapphire blue and travel the wheel
    hexes: list[str] = []
    for i in range(n):
        hue = (start + i * golden) % 1.0
        sat, val = shades[i % len(shades)]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        hexes.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return hexes


def set_style() -> None:
    """Apply the house rcParams (idempotent)."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def condition_palette(
    case: str | None,
    control: str | None,
    *,
    case_color: str = LE_RED,
    control_color: str = NORMAL_BLUE,
    others: list[str] | None = None,
) -> dict[str, str]:
    """Map case→red, control→blue; extra conditions → categorical by sorted order."""
    palette: dict[str, str] = {}
    if case is not None:
        palette[str(case)] = case_color
    if control is not None:
        palette[str(control)] = control_color
    extra = sorted({str(o) for o in (others or []) if str(o) not in palette})
    for i, name in enumerate(extra):
        palette[name] = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
    return palette


def diverging_norm(values: np.ndarray, *, vmax: float | None = None) -> TwoSlopeNorm:
    """Signed norm centered at 0 (RdBu_r companion)."""
    if vmax is not None:
        m = abs(float(vmax))
        m = m if m > 0 else 1e-6
        return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0
    mag = max(abs(lo), abs(hi), 1e-6)
    eps = mag * 1e-3
    return TwoSlopeNorm(vmin=min(lo, -eps), vcenter=0.0, vmax=max(hi, eps))


def significance_stars(p: float) -> str:
    """`***`<0.001, `**`<0.01, `*`<0.05, else `ns` (NaN → `ns`)."""
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return "ns"
    if math.isnan(pv):
        return "ns"
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    if pv < 0.05:
        return "*"
    return "ns"


def save_figure(
    fig: Figure,
    out_dir: Path | str,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Write ``fig`` to ``out_dir/stem.<fmt>`` for each format, then close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def panel_letter(ax: Axes, letter: str, *, size: int = 15) -> None:
    """Bold publication panel letter at the axes' upper-left corner."""
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


# ===========================================================================
# Theme / axis helpers (ported verbatim from the legacy style module).
# ===========================================================================


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


def get_group_palette(groups: list[str]) -> dict[str, str]:
    """Map group values to house-palette colors, deterministically by sorted order."""

    # Sort for determinism so the same groups always map to the same colors.
    ordered = sorted({str(g) for g in groups})
    colors = get_cellquorum_colors(len(ordered))
    return {group: colors[i] for i, group in enumerate(ordered)}


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
        path: Output file path (a full path, distinct from ``save_figure``).
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


# ===========================================================================
# Volcano / annotation helpers (ported verbatim from the legacy style module).
# ===========================================================================


def add_volcano_background_panels(
    ax: Axes,
    fc_threshold: float = 0.5,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Add subtle background color panels to volcano plot.

    Blue panel for downregulated genes (left), gray for non-significant genes
    (center), red for upregulated genes (right).

    Args:
        ax: Matplotlib axis for volcano plot.
        fc_threshold: Fold-change threshold for significance regions.
        xlim: Optional x-axis limits. If None, uses current axis limits.
    """
    if xlim is None:
        xlim = ax.get_xlim()

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

    Args:
        ax: Matplotlib axis.
        stats: Dictionary of statistics to display.
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
    """Draw a bold publication panel letter at the axes' upper-left corner (size 22)."""

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


# ===========================================================================
# Publication primitives (ported biology-free from the legacy publication
# module). Palettes draw from CATEGORICAL_PALETTE; no disease constants.
# ===========================================================================


def set_publication_style(*, dpi: int = 300, small: bool = False) -> None:
    """Apply the shared compact publication style globally."""

    base_font = 7 if small else FONTSIZE["tick"]
    title_font = 8 if small else FONTSIZE["title"]
    axis_font = 7 if small else FONTSIZE["axis_title"]
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": base_font,
            "axes.labelsize": axis_font,
            "axes.titlesize": title_font,
            "axes.titleweight": "bold",
            "xtick.labelsize": 5.8 if small else FONTSIZE["tick"],
            "ytick.labelsize": 6 if small else FONTSIZE["tick"],
            "legend.fontsize": 6 if small else FONTSIZE["legend"],
            "legend.title_fontsize": 6 if small else FONTSIZE["legend"],
            "legend.frameon": False,
            "legend.borderpad": 0.3,
            "axes.linewidth": 0.7 if small else 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": AXIS_COLOR,
            "text.color": AXIS_COLOR,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "lines.linewidth": 1.25,
            "patch.linewidth": 0.5,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    mpl.rcParams["axes.prop_cycle"] = cycler("color", CATEGORICAL_PALETTE)
    sns.set_style(
        "ticks",
        {
            "axes.linewidth": 0.75,
            "axes.edgecolor": AXIS_COLOR,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
        },
    )
    sns.set_context("paper", font_scale=1.0)


def categorical_palette(values: Sequence[str]) -> dict[str, str]:
    """Return a stable categorical palette for observed labels."""

    return {
        str(value): CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)]
        for index, value in enumerate(values)
    }


def cell_type_palette(cell_types: Sequence[str] | None = None) -> dict[str, str]:
    """Return a stable categorical palette for observed cell-type labels.

    Biology-free: labels map to the validated categorical palette by observed
    order; there is no hardcoded cell-type→color table.
    """

    if cell_types is None:
        return {}
    return categorical_palette(cell_types)


def add_panel_label(ax: Axes, label: str, *, x: float = -0.15, y: float = 1.06) -> None:
    """Add a bold uppercase panel label to an axis."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FONTSIZE["panel_label"],
        fontweight="bold",
        va="top",
        ha="left",
        color="#1A1A1A",
    )


def pvalue_to_stars(pvalue: float) -> str:
    """Convert a p-value to common asterisk notation."""

    if pvalue < 0.0001:
        return "****"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def add_stat_bracket(
    ax: Axes,
    x1: float,
    x2: float,
    y_data_max: float,
    pvalue: float,
    *,
    gap: float = 0.04,
) -> None:
    """Draw a significance bracket above data between two x positions."""

    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    y_line = y_data_max + span * gap
    y_text = y_line + span * 0.01
    ax.plot(
        [x1, x1, x2, x2],
        [y_line - span * 0.01, y_line, y_line, y_line - span * 0.01],
        lw=0.8,
        color=AXIS_COLOR,
    )
    label = pvalue_to_stars(float(pvalue))
    font_size = FONTSIZE["annotation"] if label != "ns" else FONTSIZE["annotation"] - 0.5
    ax.text(
        (x1 + x2) / 2,
        y_text,
        label,
        ha="center",
        va="bottom",
        fontsize=font_size,
        color=AXIS_COLOR,
    )
    ax.set_ylim(ylim[0], y_text + span * 0.08)


def violin_with_stats(
    ax: Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    palette: dict[str, str] | None = None,
    order: Sequence[str] | None = None,
    point_size: float = 2.0,
    alpha_violin: float = 0.55,
    alpha_points: float = 0.35,
) -> Axes:
    """Layer violin, box, jitter, and optional Mann-Whitney annotation."""

    from scipy import stats

    if palette is None:
        palette = categorical_palette(data[x_col].dropna().astype(str).unique().tolist())
    if order is None:
        order = [key for key in palette if key in set(data[x_col].astype(str))]
    if not order:
        order = data[x_col].dropna().astype(str).unique().tolist()

    plot_data = data.copy()
    plot_data[x_col] = plot_data[x_col].astype(str)
    sns.violinplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=x_col,
        order=list(order),
        hue_order=list(order),
        palette=palette,
        inner=None,
        linewidth=0.75,
        cut=0,
        ax=ax,
        saturation=0.9,
        legend=False,
    )
    for collection in ax.collections:
        if hasattr(collection, "set_alpha"):
            collection.set_alpha(alpha_violin)

    sns.boxplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=x_col,
        order=list(order),
        hue_order=list(order),
        width=0.10,
        palette=palette,
        linewidth=0.75,
        fliersize=0,
        ax=ax,
        boxprops={"alpha": 0.9},
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.75},
        capprops={"linewidth": 0.75},
        legend=False,
    )
    sns.stripplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=x_col,
        order=list(order),
        hue_order=list(order),
        palette=palette,
        size=point_size,
        alpha=alpha_points,
        jitter=True,
        dodge=False,
        ax=ax,
        linewidth=0,
        legend=False,
    )

    if len(order) == 2:
        g1 = plot_data.loc[plot_data[x_col].eq(order[0]), y_col].dropna().to_numpy(dtype=float)
        g2 = plot_data.loc[plot_data[x_col].eq(order[1]), y_col].dropna().to_numpy(dtype=float)
        if len(g1) >= 3 and len(g2) >= 3:
            _, pvalue = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            add_stat_bracket(ax, 0, 1, float(plot_data[y_col].max()), float(pvalue))

    sns.despine(ax=ax)
    return ax


def clean_axis(ax: Axes, *, grid: bool = False) -> None:
    """Apply the shared clean-axis finish."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.45)


def remove_embedding_axes(ax: Axes) -> None:
    """Remove embedding ticks/spines and enforce equal aspect."""

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def embedding_limits(
    xy: np.ndarray, clip_pct: float | None = None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return embedding limits with padding and optional percentile clipping."""

    points = np.asarray(xy)
    x, y = points[:, 0], points[:, 1]
    if clip_pct is not None:
        xlo, xhi = np.percentile(x, [clip_pct, 100 - clip_pct])
        ylo, yhi = np.percentile(y, [clip_pct, 100 - clip_pct])
    else:
        xlo, xhi, ylo, yhi = x.min(), x.max(), y.min(), y.max()
    px = 0.05 * (xhi - xlo)
    py = 0.05 * (yhi - ylo)
    return (float(xlo - px), float(xhi + px)), (float(ylo - py), float(yhi + py))


def add_embedding_scalebar(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    x_label: str = "UMAP1",
    y_label: str = "UMAP2",
) -> None:
    """Add the corner arrow-style embedding scale bar."""

    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    dx = 0.16 * (xlim[1] - xlim[0])
    dy = 0.16 * (ylim[1] - ylim[0])
    ax.annotate(
        "",
        xy=(x0 + dx, y0),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "color": TEXT_COLOR, "lw": 1.1},
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(x0, y0 + dy),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "color": TEXT_COLOR, "lw": 1.1},
        clip_on=False,
    )
    ax.text(
        x0 + dx / 2, y0 - 0.03 * (ylim[1] - ylim[0]), x_label, ha="center", va="top", fontsize=7
    )
    ax.text(
        x0 - 0.03 * (xlim[1] - xlim[0]),
        y0 + dy / 2,
        y_label,
        ha="right",
        va="center",
        rotation=90,
        fontsize=7,
    )


def categorical_embedding(
    adata: ad.AnnData,
    group_key: str,
    *,
    basis: str = "X_umap",
    title: str = "",
    palette: dict[str, str] | Sequence[str] | None = None,
    order: Sequence[str] | None = None,
    label_on_plot: bool = True,
    point_size: float = 2.0,
    alpha: float = 0.8,
    legend: bool = False,
    axis_labels: tuple[str, str] = ("UMAP1", "UMAP2"),
    clip_pct: float | None = None,
    panel_letter: str = "",
    figsize: tuple[float, float] = (5.2, 5.0),
    ax: Axes | None = None,
) -> Figure:
    """Draw a labelled categorical embedding in the reference style."""

    xy = np.asarray(adata.obsm[basis])
    groups = adata.obs[group_key].astype(str).to_numpy()
    categories = list(order) if order is not None else sorted(pd.unique(groups))
    if palette is None:
        resolved_palette = categorical_palette(categories)
    elif isinstance(palette, dict):
        resolved_palette = dict(palette)
    else:
        resolved_palette = {
            category: palette[index % len(palette)] for index, category in enumerate(categories)
        }

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for category in categories:
        mask = groups == category
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            c=resolved_palette.get(category, "#999999"),
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=category if legend else None,
        )

    xlim, ylim = embedding_limits(xy, clip_pct)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if label_on_plot:
        centers = (
            pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "g": groups})
            .groupby("g")[["x", "y"]]
            .median()
        )
        for category in categories:
            if category not in centers.index:
                continue
            row = centers.loc[category]
            ax.annotate(
                category,
                xy=(row.x, row.y),
                xytext=(row.x, row.y),
                fontsize=7.5,
                color=TEXT_COLOR,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
            )

    remove_embedding_axes(ax)
    add_embedding_scalebar(ax, xlim, ylim, x_label=axis_labels[0], y_label=axis_labels[1])
    if title:
        ax.set_title(title, pad=6)
    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            markerscale=3,
            handletextpad=0.2,
        )
    if panel_letter:
        add_panel_label(ax, panel_letter, x=-0.02, y=1.04)
    if own_fig:
        fig.tight_layout()
    return fig


def save_publication_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    tight: bool = True,
    facecolor: str = "white",
    **kwargs: Any,
) -> Path:
    """Save a publication figure with consistent editable-vector defaults."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    save_kwargs: dict[str, Any] = {"dpi": dpi, "facecolor": facecolor}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    save_kwargs.update(kwargs)
    fig.savefig(out, **save_kwargs)
    return out


__all__ = [
    # Core contract (pinned).
    "TEXT",
    "NORMAL_BLUE",
    "LE_RED",
    "CATEGORICAL_PALETTE",
    "distinct_palette",
    "set_style",
    "condition_palette",
    "diverging_norm",
    "significance_stars",
    "save_figure",
    "panel_letter",
    # Directional colors.
    "CELLQUORUM_BLUE",
    "CELLQUORUM_RED",
    "CELLQUORUM_GRAY",
    # QC-semantic / generic colors.
    "TEXT_COLOR",
    "NORMAL_COLOR",
    "QC_FAIL_COLOR",
    "DOUBLET_COLOR",
    # Continuous colormap knob.
    "SEQUENTIAL_CMAP",
    # Figure sizing and background panels.
    "CELLQUORUM_FIGSIZE_SMALL",
    "CELLQUORUM_FIGSIZE_WIDE",
    "CELLQUORUM_FIGSIZE_SQUARE",
    "CELLQUORUM_FIGSIZE_LARGE",
    "CELLQUORUM_CLUSTER_COLORS",
    "BACKGROUND_BLUE",
    "BACKGROUND_GRAY",
    "BACKGROUND_RED",
    # Theme / axis helpers.
    "apply_cellquorum_theme",
    "apply_cellquorum_axis_style",
    "get_cellquorum_colors",
    "get_group_palette",
    "save_cellquorum_figure",
    # Volcano / annotation helpers.
    "add_volcano_background_panels",
    "add_dashed_reference_lines",
    "add_directional_arrows",
    "add_statistical_annotation_box",
    "add_panel_letter",
    # Publication primitives.
    "add_panel_label",
    "pvalue_to_stars",
    "add_stat_bracket",
    "violin_with_stats",
    "clean_axis",
    "remove_embedding_axes",
    "embedding_limits",
    "add_embedding_scalebar",
    "categorical_embedding",
    "save_publication_figure",
    "set_publication_style",
    "categorical_palette",
    "cell_type_palette",
]
