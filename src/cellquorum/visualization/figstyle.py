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

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

TEXT = "#202428"
NORMAL_BLUE = "#1B4F8A"
LE_RED = "#C41E3A"

# 18-hue colorblind-safe ordered palette (matches the house figure library).
CATEGORICAL_PALETTE: list[str] = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


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


__all__ = [
    "TEXT",
    "NORMAL_BLUE",
    "LE_RED",
    "CATEGORICAL_PALETTE",
    "set_style",
    "condition_palette",
    "diverging_norm",
    "significance_stars",
    "save_figure",
    "panel_letter",
]
