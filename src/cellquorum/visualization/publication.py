"""Publication plotting primitives reused from the LE-KC and mast-cell projects.

This module is the shared CellQuorum home for the proven figure grammar from:

* ``le_kc_signaling_hubs/src/lekc/figstyle.py``
* ``mast_cell_scrna/scripts/02_analysis/support/viz_config.py``

The intent is to reuse the working publication aesthetic without coupling
CellQuorum to one disease, tissue, or figure script. Biological modules should
import these primitives instead of redefining fonts, palettes, panel labels, or
embedding styling locally.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

if TYPE_CHECKING:
    import anndata as ad
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


TEXT_COLOR = "#25292C"
AXIS_COLOR = "#2C2C2C"
NORMAL_COLOR = "#24608F"
DISEASE_COLOR = "#C52A45"
QC_FAIL_COLOR = "#7E858B"
DOUBLET_COLOR = "#D1495B"

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

CATEGORICAL_PALETTE = [
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
    "#E15759",
    "#4E79A7",
    "#59A14F",
    "#B07AA1",
    "#F28E2B",
    "#76B7B2",
    "#FF9DA7",
    "#9C755F",
]

MAST_CELL_CATEGORICAL_PALETTE = [
    "#D94F3D",
    "#3A7EC9",
    "#4E9A8F",
    "#7B5EA7",
    "#E8A020",
    "#2A7A30",
    "#E8749A",
    "#FF6B35",
    "#7FB8E8",
    "#9ABF5E",
    "#8B6347",
    "#BDA0CB",
    "#E07B54",
    "#5CB85C",
    "#9EB4C4",
    "#B03A2E",
    "#FFB347",
    "#4DB6AC",
    "#A66BBE",
    "#81C784",
]

CELL_TYPE_COLORS = {
    "Keratinocytes": "#4E9A8F",
    "Fibroblasts": "#7B5EA7",
    "Endothelial": "#E8749A",
    "Pericytes": "#BDA0CB",
    "Melanocytes": "#8B6347",
    "Schwann cells": "#9EB4C4",
    "T cells": "#D94F3D",
    "CD4+ T cells": "#E07B54",
    "CD8+ T cells": "#B03A2E",
    "Regulatory T cells": "#F0A87C",
    "NK cells": "#E8A020",
    "B cells": "#3A7EC9",
    "Plasma cells": "#7FB8E8",
    "Macrophages": "#2A7A30",
    "Monocytes": "#5CB85C",
    "Dendritic cells": "#9ABF5E",
    "Mast cells": "#FF6B35",
    "Neutrophils": "#FFB347",
    "Unknown": "#AAAAAA",
}

CONDITION_COLORS = {
    "Normal": NORMAL_COLOR,
    "Control": NORMAL_COLOR,
    "Healthy": NORMAL_COLOR,
    "Lymphedema": DISEASE_COLOR,
    "LE": DISEASE_COLOR,
    "Disease": DISEASE_COLOR,
    "Case": DISEASE_COLOR,
}


def set_publication_style(*, dpi: int = 300, small: bool = False) -> None:
    """Apply the shared LE-KC/mast-cell publication style globally."""

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
    mpl.rcParams["axes.prop_cycle"] = cycler("color", MAST_CELL_CATEGORICAL_PALETTE)
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


def condition_palette(values: Sequence[str] | None = None) -> dict[str, str]:
    """Return condition colors, assigning categorical fallbacks when needed."""

    if values is None:
        return dict(CONDITION_COLORS)
    out: dict[str, str] = {}
    for index, value in enumerate(values):
        text = str(value)
        out[text] = CONDITION_COLORS.get(
            text,
            MAST_CELL_CATEGORICAL_PALETTE[index % len(MAST_CELL_CATEGORICAL_PALETTE)],
        )
    return out


def categorical_palette(values: Sequence[str]) -> dict[str, str]:
    """Return a stable categorical palette for observed labels."""

    return {
        str(value): CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)]
        for index, value in enumerate(values)
    }


def cell_type_palette(cell_types: Sequence[str] | None = None) -> dict[str, str]:
    """Return cell-type colors with categorical fallbacks for unknown labels."""

    if cell_types is None:
        return dict(CELL_TYPE_COLORS)
    return {
        str(cell_type): CELL_TYPE_COLORS.get(
            str(cell_type),
            MAST_CELL_CATEGORICAL_PALETTE[index % len(MAST_CELL_CATEGORICAL_PALETTE)],
        )
        for index, cell_type in enumerate(cell_types)
    }


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
        palette = condition_palette(data[x_col].dropna().astype(str).unique().tolist())
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
    """Add the LE-KC corner arrow-style embedding scale bar."""

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
    """Draw a labelled categorical embedding in the LE-KC reference style."""

    import matplotlib.pyplot as plt

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
