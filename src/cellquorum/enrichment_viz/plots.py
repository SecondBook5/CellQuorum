"""Biology-agnostic plotting primitives for the enrichment-viz stage.

Each function takes a tidy DataFrame (or matrix) plus explicit column-name
arguments and draws. No file I/O, no config objects, no biological literals — the
caller maps its columns onto these generic parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Callable

# Discrete p-value anchors for the dot-size significance legend.
_SIZE_LEGEND_PVALUES = (0.05, 0.01, 1e-4)


def signed_norm(values: np.ndarray) -> TwoSlopeNorm:
    """TwoSlopeNorm centered at 0, guaranteed vmin < 0 < vmax.

    TwoSlopeNorm requires vmin < vcenter < vmax; data that does not straddle zero
    would raise. We pad the missing side with a small epsilon derived from the data
    magnitude so a purely-positive (or empty) input still yields a valid norm.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0
    mag = max(abs(lo), abs(hi), 1e-6)
    eps = mag * 1e-3
    vmin = min(lo, -eps)
    vmax = max(hi, eps)
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def pvalue_to_stars(p: float) -> str:
    """Significance stars: ****<1e-4, ***<1e-3, **<1e-2, *<0.05, else ''."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def select_top_bottom(df: pd.DataFrame, value_col: str, k: int) -> pd.DataFrame:
    """Top-k + bottom-k rows by ``value_col``, deterministic, deduplicated.

    Sorted deterministically (value then index) so ties break identically every
    run. Returns the union sorted ascending by ``value_col``.
    """
    ordered = df.sort_values([value_col], kind="mergesort")
    ordered = ordered.reindex(ordered[value_col].sort_values(kind="mergesort").index)
    bottom = ordered.head(k)
    top = ordered.tail(k)
    combined = pd.concat([bottom, top])
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_values([value_col], kind="mergesort")


def _size_from_padj(padj: np.ndarray) -> np.ndarray:
    """Map padj → dot area via -log10(padj), floored so p≈1 is still visible."""
    p = np.asarray(padj, dtype=float)
    p = np.where(np.isfinite(p) & (p > 0), p, 1.0)
    return 20.0 + 60.0 * np.clip(-np.log10(p), 0.0, 6.0)


def _add_size_legend(ax: Axes) -> None:
    """Discrete dot-size legend keyed to p = 0.05, 0.01, 1e-4."""
    handles = [
        ax.scatter([], [], s=_size_from_padj(np.array([p]))[0], color="0.4", label=f"p={p:g}")
        for p in _SIZE_LEGEND_PVALUES
    ]
    ax.legend(handles=handles, title="significance", loc="best", frameon=False, fontsize=8)


def diverging_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    pvalue_col: str | None = None,
    top_k: int = 12,
    ax: Axes | None = None,
) -> Axes:
    """Horizontal diverging bars colored by signed value (RdBu_r, center 0)."""
    sel = select_top_bottom(df, value_col, top_k)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(sel))))
    values = sel[value_col].to_numpy(dtype=float)
    norm = signed_norm(values)
    cmap = plt.get_cmap("RdBu_r")
    colors = cmap(norm(values))
    y = np.arange(len(sel))
    ax.barh(y, values, color=colors, edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sel[label_col].astype(str).tolist())
    ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel(value_col)
    if pvalue_col is not None and pvalue_col in sel.columns:
        for yi, (val, p) in enumerate(
            zip(values, sel[pvalue_col].to_numpy(dtype=float), strict=False)
        ):
            stars = pvalue_to_stars(p)
            if stars:
                offset = 0.01 * (abs(values).max() or 1.0)
                ha = "left" if val >= 0 else "right"
                ax.text(
                    val + (offset if val >= 0 else -offset),
                    yi,
                    stars,
                    va="center",
                    ha=ha,
                    fontsize=8,
                )
    ScalarMappable(norm=norm, cmap=cmap).set_array([])
    ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=value_col)
    return ax


def activity_dotplot(
    df: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    pvalue_col: str,
    top_k: int = 12,
    ax: Axes | None = None,
) -> Axes:
    """One source per row: x=value, color=signed value, dot area=-log10(padj)."""
    sel = select_top_bottom(df, value_col, top_k)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(sel))))
    values = sel[value_col].to_numpy(dtype=float)
    norm = signed_norm(values)
    cmap = plt.get_cmap("RdBu_r")
    sizes = _size_from_padj(sel[pvalue_col].to_numpy(dtype=float))
    y = np.arange(len(sel))
    ax.scatter(values, y, s=sizes, c=values, cmap=cmap, norm=norm, edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sel[label_col].astype(str).tolist())
    ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel(value_col)
    ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=value_col)
    _add_size_legend(ax)
    return ax


def running_es_curve(
    df: pd.DataFrame,
    *,
    rank_col: str = "rank",
    es_col: str = "running_es",
    hit_col: str = "hit",
    metric_col: str = "metric",
    title: str | None = None,
) -> Figure:
    """Classic 3-track Subramanian plot: ES line, hit ticks, ranked metric bar."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 0.4, 1]},
    )
    ax_es, ax_hits, ax_metric = axes
    rank = df[rank_col].to_numpy()
    es = df[es_col].to_numpy(dtype=float)
    ax_es.plot(rank, es, color="#2E7D32", linewidth=1.5)
    ax_es.axhline(0, color="0.6", linewidth=0.8)
    peak_i = int(np.argmax(np.abs(es)))
    ax_es.annotate(
        f"ES={es[peak_i]:.3f}",
        xy=(rank[peak_i], es[peak_i]),
        xytext=(0.6, 0.85),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    ax_es.set_ylabel("running ES")
    if title:
        ax_es.set_title(title)
    hit_ranks = rank[df[hit_col].to_numpy().astype(bool)]
    ax_hits.vlines(hit_ranks, 0, 1, color="black", linewidth=0.5)
    ax_hits.set_yticks([])
    ax_hits.set_ylabel("hits", rotation=0, ha="right", va="center")
    metric = df[metric_col].to_numpy(dtype=float)
    norm = signed_norm(metric)
    cmap = plt.get_cmap("RdBu_r")
    ax_metric.bar(rank, metric, width=1.0, color=cmap(norm(metric)))
    ax_metric.axhline(0, color="0.4", linewidth=0.8)
    ax_metric.set_ylabel("ranked metric")
    ax_metric.set_xlabel("rank")
    fig.tight_layout()
    return fig


def _facet_bar_or_dot(
    df: pd.DataFrame,
    *,
    facet_col: str,
    top_k: int,
    draw: Callable[[Axes, pd.DataFrame, str], None],
) -> Figure:
    """Shared faceting harness for ORA bar/dot plots (one column per facet value)."""
    facets = sorted(df[facet_col].astype(str).unique())
    fig, axes = plt.subplots(
        1, max(1, len(facets)), figsize=(5 * max(1, len(facets)), 5), squeeze=False
    )
    for ax, facet in zip(axes[0], facets, strict=False):
        sub = df[df[facet_col].astype(str) == facet]
        draw(ax, sub, facet)
    return fig


def ora_barplot(
    df: pd.DataFrame,
    *,
    count_col: str,
    label_col: str,
    padj_col: str,
    facet_col: str,
    top_k: int = 12,
) -> Figure:
    """clusterProfiler-style barplot: x=count, color=padj (YlOrRd), faceted."""
    cmap = plt.get_cmap("YlOrRd_r")

    def draw(ax: Axes, sub: pd.DataFrame, facet: str) -> None:
        sub = sub.sort_values(padj_col, kind="mergesort").head(top_k)
        sub = sub.sort_values(count_col, kind="mergesort")
        p = sub[padj_col].to_numpy(dtype=float)
        norm = Normalize(
            vmin=float(np.nanmin(p)) if p.size else 0.0, vmax=float(np.nanmax(p)) if p.size else 1.0
        )
        y = np.arange(len(sub))
        ax.barh(y, sub[count_col].to_numpy(dtype=float), color=cmap(norm(p)))
        ax.set_yticks(y)
        ax.set_yticklabels(sub[label_col].astype(str).tolist())
        ax.set_xlabel(count_col)
        ax.set_title(str(facet))
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=padj_col)

    return _facet_bar_or_dot(df, facet_col=facet_col, top_k=top_k, draw=draw)


def ora_dotplot(
    df: pd.DataFrame,
    *,
    ratio_col: str,
    count_col: str,
    padj_col: str,
    label_col: str,
    facet_col: str,
    top_k: int = 12,
) -> Figure:
    """clusterProfiler-style dotplot: x=gene_ratio, size=count, color=padj, faceted."""
    cmap = plt.get_cmap("YlOrRd_r")

    def draw(ax: Axes, sub: pd.DataFrame, facet: str) -> None:
        sub = sub.sort_values(padj_col, kind="mergesort").head(top_k)
        sub = sub.sort_values(ratio_col, kind="mergesort")
        p = sub[padj_col].to_numpy(dtype=float)
        norm = Normalize(
            vmin=float(np.nanmin(p)) if p.size else 0.0, vmax=float(np.nanmax(p)) if p.size else 1.0
        )
        y = np.arange(len(sub))
        counts = sub[count_col].to_numpy(dtype=float)
        sizes = 20.0 + 20.0 * counts
        ax.scatter(
            sub[ratio_col].to_numpy(dtype=float),
            y,
            s=sizes,
            c=p,
            cmap=cmap,
            norm=norm,
            edgecolor="0.3",
            linewidth=0.4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(sub[label_col].astype(str).tolist())
        ax.set_xlabel(ratio_col)
        ax.set_title(str(facet))
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=padj_col)

    return _facet_bar_or_dot(df, facet_col=facet_col, top_k=top_k, draw=draw)


def annotated_clustermap(
    matrix: pd.DataFrame,
    *,
    col_colors: pd.Series | None = None,
    top_n: int | None = None,
) -> sns.matrix.ClusterGrid:
    """sns.clustermap of a source×sample matrix (RdBu_r, center 0).

    ``col_colors`` (a condition strip) is drawn only when supplied; when present,
    column clustering is turned off and columns are ordered by the strip. Otherwise
    column clustering stays on. ``top_n`` keeps the top-N most variable rows.
    """
    mat = matrix
    if top_n is not None and mat.shape[0] > top_n:
        top_idx = mat.var(axis=1).sort_values(kind="mergesort", ascending=False).head(top_n).index
        mat = mat.loc[top_idx]
    col_cluster = col_colors is None
    if col_colors is not None:
        order = col_colors.sort_values(kind="mergesort").index
        mat = mat[order]
        col_colors = col_colors.reindex(order)
    return sns.clustermap(
        mat,
        cmap="RdBu_r",
        center=0,
        col_cluster=col_cluster,
        row_cluster=True,
        col_colors=col_colors,
        figsize=(max(6, 0.4 * mat.shape[1]), max(6, 0.3 * mat.shape[0])),
    )


def cross_group_dotplot(
    df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    top_k: int = 12,
) -> Figure:
    """Grid col_col (x) × row_col (y): color=signed value, dot area=|value|.

    Rows selected as the top-K sources by across-group variance (deterministic).
    """
    var_by_row = df.groupby(row_col)[value_col].var().fillna(0.0)
    keep_rows = var_by_row.sort_values(kind="mergesort", ascending=False).head(top_k).index
    keep_rows = sorted(keep_rows)
    cols = sorted(df[col_col].astype(str).unique())
    sub = df[df[row_col].isin(keep_rows)]
    row_pos = {r: i for i, r in enumerate(keep_rows)}
    col_pos = {c: i for i, c in enumerate(cols)}
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(cols)), max(4, 0.4 * len(keep_rows))))
    values = sub[value_col].to_numpy(dtype=float)
    norm = signed_norm(values)
    cmap = plt.get_cmap("RdBu_r")
    xs = [col_pos[str(c)] for c in sub[col_col]]
    ys = [row_pos[r] for r in sub[row_col]]
    mag = np.abs(values)
    sizes = 20.0 + 200.0 * (mag / (mag.max() or 1.0))
    ax.scatter(xs, ys, s=sizes, c=values, cmap=cmap, norm=norm, edgecolor="0.3", linewidth=0.4)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(keep_rows)))
    ax.set_yticklabels(keep_rows)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=value_col)
    fig.tight_layout()
    return fig


__all__ = [
    "signed_norm",
    "pvalue_to_stars",
    "select_top_bottom",
    "diverging_bar",
    "activity_dotplot",
    "running_es_curve",
    "ora_barplot",
    "ora_dotplot",
    "annotated_clustermap",
    "cross_group_dotplot",
]
