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

    Ties are broken by the DataFrame index (ascending), ensuring deterministic
    selection regardless of incoming row order. Returns the union sorted ascending
    by ``value_col`` (then index for tied values).
    """
    # Sort by value, then by index as tie-breaker
    # We do this by temporarily making the index a column for multi-key sort
    work = df.copy()
    work["__index_tiebreak"] = work.index
    work = work.sort_values([value_col, "__index_tiebreak"], kind="mergesort")
    bottom = work.head(k)
    top = work.tail(k)
    combined = pd.concat([bottom, top])
    # Deduplicate by index
    combined = combined[~combined.index.duplicated(keep="first")]
    # Drop the temporary column and return sorted by value then index
    combined = combined.drop(columns=["__index_tiebreak"])
    combined["__index_tiebreak"] = combined.index
    combined = combined.sort_values([value_col, "__index_tiebreak"], kind="mergesort")
    combined = combined.drop(columns=["__index_tiebreak"])
    return combined


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
    ax.set_yticklabels([_wrap_label(s) for s in sel[label_col].astype(str)])
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
    ax.set_yticklabels([_wrap_label(s) for s in sel[label_col].astype(str)])
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
    gene_col: str = "gene",
    title: str | None = None,
    max_labeled_genes: int = 12,
) -> Figure:
    """Classic 3-track Subramanian plot: ES line, hit ticks, ranked metric area.

    The ranked-metric track used to be one Rectangle per ranked gene, which
    looks fine on a handful of rows but anti-aliases into a washed-out,
    near-invisible panel once the collection ranks a real transcriptome
    (20k+ genes at sub-pixel bar width) -- confirmed live against the real
    hallmark output. A filled area renders as a solid shape independent of
    point density, so both the ES curve and the ranked metric use
    fill_between instead of per-point patches.

    When ``gene_col`` is present, the hits track also labels the leading-edge
    genes -- hits at or before the peak for a positive score, at or after the
    trough for a negative one, the same split ``gsea_method.leading_edge()``
    uses to define which genes carry the enrichment. A real leading edge can
    run to dozens of genes, so labeling is capped at ``max_labeled_genes``,
    keeping the ones closest to the extreme end of the ranking (the
    strongest individual contributors) and staggering label height so
    closely-ranked genes don't collide.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1, 1]},
    )
    ax_es, ax_hits, ax_metric = axes
    for ax in (ax_es, ax_hits, ax_metric):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    rank = df[rank_col].to_numpy()
    es = df[es_col].to_numpy(dtype=float)
    es_color = "#2E7D32"
    ax_es.plot(rank, es, color=es_color, linewidth=1.8, zorder=3)
    ax_es.fill_between(rank, es, 0, color=es_color, alpha=0.15, linewidth=0, zorder=2)
    ax_es.axhline(0, color="0.6", linewidth=0.8, zorder=1)
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

    hit_mask = df[hit_col].to_numpy().astype(bool)
    ax_hits.vlines(rank[hit_mask], 0, 1, color="0.15", linewidth=0.5)
    ax_hits.set_yticks([])
    ax_hits.set_ylabel("hits", rotation=0, ha="right", va="center")
    ax_hits.spines["left"].set_visible(False)

    if gene_col in df.columns and max_labeled_genes > 0:
        positions = np.arange(len(es))
        if es[peak_i] >= 0:
            edge_positions = positions[hit_mask & (positions <= peak_i)][:max_labeled_genes]
        else:
            edge_positions = positions[hit_mask & (positions >= peak_i)][-max_labeled_genes:]
        genes = df[gene_col].to_numpy()
        # A strongly enriched real pathway packs most leading-edge hits into
        # the first few hundred of 20k+ ranks (confirmed against the real
        # EMT output) -- labeling each gene at its own true rank collapses
        # every label onto the same few pixels. Spread labels across an
        # artificial x-window instead, connected back to their true tick by
        # annotate()'s own leader line; the window is at least a fixed
        # fraction of the full rank axis so it can't collapse either, even
        # when every hit truly does land one rank apart.
        n_total = max(len(es), 1)
        min_window = 0.12 * (rank.max() - rank.min() if len(rank) > 1 else n_total)
        label_ranks = rank[edge_positions]
        if es[peak_i] >= 0:
            start = float(label_ranks.min()) if len(label_ranks) else float(rank.min())
            span = max(
                float(label_ranks.max() - label_ranks.min()) if len(label_ranks) else 0.0,
                min_window,
            )
            label_xs = np.linspace(start, start + span, len(edge_positions))
        else:
            end = float(label_ranks.max()) if len(label_ranks) else float(rank.max())
            span = max(
                float(label_ranks.max() - label_ranks.min()) if len(label_ranks) else 0.0,
                min_window,
            )
            label_xs = np.linspace(end - span, end, len(edge_positions))
        for j, pos in enumerate(edge_positions):
            label_y = 1.35 if j % 2 == 0 else 1.85
            ax_hits.annotate(
                str(genes[pos]),
                xy=(rank[pos], 1.0),
                xytext=(label_xs[j], label_y),
                rotation=90,
                fontsize=6.5,
                ha="center",
                va="bottom",
                arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.4},
                annotation_clip=False,
            )
        ax_hits.set_ylim(0, 2.6)
    else:
        ax_hits.set_ylim(0, 1)

    metric = df[metric_col].to_numpy(dtype=float)
    pos_color, neg_color = "#B2182B", "#2166AC"
    ax_metric.fill_between(
        rank, metric, 0, where=metric >= 0, color=pos_color, linewidth=0, interpolate=True
    )
    ax_metric.fill_between(
        rank, metric, 0, where=metric <= 0, color=neg_color, linewidth=0, interpolate=True
    )
    ax_metric.axhline(0, color="0.3", linewidth=0.8)
    ax_metric.set_ylabel("ranked metric")
    ax_metric.set_xlabel("rank")
    fig.tight_layout()
    return fig


def _wrap_label(text: str, *, width: int = 28, max_lines: int = 2) -> str:
    """Wrap a gene-set name onto at most ``max_lines`` lines, on "_" boundaries.

    Real Reactome/Hallmark names run 30-140+ characters -- constrained_layout
    can reflow axes around a long label, but it can't shrink the text itself,
    and a figure this size has no room for an 80-char single-line label.
    ``textwrap.fill`` alone breaks mid-word (there are no real spaces in a
    gene-set name, so it treats the whole underscore-joined string as one
    unbreakable "word" and splits it at an arbitrary character offset --
    confirmed live: "...BIOSYNTHES" / "IS_AND..."). Splitting on "_" and
    packing whole tokens per line keeps every break at a real word boundary,
    still capped at ``max_lines`` with a trailing placeholder for anything
    left over.
    """
    tokens = text.split("_")
    lines: list[str] = []
    current = ""
    consumed = 0
    for tok in tokens:
        candidate = tok if not current else f"{current}_{tok}"
        if len(candidate) <= width or not current:
            current = candidate
            consumed += 1
            continue
        if len(lines) + 1 >= max_lines:
            break
        lines.append(current)
        current = tok
        consumed += 1
    else:
        if current:
            lines.append(current)
        return "\n".join(lines)
    if current:
        lines.append(current)
    if consumed < len(tokens):
        lines[-1] = lines[-1] + " …"
    return "\n".join(lines[:max_lines])


def _facet_bar_or_dot(
    df: pd.DataFrame,
    *,
    facet_col: str,
    top_k: int,
    draw: Callable[[Axes, pd.DataFrame, str], None],
) -> Figure:
    """Shared faceting harness for ORA bar/dot plots (one column per facet value)."""
    facets = sorted(df[facet_col].astype(str).unique())
    # constrained_layout accounts for each facet's own post-hoc colorbar and
    # tick-label extents when it solves subplot positions, so a long y-tick
    # label (or a colorbar) can no longer bleed into the neighboring facet --
    # a fixed-position plt.subplots() grid has no mechanism to reflow around
    # artists added after the axes are created.
    fig, axes = plt.subplots(
        1,
        max(1, len(facets)),
        figsize=(5 * max(1, len(facets)), 5),
        squeeze=False,
        layout="constrained",
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
        ax.set_yticklabels([_wrap_label(s) for s in sub[label_col].astype(str)])
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
        ax.set_yticklabels([_wrap_label(s) for s in sub[label_col].astype(str)])
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
    # A reactome-scale collection can carry hundreds of rows even after
    # top_n filtering -- the per-row/per-column inch formula alone grows
    # without bound, so both dimensions are capped at a size that still
    # renders as a readable figure rather than a many-foot-tall PNG.
    width = min(16.0, max(6.0, 0.4 * mat.shape[1]))
    height = min(20.0, max(6.0, 0.3 * mat.shape[0]))
    # seaborn generates its own y-tick labels straight from the row index --
    # unlike every other figure in this module, there's no set_yticklabels
    # call to hook _wrap_label into, so the index itself is renamed instead.
    # Row CLUSTERING runs on the values, not the labels, so renaming the
    # index here (after top_n selection, before clustermap builds the
    # dendrogram) doesn't touch what gets clustered. max_lines=1, not the
    # ORA/GSEA facets' 2 -- a real GSVA heatmap can carry 20-30 rows in a
    # height-capped figure, and 2-line labels don't fit that little
    # per-row space without colliding with the row above/below (confirmed
    # live: a 2-line wrap rendered as an overlapping unreadable stack).
    mat = mat.rename(index=lambda s: _wrap_label(str(s), max_lines=1))
    return sns.clustermap(
        mat,
        cmap="RdBu_r",
        center=0,
        col_cluster=col_cluster,
        row_cluster=True,
        col_colors=col_colors,
        figsize=(width, height),
    )


def cross_group_dotplot(
    df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    top_k: int = 12,
    col_labels: dict[str, str] | None = None,
) -> Figure:
    """Grid col_col (x) × row_col (y): color=signed value, dot area=|value|.

    Rows selected as the top-K sources by across-group variance (deterministic).
    ``col_labels`` renames the x-axis tick text only -- column selection,
    ordering, and the scatter positions all still run on the raw ids, same
    display-only convention as every other cluster_labels site.
    """
    col_labels = col_labels or {}
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
    ax.set_xticklabels([col_labels.get(c, c) for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(keep_rows)))
    ax.set_yticklabels(keep_rows)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def cross_group_dotplot_by_condition(
    df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    condition_col: str,
    control: str,
    case: str,
    top_k: int = 12,
    col_labels: dict[str, str] | None = None,
    condition_labels: dict[str, str] | None = None,
) -> Figure:
    """Two panels (control | case), color = per-row z-score, shared scale.

    Matches decoupler's own documented convention for this exact plot --
    ``sc.pl.matrixplot(..., standard_scale="var")`` -- z-scoring each row
    (source) so one dominant source (e.g. MYC, an order of magnitude above
    everything else) doesn't compress every other row to a single shade on
    the color scale. The mean/std baseline is computed once, pooling BOTH
    conditions, so the two panels share one scale and stay comparable --
    same principle as embeddings.plot_score_by_condition's shared vmin/vmax.
    """
    col_labels = col_labels or {}
    condition_labels = condition_labels or {}
    sub = df[df[condition_col].astype(str).isin([control, case])].copy()
    sub["_group"] = sub[col_col].astype(str) + "||" + sub[condition_col].astype(str)
    pivot = sub.pivot_table(index=row_col, columns="_group", values=value_col, aggfunc="mean")

    var_by_row = pivot.var(axis=1).fillna(0.0)
    keep_rows = var_by_row.sort_values(kind="mergesort", ascending=False).head(top_k).index
    keep_rows = sorted(keep_rows)

    row_mean = pivot.loc[keep_rows].mean(axis=1)
    row_std = pivot.loc[keep_rows].std(axis=1).replace(0.0, 1.0).fillna(1.0)
    z = pivot.loc[keep_rows].sub(row_mean, axis=0).div(row_std, axis=0)

    cols = sorted(sub[col_col].astype(str).unique())
    row_pos = {r: i for i, r in enumerate(keep_rows)}
    col_pos = {c: i for i, c in enumerate(cols)}

    vmax = float(np.nanmax(np.abs(z.to_numpy(dtype=float)))) if z.size else 1.0
    vmax = vmax or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(8.0, 1.1 * len(cols) * 2), max(4.0, 0.4 * len(keep_rows))),
        sharey=True,
        layout="constrained",
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    for ax, condition in zip(axes, [control, case], strict=True):
        xs, ys, vals = [], [], []
        for r in keep_rows:
            for c in cols:
                key = f"{c}||{condition}"
                if key not in z.columns:
                    continue
                v = z.loc[r, key]
                if pd.notna(v):
                    xs.append(col_pos[c])
                    ys.append(row_pos[r])
                    vals.append(float(v))
        vals_arr = np.array(vals, dtype=float)
        sizes = 20.0 + 200.0 * (np.abs(vals_arr) / vmax) if len(vals_arr) else []
        ax.scatter(
            xs, ys, s=sizes, c=vals_arr, cmap=cmap, norm=norm, edgecolor="0.3", linewidth=0.4
        )
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([col_labels.get(c, c) for c in cols], rotation=45, ha="right")
        ax.set_title(condition_labels.get(condition, condition))
    axes[0].set_yticks(range(len(keep_rows)))
    axes[0].set_yticklabels(keep_rows)
    fig.colorbar(mappable, ax=axes, label=f"{value_col} (z-score)")
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
    "cross_group_dotplot_by_condition",
]
