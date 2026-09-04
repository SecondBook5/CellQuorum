"""GSEA figures: the pathway panel, and the walk behind one pathway's score.

A GSEA result is normally published as a table of pathway names and NES values, which
hides the three things a reader needs to judge it.

* **How much of the pathway is actually carrying the score.** A NES of 1.6 over a
  200-gene annotation of which 28 genes sit in the leading edge is a claim about 28
  genes. The table's ``size`` column is the annotation; the enrichment is the edge. So
  :func:`gsea_dotplot` puts the leading-edge size in the dot area, at a fixed scale, and
  the annotation size stays in the table where it belongs.
* **Whether the p-value was measured or floored.** A permutation test cannot resolve
  below ``1/(permutations + 1)``. Pathways sitting on that floor are the strongest
  results in the run and they all report the same number, so a table sorted by p-value
  orders its own headline arbitrarily. Rows at the floor are marked, and the panel is
  ordered by score.
* **Where in the ranking the enrichment happened.** That is the running walk, and it is
  the only view in which the score is checkable rather than taken on faith:
  :func:`gsea_running_es` draws it with the leading edge shaded, so the reported ``es``
  and the reported edge size are both visibly the peak and the region left of it.

A fourth thing is hidden not by the table but by there being two of them.
:func:`gsea_arm_comparison` draws one pathway per row with a dot per arm, because a
specificity claim — "enriched here and not there" — is a statement about a *difference*, and
two tables read one after the other are the worst way to see one. The panel is ordered by
that difference, and it distinguishes the three ways an arm can fail to support a pathway:
significant the other way, tested and null, and never tested at all. The last is not a null
result and looks exactly like one in a table, because the row is simply absent.

All of these read the frames the enrichment stage writes — ``enrichment_gsea_*.csv`` and
``enrichment_gsea_runningES_*.csv`` — and nothing else, so a figure can always be redrawn
from a finished run directory with no recompute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import (
    CATEGORICAL_PALETTE,
    LE_RED,
    NORMAL_BLUE,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    significance_stars,
)
from cellquorum.visualization.measured_layout import (
    INK,
    MUTED,
    row_panel_canvas,
    stacked_panel_canvas,
    widest_label_in,
    write_notes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Width of the dotplot's data area, in inches. Fixed rather than derived, for the same
#: reason the slope panel's is: two collections drawn side by side are only comparable if
#: a NES of 2 is the same distance from zero in both.
_DOT_DATA_IN = 3.4

#: Point-area per leading-edge gene. Fixed across figures, so a dot means the same gene
#: count everywhere — the alternative is a per-figure scale, under which the same pathway
#: looks like a bigger finding in the panel that happened to hold weaker ones.
_AREA_PER_GENE = 3.0

#: Floor on the drawn dot area. A four-gene edge would otherwise be a dot a reader cannot
#: tell from a mark on the paper.
_MIN_DOT_AREA = 10.0

#: Room per row in the dotplot. Wider than the slope panel's, because the largest dots are
#: ~17 pt across and a 0.24 in row would have them touching.
_DOT_ROW_IN = 0.30

#: Mark for a row whose p-value is at the permutation floor. The star beside it is a bound.
_AT_LIMIT_MARK = "†"

#: Room the size key occupies above the axes, in inches. The title is padded by this much
#: so the two do not land on each other, which is what a default title pad does. It covers
#: the markers *and* the key's own heading — measured, because reserving only the marker row
#: leaves the heading a line's height under the title and the two read as one block.
_LEGEND_IN = 0.50

#: Dot area in the arm-comparison panel. Fixed, not a leading-edge size: the quantity that
#: panel is about is the gap between two dots, and a dot whose area also varies makes that gap
#: harder to read rather than richer.
_COMPARISON_DOT_AREA = 46.0

#: Room per row in the arm-comparison panel. Narrower than the dotplot's, since the dots are a
#: fixed size and cannot grow into each other.
_COMPARISON_ROW_IN = 0.26

#: Heights of the running-ES tracks, top to bottom: the walk, the hit positions, the
#: ranking metric. The walk is the figure; the other two are what it was computed from.
_WALK_TRACK_IN = (1.55, 0.24, 0.62)


def _direction_color(score: float) -> str:
    """Case colour for a positive score, control colour for a negative one."""
    return LE_RED if float(score) > 0 else NORMAL_BLUE


def _pretty(name: str, labels: Mapping[str, str] | None) -> str:
    if labels and name in labels:
        return labels[name]
    return str(name).replace("_", " ")


def _row_labels(
    names: Sequence[str], labels: Mapping[str, str] | None, max_chars: int | None
) -> tuple[list[str], list[str]]:
    """Row labels for a set of pathway names, plus the notes explaining what was done to them.

    Two things happen here, both because a pathway name is not a label anyone chose. MSigDB
    names carry their collection as a prefix — every row of a Reactome panel begins
    ``REACTOME`` — which is nine characters of gutter repeated down the figure and says
    nothing a caption does not. And they have no length bound: one Reactome set is 103
    characters, and since the gutter is the widest label, that one name sets the width of the
    whole figure and shrinks every other element to fit beside it.

    So a prefix shared by *every* drawn row is dropped, and the rest are truncated to a stated
    width. Both are recorded in the returned notes rather than done silently, because a
    truncated name is no longer something a reader can look up — the note says the table has
    it in full.

    Neither applies to a name the caller supplied through ``set_labels``. Those are drawn as
    given: a caller who wrote a display name has already made this decision, and a rule that
    edits it is second-guessing an explicit choice rather than filling a gap. It also cannot
    be done safely — two chosen labels that happen to share a first word are not a collection
    prefix, and stripping it would mangle both.
    """
    supplied = {str(name) for name in names if labels and str(name) in labels}
    drawn = {str(name): _pretty(str(name), labels) for name in names}
    derived = [name for name in drawn if name not in supplied]
    notes: list[str] = []

    heads = {drawn[name].split(" ", 1)[0] for name in derived}
    if len(heads) == 1 and len(derived) > 1:
        head = heads.pop()
        stripped = {name: drawn[name][len(head) :].strip() for name in derived}
        if all(stripped.values()):
            drawn.update(stripped)
            notes.append(f"Pathway names have their shared '{head}' prefix removed.")

    if max_chars is not None:
        long = [name for name in derived if len(drawn[name]) > max_chars]
        for name in long:
            drawn[name] = drawn[name][: max_chars - 1].rstrip() + "…"
        if long:
            notes.append(
                f"{len(long)} name(s) longer than {max_chars} characters are truncated with "
                "an ellipsis; the table carries them in full."
            )
    return [drawn[str(name)] for name in names], notes


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """One column as floats, or all-NaN if the frame does not carry it."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _flags(frame: pd.DataFrame, column: str) -> pd.Series:
    """One boolean column, read tolerantly of a CSV round-trip.

    ``bool("False")`` is ``True``, so a frame that came back off disk with this column
    parsed as text would mark every row. The strings are matched instead.
    """
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == object:
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return values.fillna(False).astype(bool)


def _legend_sizes(values: Sequence[float]) -> list[int]:
    """Up to three round gene counts spanning the drawn range, smallest first.

    Anchored on the data rather than on fixed round numbers: a legend offering 10/50/100
    against a panel whose edges run 4 to 40 asks the reader to interpolate outside the
    figure.
    """
    finite = sorted({int(round(v)) for v in values if np.isfinite(v) and v > 0})
    if not finite:
        return []
    if len(finite) <= 3:
        return finite
    low, high = finite[0], finite[-1]
    middle = finite[len(finite) // 2]
    # Round the middle to something a reader can hold in mind, without letting it collide
    # with either end.
    step = 10 if high >= 40 else 5
    rounded = max(step, int(round(middle / step)) * step)
    if not low < rounded < high:
        return [low, high]
    return [low, rounded, high]


def gsea_notes(
    table: pd.DataFrame,
    *,
    alpha: float = 0.05,
    dropped: int = 0,
) -> list[str]:
    """What a reader has to be told to read a GSEA panel correctly.

    Derived from the frame rather than written into the caller, so the permutation count
    and the resolution limit in the caption are the ones the run actually used.

    Args:
        table: A frame from the enrichment stage's ``enrichment_gsea_*.csv``.
        alpha: FDR threshold the panel marked significance at.
        dropped: How many sets the panel did not draw, if it truncated.

    Returns:
        The notes, in reading order.
    """
    notes = [
        "Dot area is the number of leading-edge genes — the part of the annotation that "
        "carries the enrichment, not the whole pathway. Filled = FDR < "
        f"{alpha:g}; open = not significant.",
        "Score is the normalised enrichment score (NES); the raw ES the walk peaks at is "
        "in the table beside it.",
    ]

    permutations = _numeric(table, "permutations").dropna().unique()
    limits = _numeric(table, "p_resolution_limit").dropna().unique()
    at_limit = int(_flags(table, "p_at_resolution_limit").sum())
    if at_limit and len(permutations) == 1 and len(limits) == 1:
        notes.append(
            f"{_AT_LIMIT_MARK} p is at the resolution limit of the permutation test "
            f"(p ≤ 1/(n+1) = {limits[0]:.1g} over {int(permutations[0]):,} "
            f"permutations) for {at_limit} pathway(s): the star is a bound, not a "
            "measurement."
        )
    elif at_limit:
        notes.append(
            f"{_AT_LIMIT_MARK} p is at the permutation test's resolution limit for "
            f"{at_limit} pathway(s); the star is a bound, not a measurement."
        )

    if dropped:
        notes.append(
            f"{dropped} further pathway(s) with smaller |NES| are not drawn; every "
            "pathway is in the table."
        )
    return notes


def gsea_dotplot(
    table: pd.DataFrame,
    *,
    alpha: float = 0.05,
    max_sets: int | None = None,
    significant_only: bool = False,
    set_labels: Mapping[str, str] | None = None,
    max_label_chars: int | None = 48,
    case_label: str | None = None,
    control_label: str | None = None,
    title: str | None = None,
    footnotes: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw one row per pathway: score on x, leading-edge size in the dot.

    Ordered by signed score, so the two directions separate and the panel reads like the
    contrast it came from rather than like an alphabetical list. Never ordered by p-value:
    at a high permutation count the strongest pathways share one floored p, and sorting on
    it would order the headline result arbitrarily.

    Args:
        table: A frame from ``enrichment_gsea_*.csv``. Needs ``source`` and ``score``;
            ``padj``, ``leading_edge_size``, ``p_at_resolution_limit``,
            ``p_resolution_limit`` and ``permutations`` are each used if present.
        alpha: FDR threshold. Below it a dot is filled, above it open.
        max_sets: Keep at most this many pathways, the largest ``|score|`` first. ``None``
            draws every row. Truncation is footnoted, never silent.
        significant_only: Drop the rows above ``alpha`` before selecting, rather than
            drawing them open.
        set_labels: Display names, keyed by the values in ``source``.
        max_label_chars: Truncate a row label longer than this, and footnote that it was
            truncated. ``None`` draws every name in full, at the cost of letting one long
            pathway name set the width of the whole figure — the gutter is the widest label,
            so a 103-character Reactome name shrinks every other element to fit beside it.
        case_label: Name of the arm a positive score points to, for the axis label.
        control_label: Name of the arm a negative score points to.
        title: Figure title.
        footnotes: Place :func:`gsea_notes` under the axes.
        figsize: Overrides the size derived from the number of rows.

    Returns:
        The figure.

    Raises:
        ValueError: ``table`` has no row with a finite score to draw.
    """
    apply_cellquorum_theme()

    frame = table.copy()
    frame["_score"] = _numeric(frame, "score")
    frame["_fdr"] = _numeric(frame, "padj")
    frame["_edge"] = _numeric(frame, "leading_edge_size")
    frame["_at_limit"] = _flags(frame, "p_at_resolution_limit")
    frame = frame[frame["_score"].notna()]
    if significant_only:
        frame = frame[frame["_fdr"] < alpha]
    if frame.empty:
        raise ValueError("no pathway has a finite score to draw")

    frame = frame.reindex(
        frame["_score"].abs().sort_values(ascending=False, kind="mergesort").index
    )
    dropped = 0
    if max_sets is not None and len(frame) > max_sets:
        dropped = len(frame) - max_sets
        frame = frame.iloc[:max_sets]
    # Selected by strength, drawn by signed score: the selection is "which pathways
    # matter", the order is "which way do they point".
    frame = frame.reindex(frame["_score"].sort_values(ascending=False).index)

    labels, label_notes = _row_labels(list(frame["source"]), set_labels, max_label_chars)
    notes = gsea_notes(table, alpha=alpha, dropped=dropped) + label_notes if footnotes else []

    # The size key sits above the axes and the title above that, so both need room: a
    # title placed with the default pad lands on top of the legend.
    fig, ax = row_panel_canvas(
        n_rows=len(frame),
        label_in=widest_label_in(labels, fontsize=8.5),
        data_in=_DOT_DATA_IN,
        notes=notes,
        has_title=bool(title),
        row_in=_DOT_ROW_IN,
        top_in=_LEGEND_IN + (0.30 if title else 0.0),
        figsize=figsize,
    )

    ax.axvline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    for index, (_, row) in enumerate(frame.iterrows()):
        score = float(row["_score"])
        color = _direction_color(score)
        edge = row["_edge"]
        area = _MIN_DOT_AREA if not np.isfinite(edge) else max(_MIN_DOT_AREA, _AREA_PER_GENE * edge)
        # The stem is what makes the row's magnitude readable at a glance; the dot alone
        # leaves the eye measuring distances against a gridline that is not there.
        ax.plot([0.0, score], [index, index], color=color, linewidth=1.0, zorder=2, alpha=0.55)
        significant = bool(np.isfinite(row["_fdr"]) and row["_fdr"] < alpha)
        ax.scatter(
            [score],
            [index],
            s=area,
            facecolor=color if significant else "white",
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )

    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(len(frame) - 0.5, -0.5)
    limit = float(np.nanmax(np.abs(frame["_score"].to_numpy()))) * 1.18
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(_score_axis_label(case_label, control_label), fontsize=9)
    apply_cellquorum_axis_style(ax)
    ax.tick_params(axis="y", length=0)

    # Marks outside the axes on the right, where they cannot land on a dot.
    for index, (_, row) in enumerate(frame.iterrows()):
        mark = significance_stars(row["_fdr"])
        if mark == "ns":
            mark = ""
        if bool(row["_at_limit"]):
            mark += _AT_LIMIT_MARK
        if mark:
            ax.annotate(
                mark,
                xy=(1.0, index),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=INK,
                annotation_clip=False,
            )

    _dot_size_legend(ax, frame["_edge"])
    if title:
        ax.set_title(title, fontsize=10, pad=_LEGEND_IN * 72.0)
    if notes:
        write_notes(fig, notes)
    return fig


def _score_axis_label(
    case_label: str | None,
    control_label: str | None,
    quantity: str = "Normalised enrichment score",
) -> str:
    """``"NES (→ Lymphedema / ← Normal)"``, or the bare quantity if the arms are unnamed.

    The direction of a signed score is the one thing a reader cannot recover from the
    figure, and "NES" alone leaves them guessing which sign is the disease.

    ``quantity`` is named rather than assumed because the two-arm panel is useful for any
    signed per-row score — a log fold change over the genes of one leading edge asks the same
    question of the same shape — and an axis that says "enrichment score" over fold changes is
    a mislabelled figure, which is worse than an unlabelled one.
    """
    if case_label and control_label:
        return f"{quantity}  (→ {case_label} / ← {control_label})"
    if case_label:
        return f"{quantity}  (→ {case_label})"
    return quantity


def _dot_size_legend(ax: Axes, edges: pd.Series) -> None:
    """A size key above the axes, in the same units and at the same scale as the dots."""
    from matplotlib.lines import Line2D

    sizes = _legend_sizes(edges.dropna().tolist())
    if not sizes:
        return
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=MUTED,
            markeredgecolor=MUTED,
            # Line2D wants a diameter in points where scatter wants an area.
            markersize=float(np.sqrt(max(_MIN_DOT_AREA, _AREA_PER_GENE * size))),
            label=f"{size}",
        )
        for size in sizes
    ]
    # The handle box and the gaps around it are sized from the largest marker, in font-size
    # units as matplotlib wants them. All three default to a fixed value, which is fine for a
    # line handle and wrong here: the marker whose whole point is to be large overflows a
    # 2-font-unit handle box and prints the reader's number on top of the dot it labels.
    #
    # A marker is drawn centred in the handle box and is not clipped to it, so it overhangs by
    # its radius on each side. The pads have to cover that overhang, since matplotlib measures
    # them from the nominal box rather than from the glyph that was drawn in it.
    radius_pt = 0.5 * max(float(handle.get_markersize()) for handle in handles)
    fontsize = 7.5
    ax.legend(
        handles=handles,
        title="Leading-edge genes",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=len(handles),
        frameon=False,
        fontsize=fontsize,
        title_fontsize=fontsize,
        handlelength=2.0 * radius_pt / fontsize,
        handletextpad=(radius_pt + 2.0) / fontsize,
        columnspacing=(radius_pt + 6.0) / fontsize,
        borderpad=0.0,
    )


def gsea_running_es(
    walk: pd.DataFrame,
    source: str,
    *,
    table: pd.DataFrame | None = None,
    set_labels: Mapping[str, str] | None = None,
    metric_label: str = "Ranking metric",
    case_label: str | None = None,
    control_label: str | None = None,
    title: str | None = None,
    footnotes: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw one pathway's running walk, its hit positions, and the ranking beneath them.

    This is the figure in which a GSEA result is checkable. The reported raw ES is the
    peak of the top track, the reported leading-edge size is the number of hits in the
    shaded region, and a pathway whose enrichment comes from a handful of genes at the
    very top of the ranking looks nothing like one spread down it — a distinction no
    table of scores can carry.

    Args:
        walk: The frame from ``enrichment_gsea_runningES_*.csv``, holding every drawn
            pathway. Needs ``source``, ``rank``, ``running_es``, ``hit`` and ``metric``.
        source: Which pathway to draw, as named in ``walk["source"]``.
        table: The matching ``enrichment_gsea_*.csv``, for the score and FDR in the notes.
        set_labels: Display names, keyed by the values in ``source``.
        metric_label: What the bottom track holds, e.g. ``"Signed -log10 p"``.
        case_label: Name of the arm the top of the ranking belongs to.
        control_label: Name of the arm the bottom of the ranking belongs to.
        title: Figure title. Defaults to the pathway's display name.
        footnotes: Place the derived notes under the axes.
        figsize: Overrides the computed size.

    Returns:
        The figure.

    Raises:
        ValueError: ``walk`` holds no rows for ``source``.
    """
    apply_cellquorum_theme()

    rows = walk[walk["source"].astype(str) == str(source)]
    if rows.empty:
        available = sorted({str(name) for name in walk["source"].unique()})
        raise ValueError(
            f"no running-ES walk for '{source}'; the frame holds {len(available)} "
            f"pathway(s), the first few being {available[:5]}"
        )
    rows = rows.sort_values("rank", kind="mergesort")
    rank = pd.to_numeric(rows["rank"], errors="coerce").to_numpy(dtype=float)
    running = pd.to_numeric(rows["running_es"], errors="coerce").to_numpy(dtype=float)
    metric = pd.to_numeric(rows["metric"], errors="coerce").to_numpy(dtype=float)
    hit = pd.to_numeric(rows["hit"], errors="coerce").to_numpy(dtype=float) > 0

    # The peak is the extreme the walk reaches, which is where the ES is read off. Taken
    # by |value| rather than as a maximum, so a depleted pathway is handled by the same
    # line of code as an enriched one instead of by a branch that can disagree with it.
    peak = int(np.nanargmax(np.abs(running)))
    es = float(running[peak])
    color = _direction_color(es)
    in_edge = (rank <= rank[peak]) if es > 0 else (rank >= rank[peak])
    edge_hits = int(np.count_nonzero(hit & in_edge))

    label = _pretty(str(source), set_labels)
    notes = (
        _walk_notes(
            source=str(source),
            es=es,
            n_hits=int(np.count_nonzero(hit)),
            edge_hits=edge_hits,
            peak_rank=int(rank[peak]),
            n_genes=len(rank),
            table=table,
        )
        if footnotes
        else []
    )

    fig, (ax_es, ax_hits, ax_metric) = stacked_panel_canvas(
        heights_in=_WALK_TRACK_IN,
        label_in=widest_label_in(["Running ES", "-0.66", metric_label], fontsize=8.5),
        data_in=4.0,
        notes=notes,
        has_title=True,
        figsize=figsize,
    )

    # --- the walk
    ax_es.fill_between(
        rank, 0.0, running, where=in_edge, color=color, alpha=0.13, linewidth=0.0, zorder=1
    )
    ax_es.axhline(0.0, color=MUTED, linewidth=0.8, zorder=2)
    ax_es.plot(rank, running, color=color, linewidth=1.3, zorder=3)
    ax_es.plot([rank[peak]], [es], marker="o", markersize=4.5, color=color, zorder=4)
    # The label goes past the peak, in the direction the peak is extreme in — the only
    # direction with nothing in it, since the peak is by definition the furthest the curve
    # travels. Beside it would land on the descending arm; the room is opened below.
    span = float(np.nanmax(running) - np.nanmin(running)) or 1.0
    pad = 0.10 * span
    if es > 0:
        ax_es.set_ylim(float(np.nanmin(running)) - pad, float(np.nanmax(running)) + 3 * pad)
    else:
        ax_es.set_ylim(float(np.nanmin(running)) - 3 * pad, float(np.nanmax(running)) + pad)
    # And nudged inward when the peak is against the right edge, where a left-aligned
    # label runs off the page.
    late = rank[peak] > rank[0] + 0.75 * (rank[-1] - rank[0])
    ax_es.annotate(
        f"ES = {es:+.3f}",
        xy=(rank[peak], es),
        xytext=(0, 9 if es > 0 else -15),
        textcoords="offset points",
        ha="right" if late else "center",
        fontsize=8,
        color=INK,
    )
    ax_es.set_ylabel("Running ES", fontsize=8.5)
    ax_es.set_title(title if title is not None else label, fontsize=10)

    # --- where the pathway's genes sit in the ranking
    ax_hits.vlines(rank[hit], 0.0, 1.0, color=INK, linewidth=0.4, alpha=0.75)
    ax_hits.set_ylabel("Hits", fontsize=8.5, rotation=0, ha="right", va="center")
    ax_hits.set_yticks([])
    for spine in ax_hits.spines.values():
        spine.set_visible(False)

    # --- the ranking the walk was computed on, so "the top of the list" is a value
    ax_metric.fill_between(rank, 0.0, metric, color=MUTED, alpha=0.45, linewidth=0.0)
    ax_metric.axhline(0.0, color=MUTED, linewidth=0.8)
    crossing = np.flatnonzero(metric <= 0)
    if crossing.size and crossing[0] > 0:
        ax_metric.axvline(rank[crossing[0]], color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
        ax_metric.annotate(
            f"metric crosses zero at {int(rank[crossing[0]]):,}",
            xy=(rank[crossing[0]], 0.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    ax_metric.set_ylabel(metric_label, fontsize=8.5)
    ax_metric.set_xlabel(_rank_axis_label(len(rank), case_label, control_label), fontsize=9)

    for axis in (ax_es, ax_hits, ax_metric):
        axis.set_xlim(float(rank[0]), float(rank[-1]))
    for axis in (ax_es, ax_hits):
        axis.tick_params(labelbottom=False)
    apply_cellquorum_axis_style(ax_es)
    apply_cellquorum_axis_style(ax_metric)
    ax_hits.tick_params(length=0)

    if notes:
        write_notes(fig, notes)
    return fig


def _rank_axis_label(n_genes: int, case_label: str | None, control_label: str | None) -> str:
    """``"Rank in 4,118 genes (Lymphedema-high → Normal-high)"``."""
    base = f"Rank in {n_genes:,} ranked genes"
    if case_label and control_label:
        return f"{base}  ({case_label}-high → {control_label}-high)"
    return base


def _walk_notes(
    *,
    source: str,
    es: float,
    n_hits: int,
    edge_hits: int,
    peak_rank: int,
    n_genes: int,
    table: pd.DataFrame | None,
) -> list[str]:
    """The three sentences that make the walk readable, derived from the walk itself."""
    region = (
        f"from the start of the ranking to the peak of the walk at rank {peak_rank:,} "
        f"of {n_genes:,}"
        if es > 0
        else f"from the peak of the walk at rank {peak_rank:,} to the end of the "
        f"ranking ({n_genes:,})"
    )
    notes = [
        f"Shaded: the leading edge — the {edge_hits} of {n_hits} pathway genes {region}.",
        f"ES = {es:+.3f} is the peak itself. NES divides it by the mean of the same-sign "
        "permutation scores, so the two share sign and differ in magnitude.",
    ]
    if table is None:
        return notes

    match = table[table["source"].astype(str) == source]
    if match.empty:
        return notes
    row = match.iloc[0]
    parts = []
    score = _numeric(match, "score").iloc[0]
    if np.isfinite(score):
        parts.append(f"NES = {score:+.2f}")
    fdr = _numeric(match, "padj").iloc[0]
    if np.isfinite(fdr):
        parts.append(f"FDR = {fdr:.2g}")
    permutations = _numeric(match, "permutations").iloc[0]
    if np.isfinite(permutations):
        parts.append(f"{int(permutations):,} permutations")
    if parts:
        note = ", ".join(parts) + "."
        if _flags(match, "p_at_resolution_limit").iloc[0]:
            limit = _numeric(match, "p_resolution_limit").iloc[0]
            floor = f" (p ≤ {limit:.1g})" if np.isfinite(limit) else ""
            note += (
                f" p is at the test's resolution limit{floor}, so the FDR is an upper "
                "bound rather than a measurement."
            )
        notes.append(note)
    # Reported by the stage over the whole ranked universe, in closed form; the walk here
    # is drawn independently. Naming the disagreement is better than quietly showing one.
    reported = _numeric(match, "leading_edge_size").iloc[0]
    if np.isfinite(reported) and int(reported) != edge_hits:
        notes.append(
            f"The stage reported a leading edge of {int(reported)} genes for "
            f"{source}; this walk shades {edge_hits}. The two are computed differently "
            "and should agree — a difference means the table and the figure came from "
            "different runs."
        )
    if "leading_edge" in row.index and isinstance(row["leading_edge"], str):
        genes = [gene for gene in row["leading_edge"].split(";") if gene][:8]
        if genes:
            more = int(reported) - len(genes) if np.isfinite(reported) else 0
            tail = f", and {more} more" if more > 0 else ""
            notes.append("Leading edge: " + ", ".join(genes) + tail + ".")
    return notes


def gsea_arm_comparison(
    tables: Mapping[str, pd.DataFrame],
    *,
    alpha: float = 0.05,
    sources: Sequence[str] | None = None,
    max_sets: int | None = 20,
    set_labels: Mapping[str, str] | None = None,
    max_label_chars: int | None = 48,
    case_label: str | None = None,
    control_label: str | None = None,
    score_label: str = "Normalised enrichment score",
    row_noun: str = "pathway",
    order_label: str | None = None,
    title: str | None = None,
    footnotes: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """One row per pathway, one dot per arm, ordered by how far the arms disagree.

    This is the figure a specificity argument needs. "Up in lymphatic endothelium and not in
    blood endothelium" is a claim about a difference between two contrasts, and a reader given
    two tables has to hold twenty numbers in their head to check it. Here the two dots sit on
    one axis with the gap between them drawn, so the disagreement is the thing on the page.

    The ordering is by the spread between the arms rather than by either arm's score, because
    a panel sorted by the reference arm's NES puts the strongest pathways at the top and the
    strongest pathways are usually the ones both arms agree on — the tissue-level response,
    which is exactly what a specificity figure is not about.

    Three ways an arm can fail to support a pathway are drawn differently, because a table
    conflates the last two and the difference matters:

    * significant with the opposite sign — a filled dot on the other side of zero, which is
      the strongest form of dissociation and stronger than mere absence;
    * tested and not significant — an open dot, so the reader can see the effect was estimated
      and how big it was;
    * never tested — no dot, and a note naming the pathways it happened to. Size filters are
      applied to each set's *detected* membership, so two arms that rank different numbers of
      genes do not test quite the same collection. An untested set is not a null result, and
      in a table its row is simply missing, which is indistinguishable from a null by eye.

    Args:
        tables: One GSEA table per arm, keyed by the arm's display name. Drawn in the order
            the mapping gives, which is the order the legend and the notes use.
        alpha: FDR threshold separating a filled dot from an open one.
        sources: Draw exactly these pathways, in this order, overriding both the
            union-of-significant selection and ``max_sets``. For a chosen panel — the block a
            manuscript argues about — rather than the top of a ranking.
        max_sets: Cap on the automatic selection (pathways significant in at least one arm,
            largest between-arm spread first). Ignored when ``sources`` is given.
        set_labels: Display names, keyed by the values in ``source``.
        max_label_chars: As :func:`gsea_dotplot`.
        case_label: Name of the arm a positive score points to, for the axis label.
        control_label: Name of the arm a negative score points to.
        score_label: The quantity in ``score``, for the axis. The panel's shape suits any
            signed per-row score in two arms — the log fold changes of one leading edge's
            genes, for instance — and the axis has to say which one it is drawing.
        row_noun: What one row is, for the notes. ``"gene"`` when the rows are genes.
        order_label: What the row order is, when ``sources`` supplies one. A supplied list is
            often ordered by something the function cannot see — a fold change in one arm, or
            the sequence an argument is made in — and the note can only report an ordering
            the caller names. Without it the note says the rows were given, and claims
            nothing about why they are in that sequence.
        title: Figure title.
        footnotes: Place the notes under the axes.
        figsize: Overrides the size derived from the number of rows.

    Returns:
        The figure.

    Raises:
        ValueError: If fewer than two arms are given, or if no pathway survives selection.
    """
    apply_cellquorum_theme()
    if len(tables) < 2:
        raise ValueError(
            f"an arm comparison needs at least two tables, got {len(tables)} "
            f"({', '.join(tables) or 'none'})"
        )

    arms = list(tables)
    scores: dict[str, pd.Series] = {}
    fdrs: dict[str, pd.Series] = {}
    for arm, table in tables.items():
        if "source" not in table.columns:
            raise ValueError(f"the {arm} table has no 'source' column")
        indexed = table.set_index(table["source"].astype(str))
        scores[arm] = _numeric(indexed, "score")
        fdrs[arm] = _numeric(indexed, "padj")

    if sources is not None:
        chosen = [str(name) for name in sources]
        missing = [name for name in chosen if not any(name in s.index for s in scores.values())]
        if missing:
            raise ValueError(
                f"no arm has a row for {', '.join(missing[:4])}"
                + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
            )
        dropped = 0
    else:
        significant = {
            name
            for arm in arms
            for name, fdr in fdrs[arm].items()
            if np.isfinite(fdr) and fdr < alpha
        }
        if not significant:
            raise ValueError(f"no pathway is significant at FDR < {alpha:g} in any arm")
        spread = {
            name: float(
                np.nanmax([scores[arm].get(name, np.nan) for arm in arms])
                - np.nanmin([scores[arm].get(name, np.nan) for arm in arms])
            )
            for name in significant
        }
        ordered = sorted(spread, key=lambda name: (-spread[name], name))
        chosen = ordered if max_sets is None else ordered[:max_sets]
        dropped = len(ordered) - len(chosen)

    labels, label_notes = _row_labels(chosen, set_labels, max_label_chars)
    colors = _arm_colors(arms)
    untested = {
        arm: [name for name in chosen if not np.isfinite(scores[arm].get(name, np.nan))]
        for arm in arms
    }
    notes = (
        _comparison_notes(
            arms,
            alpha,
            dropped,
            untested,
            case_label,
            control_label,
            supplied=sources is not None,
            row_noun=row_noun,
            order_label=order_label,
        )
        + label_notes
        if footnotes
        else []
    )

    fig, ax = row_panel_canvas(
        n_rows=len(chosen),
        label_in=widest_label_in(labels, fontsize=8.5),
        data_in=_DOT_DATA_IN,
        notes=notes,
        has_title=bool(title),
        row_in=_COMPARISON_ROW_IN,
        top_in=_LEGEND_IN + (0.30 if title else 0.0),
        figsize=figsize,
    )
    ax.axvline(0.0, color=MUTED, linewidth=0.8, zorder=1)

    for index, name in enumerate(chosen):
        drawn = [(arm, scores[arm].get(name, np.nan)) for arm in arms]
        finite = [value for _, value in drawn if np.isfinite(value)]
        # The connector is what makes the row a comparison rather than two marks that happen
        # to share a line. Drawn under the dots, and only when there are two ends to join.
        if len(finite) > 1:
            ax.plot(
                [min(finite), max(finite)],
                [index, index],
                color=MUTED,
                linewidth=1.0,
                zorder=2,
                alpha=0.7,
            )
        for arm, value in drawn:
            if not np.isfinite(value):
                continue
            fdr = fdrs[arm].get(name, np.nan)
            significant = bool(np.isfinite(fdr) and fdr < alpha)
            ax.scatter(
                [value],
                [index],
                s=_COMPARISON_DOT_AREA,
                facecolor=colors[arm] if significant else "white",
                edgecolor=colors[arm],
                linewidth=1.2,
                zorder=3,
            )

    ax.set_yticks(np.arange(len(chosen)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(len(chosen) - 0.5, -0.5)
    every = [
        value
        for arm in arms
        for value in (scores[arm].get(name, np.nan) for name in chosen)
        if np.isfinite(value)
    ]
    limit = float(np.nanmax(np.abs(every))) * 1.18
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(_score_axis_label(case_label, control_label, score_label), fontsize=9)
    apply_cellquorum_axis_style(ax)
    ax.tick_params(axis="y", length=0)
    _arm_legend(ax, arms, colors)
    if title:
        ax.set_title(title, fontsize=10, pad=_LEGEND_IN * 72.0)
    if notes:
        write_notes(fig, notes)
    return fig


def _arm_colors(arms: Sequence[str]) -> dict[str, str]:
    """One colour per arm, from the palette whose all-pairs CVD safety was measured.

    Two arms take the case/control pair the rest of the module already uses, so a two-arm
    comparison sits beside a single-arm dotplot without recolouring the reader's expectations.
    Beyond two, the categorical palette's measured-safe core.
    """
    if len(arms) == 2:
        return {arms[0]: LE_RED, arms[1]: NORMAL_BLUE}
    return {
        arm: CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)] for index, arm in enumerate(arms)
    }


def _arm_legend(ax: Axes, arms: Sequence[str], colors: Mapping[str, str]) -> None:
    """A key naming the arms, in the order they were given."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[arm],
            markeredgecolor=colors[arm],
            markersize=float(np.sqrt(_COMPARISON_DOT_AREA)),
            label=arm,
        )
        for arm in arms
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.0,
    )


def _order_note(row_noun: str, order_label: str | None) -> str:
    """What put the rows in this sequence, or that the caller chose it and said no more.

    The note has to be true of the figure in front of the reader. "Not a ranking" was wrong
    the first time a caller supplied a list ordered by fold change: the ordering was real and
    the footnote denied it.
    """
    if order_label:
        return f"Rows are a chosen set of {row_noun}s, ordered by {order_label}."
    return f"Rows are a chosen set of {row_noun}s, drawn in the order given."


def _comparison_notes(
    arms: Sequence[str],
    alpha: float,
    dropped: int,
    untested: Mapping[str, Sequence[str]],
    case_label: str | None,
    control_label: str | None,
    *,
    supplied: bool,
    row_noun: str,
    order_label: str | None = None,
) -> list[str]:
    """What a dot, an open dot, and a missing dot each mean, and where the rows came from.

    The ordering note is a claim about the figure, so it is only made when the figure earned
    it. A caller-supplied ``sources`` list is drawn in the caller's order, and telling the
    reader it is sorted by disagreement when it is not is worse than saying nothing.
    """
    direction = (
        f" A dot right of zero is up in {case_label}, left of zero up in {control_label}."
        if case_label and control_label
        else ""
    )
    notes = [
        f"One row per {row_noun}, one dot per arm ({', '.join(arms)}); the bar joins them. "
        f"Filled = FDR < {alpha:g} in that arm, open = tested and not significant." + direction,
        _order_note(row_noun, order_label)
        if supplied
        else (
            "Ordered by the spread between the arms, largest first — not by either arm's "
            f"score, since the strongest {row_noun}s are usually the ones both arms agree on."
        ),
    ]
    absent = {arm: list(names) for arm, names in untested.items() if names}
    if absent:
        for arm, names in absent.items():
            shown = ", ".join(_pretty(name, None) for name in names[:3])
            more = f", and {len(names) - 3} more" if len(names) > 3 else ""
            notes.append(
                f"No dot for {arm} on {shown}{more}: the {row_noun} was never tested in "
                "that arm, which is not the same as tested and null. Size filters apply to "
                "each set's detected membership, and the arms rank different numbers of genes."
            )
    if dropped:
        notes.append(
            f"{dropped} further {row_noun}(s) significant in at least one arm are not "
            f"drawn; every {row_noun} is in the table."
        )
    return notes


__all__ = ["gsea_arm_comparison", "gsea_dotplot", "gsea_notes", "gsea_running_es"]
