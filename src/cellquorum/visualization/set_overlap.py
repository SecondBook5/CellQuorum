"""Gene-set overlap drawn so the reader can see whether the panel is a panel.

A study that scores eight programs and reports eight effects is making an implicit
claim — that there are eight things being measured. The two figures here test that
claim, and they are the figure forms of the two tables such analyses normally print.

:func:`set_overlap_upset` replaces the exclusive-membership table ("69 genes are in
mitotic spindle only, 5 are shared with apical junction"). As a table that is a list
of numbers with no shape; as an UpSet plot the shape is the point — a program whose
bar is almost entirely shared with its neighbour is visibly not an independent
readout, and no arithmetic is needed to see it.

:func:`set_overlap_matrix` replaces the pairwise-similarity table, and it exists
because that table is where the mistake happens. Similarity and significance are
different quantities: a Jaccard of 0.052 can be strong evidence of shared membership
while a Jaccard of 0.047 is nothing, because the p-value depends on the set sizes and
the universe. A heatmap coloured by similarity alone therefore *ranks pairs wrong*, so
the significance is drawn into the same cell as the coefficient — a colour and a mark
that disagree is the honest picture, and the reference table this replaces had exactly
that disagreement in it without saying so.

Both figures read from the frames :mod:`cellquorum.stats.gene_set_overlap` writes, not
from the sets themselves where a table exists, so a figure cannot disagree with the
CSV beside it. Both state the universe size on the figure: the p-values are a function
of it, and a concordance panel without it is not interpretable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cellquorum.stats.gene_set_overlap import exclusive_combinations, set_sizes
from cellquorum.visualization.figstyle import (
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    significance_stars,
)
from cellquorum.visualization.measured_layout import (
    ABSENT,
    INK,
    square_matrix_canvas,
    widest_label_in,
    wrap_notes,
    write_notes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

# Exclusive members and shared members are the two halves of the question the figure
# is asked, so they are the two tones used throughout — in the sidebar's stacked bars,
# in the combination bars, and in the matrix dots. Drawn from the house colours rather
# than a categorical palette because this is one contrast repeated, not a set of
# categories, and a categorical palette here would imply the combinations are groups.
_EXCLUSIVE = "#24608F"
_SHARED = "#C45A5A"

#: Colormap for the similarity matrix. ColorBrewer sequential, so it is monotone in
#: lightness and safe under every form of colour-vision deficiency by construction —
#: which a hand-built white-to-house-blue ramp would not be.
_SIMILARITY_CMAP = "Blues"

_VALUE_LABELS = {
    "jaccard": "Jaccard index",
    "overlap_coefficient": "Overlap coefficient",
    "fold_enrichment": "Fold enrichment over chance",
    "intersection": "Shared genes",
}


def _pretty(name: str, labels: Mapping[str, str] | None) -> str:
    if labels and name in labels:
        return labels[name]
    return name.replace("_", " ")


def set_overlap_upset(
    sets: Mapping[str, Iterable[str]],
    *,
    min_size: int = 1,
    max_combinations: int | None = None,
    set_labels: Mapping[str, str] | None = None,
    sort_sets: str = "exclusivity",
    title: str | None = None,
    universe: Iterable[str] | None = None,
    element_noun: str = "genes",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Draw the exclusive-membership structure of a panel of sets as an UpSet plot.

    Four layout choices worth knowing about:

    * **Columns are ordered by degree, then by size** — everything belonging to
      exactly one set first, largest first, then the pairs, and so on. This is the
      order :func:`~cellquorum.stats.gene_set_overlap.exclusive_combinations` writes,
      so the figure and its CSV can be read against each other row by row. The
      commoner size-descending order interleaves degrees and makes that impossible.
    * **Rows are ordered by exclusivity by default**, least exclusive at the top,
      because that is the finding a reader is here for: the set that turns out to be
      a restatement of its neighbours belongs at the top of the figure, not buried
      in the middle of a size ranking. Pass ``sort_sets="size"`` for the
      conventional order.
    * **The sidebar bar is split** into the part of each set that is exclusive to it
      and the part shared with any other set, because "how much of this program is
      its own" is the question the panel is being asked and it should not need a
      second figure.
    * **Truncation is reported, never silent.** If ``max_combinations`` drops
      columns, a note says how many and how many elements were in them.

    A set with *no* exclusive members occupies no column of its own, which is easy to
    miss in a wide matrix, so its row label is marked ``‡`` and footnoted. That is not
    a decoration: a set every one of whose members belongs to another set cannot be an
    independent readout of anything, whatever its scores go on to show.

    Args:
        sets: Name to members.
        min_size: Drop combinations occupied by fewer than this many elements.
        max_combinations: Keep at most this many columns, taken in the sort order
            above. ``None`` keeps everything observed.
        set_labels: Display names, keyed by the names in ``sets``.
        sort_sets: Row order — ``"exclusivity"`` (least exclusive first),
            ``"size"`` (largest first), or ``"given"`` to keep the order of ``sets``.
        title: Figure title.
        universe: Optional; when given, sets are counted after restricting to it and
            the number of members falling outside is footnoted. Pass the same
            universe used for :func:`~cellquorum.stats.gene_set_overlap.set_overlap_tests`
            so the two figures describe the same sets.
        element_noun: What the elements are, for the axis labels.
        figsize: Overrides the size derived from the number of sets and columns. The
            margins are derived from the same measurements either way, so a set with
            a long name keeps its label room.

    Returns:
        The figure.

    Raises:
        ValueError: No sets were given, ``min_size`` left no combinations to draw, or
            ``sort_sets`` is not one of the three orders.
    """
    apply_cellquorum_theme()

    combos = exclusive_combinations(sets, min_size=min_size, list_elements=False)
    if combos.empty:
        raise ValueError(
            f"no combination has at least {min_size} {element_noun}; "
            "nothing to draw (lower min_size, or check the sets are not empty)"
        )

    dropped_columns, dropped_elements = 0, 0
    if max_combinations is not None and len(combos) > max_combinations:
        dropped = combos.iloc[max_combinations:]
        dropped_columns, dropped_elements = len(dropped), int(dropped["size"].sum())
        combos = combos.iloc[:max_combinations]

    sizes = _order_sets(set_sizes(sets, universe=universe), sort_sets)
    row_order = list(sizes["set"])
    exclusive_counts = dict(zip(sizes["set"], sizes["exclusive"], strict=True))
    memberships = [set(str(part) for part in label.split(" & ")) for label in combos["combination"]]

    labels = [_row_label(name, set_labels, exclusive_counts[name]) for name in row_order]
    notes = _upset_notes(
        sizes,
        universe=universe,
        element_noun=element_noun,
        dropped_columns=dropped_columns,
        dropped_elements=dropped_elements,
        set_labels=set_labels,
    )

    fig, ax_bars, ax_matrix, ax_sizes = _upset_canvas(
        n_rows=len(row_order),
        n_cols=len(combos),
        label_in=widest_label_in(labels),
        notes=notes,
        has_title=title is not None,
        figsize=figsize,
    )

    _draw_combination_bars(ax_bars, combos, memberships, element_noun)
    _draw_membership_matrix(ax_matrix, row_order, labels, memberships)
    _draw_size_sidebar(ax_sizes, sizes, element_noun)

    if title:
        fig.suptitle(title, x=0.01, y=0.985, ha="left", va="top", fontsize=11, color=INK)
    write_notes(fig, notes)
    return fig


_SET_ORDERS = ("exclusivity", "size", "given")


def _order_sets(sizes: pd.DataFrame, sort_sets: str) -> pd.DataFrame:
    if sort_sets not in _SET_ORDERS:
        raise ValueError(f"sort_sets must be one of {_SET_ORDERS}, got {sort_sets!r}")
    if sort_sets == "given":
        return sizes.reset_index(drop=True)
    if sort_sets == "size":
        return sizes.sort_values("size", ascending=False, kind="stable").reset_index(drop=True)
    # Least exclusive first, and among ties the larger set — a big set that is wholly
    # contained in others is a worse problem than a small one.
    return sizes.sort_values(
        ["fraction_exclusive", "size"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)


def _row_label(name: str, set_labels: Mapping[str, str] | None, exclusive: int) -> str:
    label = _pretty(name, set_labels)
    return f"{label} ‡" if exclusive == 0 else label


def _upset_canvas(
    *,
    n_rows: int,
    n_cols: int,
    label_in: float,
    notes: Sequence[str],
    has_title: bool,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, Axes, Axes, Axes]:
    """Lay the four regions out in inches, reserving a gutter for the row labels.

    The row labels belong to the matrix and are drawn leftward from it, into whatever
    space happens to be there — which, in a two-column grid, is the sidebar's own
    axes. The sidebar's opaque background then paints over the middle of every long
    label, leaving the ends visible and the figure looking like a rendering bug. So
    the grid has a third, empty column whose width is the longest label's, and the
    labels draw into that.

    Everything is sized in inches and converted to figure fractions at the end,
    because ``tight_layout`` cannot see text that overhangs an axes and would size
    the gutter to nothing.
    """
    import matplotlib.pyplot as plt

    # The measured label width plus the tick pad the labels are offset by.
    label_in += 0.14
    side_in = 1.15
    matrix_in = max(2.6, 0.36 * n_cols)
    bars_in = max(1.5, 0.26 * n_rows)
    rows_in = max(1.4, 0.34 * n_rows)

    left_in, right_in = 0.60, 0.30
    # The legend sits above the bar panel, so the room over it is never zero.
    top_in = (0.44 if has_title else 0.14) + 0.24

    width = left_in + side_in + label_in + matrix_in + right_in
    # Wrapped to the same width ``write_notes`` will wrap to, so the room reserved
    # below the grid is the room the notes take. Counting the notes instead reserves
    # one line for a caveat that renders as three.
    bottom_in = 0.58 + 0.20 * len(wrap_notes(notes, width_in=width * 0.98))
    height = top_in + bars_in + rows_in + bottom_in
    if figsize is not None:
        width, height = figsize

    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(side_in, label_in, matrix_in),
        height_ratios=(bars_in, rows_in),
        wspace=0.0,
        hspace=0.06,
        left=left_in / width,
        right=1.0 - right_in / width,
        top=1.0 - top_in / height,
        bottom=bottom_in / height,
    )
    ax_bars = fig.add_subplot(grid[0, 2])
    ax_matrix = fig.add_subplot(grid[1, 2], sharex=ax_bars)
    ax_sizes = fig.add_subplot(grid[1, 0], sharey=ax_matrix)
    return fig, ax_bars, ax_matrix, ax_sizes


def _draw_combination_bars(
    ax: Axes, combos: pd.DataFrame, memberships: Sequence[set[str]], element_noun: str
) -> None:
    """The count for each occupied combination, exclusive columns in their own tone."""
    positions = np.arange(len(combos))
    heights = combos["size"].to_numpy(dtype=float)
    colors = [_EXCLUSIVE if len(members) == 1 else _SHARED for members in memberships]
    ax.bar(positions, heights, color=colors, width=0.62, linewidth=0)

    headroom = max(heights.max(), 1.0)
    for position, height in zip(positions, heights, strict=True):
        ax.text(
            position,
            height + headroom * 0.03,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK,
        )
    ax.set_ylim(0, headroom * 1.16)
    ax.set_ylabel(element_noun.capitalize(), fontsize=9)
    apply_cellquorum_axis_style(ax)
    # After the house style, not before: it resets tick length on both axes, so
    # hiding a tick first and styling second puts the tick back.
    ax.tick_params(axis="x", length=0, labelbottom=False)
    # Above the panel rather than inside its upper-right corner: the bars are sorted
    # by size, so the tallest are on the left and the rightmost bars are short — but
    # not reliably. A legend inside the axes lands on whichever count label happens to
    # be under it, and which one that is changes with the data.
    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="", color=_EXCLUSIVE, label="In one set only"),
            Line2D([], [], marker="s", linestyle="", color=_SHARED, label="Shared"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
    )


def _draw_membership_matrix(
    ax: Axes,
    row_order: Sequence[str],
    labels: Sequence[str],
    memberships: Sequence[set[str]],
) -> None:
    """Which sets each column belongs to: filled dots joined by a stem."""
    for index, members in enumerate(memberships):
        rows = [position for position, name in enumerate(row_order) if name in members]
        if len(rows) > 1:
            ax.plot([index, index], [min(rows), max(rows)], color=INK, linewidth=1.1, zorder=1)
        absent = [position for position in range(len(row_order)) if position not in rows]
        ax.scatter([index] * len(absent), absent, s=26, color=ABSENT, zorder=2)
        color = _EXCLUSIVE if len(members) == 1 else _SHARED
        ax.scatter([index] * len(rows), rows, s=34, color=color, zorder=3)

    ax.set_xlim(-0.6, len(memberships) - 0.4)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # Faint banding, which is what makes a wide matrix readable across a row.
    for position in range(0, len(row_order), 2):
        ax.axhspan(position - 0.5, position + 0.5, color="#f4f4f4", zorder=0)


def _draw_size_sidebar(ax: Axes, sizes: pd.DataFrame, element_noun: str) -> None:
    """Each set's total, split into the exclusive and the shared part."""
    positions = np.arange(len(sizes))
    exclusive = sizes["exclusive"].to_numpy(dtype=float)
    shared = sizes["shared"].to_numpy(dtype=float)
    ax.barh(positions, exclusive, color=_EXCLUSIVE, height=0.62, linewidth=0)
    ax.barh(positions, shared, left=exclusive, color=_SHARED, height=0.62, linewidth=0)

    totals = exclusive + shared
    widest = max(totals.max(), 1.0)
    for position, total in zip(positions, totals, strict=True):
        ax.text(
            total + widest * 0.03,
            position,
            f"{int(total)}",
            # The x-axis is reversed, so a larger data value sits further *left* on
            # screen and "ha=left" would grow the label back over the bar it labels.
            ha="right",
            va="center",
            fontsize=7.5,
            color=INK,
        )
    # Bars grow leftward from the matrix, the conventional UpSet sidebar.
    ax.set_xlim(widest * 1.22, 0)
    ax.set_xlabel(f"Set size ({element_noun})", fontsize=9)
    apply_cellquorum_axis_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelleft=False)


def _upset_notes(
    sizes: pd.DataFrame,
    *,
    universe: Iterable[str] | None,
    element_noun: str,
    dropped_columns: int,
    dropped_elements: int,
    set_labels: Mapping[str, str] | None = None,
) -> list[str]:
    notes: list[str] = []

    wholly_shared = sizes[sizes["exclusive"] == 0]
    if not wholly_shared.empty:
        named = ", ".join(_pretty(str(name), set_labels) for name in wholly_shared["set"])
        notes.append(
            f"‡ {named}: every member is also in another set, so this is not an "
            "independent readout — it has no column of its own above."
        )

    if universe is not None:
        universe_size = len({str(element) for element in universe})
        notes.append(f"Counted against {universe_size:,} {element_noun} in the universe.")
        outside = sizes[sizes["outside_universe"] > 0]
        if not outside.empty:
            detail = ", ".join(
                f"{_pretty(str(row['set']), set_labels)} {int(row['outside_universe'])}"
                for _, row in outside.iterrows()
            )
            notes.append(
                f"Members outside the universe, excluded from every count ({detail}) — "
                "a large share usually means a gene-naming mismatch, not biology."
            )
    if dropped_columns:
        notes.append(
            f"{dropped_columns} further combination(s) holding {dropped_elements} "
            f"{element_noun} are not drawn; the full set is in the accompanying table."
        )
    return notes


def set_overlap_matrix(
    overlaps: pd.DataFrame,
    *,
    value: str = "jaccard",
    alpha: float = 0.05,
    set_order: Sequence[str] | None = None,
    set_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    footnotes: bool = True,
) -> Figure:
    """
    Draw a pairwise-overlap table as a matrix with the significance in the cell.

    Each drawn cell carries three things, deliberately: the colour is the similarity
    coefficient, the number is the count of shared elements, and the stars are the
    FDR. A pair can be coloured pale and starred, or coloured strongly and bare, and
    every such disagreement is a pair a similarity-only heatmap would rank wrong.
    The count is there because it is the quantity a reader can act on — "these two
    modules share two genes" is checkable, a Jaccard of 0.056 is not — and because on
    a panel with one dominant pair the colour scale compresses everything else into
    indistinguishable pale blues, where the printed count still reads.

    Args:
        overlaps: The frame from
            :func:`~cellquorum.stats.gene_set_overlap.set_overlap_tests`, or a CSV of
            it read back. Only the lower triangle is drawn, and the diagonal is left
            empty — a set overlaps itself perfectly, so drawing it would take the top
            of the colour scale and flatten every real pair.
        value: Which column to colour by.
        alpha: FDR threshold below which a cell is marked significant.
        set_order: Row and column order. Defaults to first appearance in
            ``overlaps``, which preserves the order the sets were passed in.
        set_labels: Display names.
        title: Panel title, drawn left-aligned.
        figsize: Overrides the size derived from the number of sets and the length of
            their names. The margins come from the same measurements either way.
        ax: Draw into an existing axes instead of a new figure, for composing panels.
        footnotes: Write the notes under the figure. Turn off when composing panels
            and place them once for the whole figure.

    Returns:
        The figure the matrix was drawn into.

    Raises:
        ValueError: The frame is empty, or ``value`` is not one of its columns.
    """
    if overlaps.empty:
        raise ValueError("the overlap table is empty; there is nothing to draw")
    if value not in overlaps.columns:
        raise ValueError(
            f"{value!r} is not a column of the overlap table; "
            f"available: {sorted(overlaps.columns)}"
        )

    apply_cellquorum_theme()

    if set_order is None:
        set_order = list(dict.fromkeys([*overlaps["set_a"], *overlaps["set_b"]]))
    index = {name: position for position, name in enumerate(set_order)}
    n = len(set_order)
    labels = [_pretty(str(name), set_labels) for name in set_order]

    values = np.full((n, n), np.nan)
    marks: dict[tuple[int, int], str] = {}
    counts: dict[tuple[int, int], int] = {}
    untested: list[str] = []
    for _, row in overlaps.iterrows():
        a, b = str(row["set_a"]), str(row["set_b"])
        if a not in index or b not in index:
            continue
        # Lower triangle only, whichever way round the pair was written.
        i, j = max(index[a], index[b]), min(index[a], index[b])
        values[i, j] = float(row[value])
        shared = int(row["intersection"]) if "intersection" in overlaps.columns else 0
        if shared > 0:
            counts[i, j] = shared
        fdr = float(row["fdr"]) if "fdr" in overlaps.columns else np.nan
        if not np.isfinite(fdr):
            marks[i, j] = "?"
            untested.append(f"{_pretty(a, set_labels)}/{_pretty(b, set_labels)}")
        elif fdr < alpha:
            marks[i, j] = significance_stars(fdr)

    owns_figure = ax is None
    notes = set_overlap_notes(overlaps, alpha=alpha, untested=untested) if footnotes else []
    if owns_figure:
        fig, ax, cax = square_matrix_canvas(
            n=n,
            label_in=widest_label_in(labels),
            notes=notes,
            has_title=title is not None,
            figsize=figsize,
        )
    else:
        fig, cax = ax.get_figure(), None

    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap=_SIMILARITY_CMAP, vmin=0.0, aspect="equal")
    image.cmap.set_bad("#ffffff")

    ceiling = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    for cell in sorted(set(counts) | set(marks)):
        i, j = cell
        # White on the dark end of the ramp, ink on the pale end.
        color = "#ffffff" if values[i, j] > ceiling * 0.6 else INK
        count, mark = counts.get(cell), marks.get(cell)
        # Count above, stars below, each its own artist so a cell can carry one
        # without the other: an untested pair has no count to print, and a real
        # overlap that misses the threshold has no stars.
        if count is not None:
            ax.text(
                j,
                i - (0.16 if mark else 0.0),
                f"{count}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
        if mark:
            ax.text(
                j,
                i + (0.19 if count is not None else 0.0),
                mark,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)

    bar = (
        fig.colorbar(image, cax=cax)
        if cax is not None
        else fig.colorbar(image, ax=ax, fraction=0.042, pad=0.03)
    )
    bar.set_label(_VALUE_LABELS.get(value, value.replace("_", " ")), fontsize=9)
    bar.outline.set_visible(False)

    if notes:
        write_notes(fig, notes)
    return fig


def set_overlap_notes(
    overlaps: pd.DataFrame, *, alpha: float = 0.05, untested: Sequence[str] | None = None
) -> list[str]:
    """
    The notes a pairwise-overlap panel has to carry, derived from the table.

    Public and pure so a composed figure can place them once, and so the notes on a
    figure can be tested against the frame they came from rather than against a
    rendering.

    Args:
        overlaps: The frame from
            :func:`~cellquorum.stats.gene_set_overlap.set_overlap_tests`.
        alpha: The FDR threshold the marks used.
        untested: Pair labels whose test was not attempted, if the caller has already
            worked them out; otherwise they are recovered from a missing ``fdr``.

    Returns:
        Sentences, in reading order. Empty only if the table records nothing that
        needs saying, which for an overlap table means every pair was testable and
        no set lost members to the universe.
    """
    notes: list[str] = []

    if "universe_size" in overlaps.columns:
        universes = sorted({int(size) for size in overlaps["universe_size"].dropna().unique()})
        if len(universes) == 1:
            notes.append(
                f"One-sided hypergeometric test against {universes[0]:,} testable genes, "
                f"Benjamini–Hochberg across the {len(overlaps)} pairs; "
                f"* FDR < {alpha:g}, ** < 0.01, *** < 0.001."
            )
        elif universes:
            # Pairs tested against different universes cannot be compared, and the
            # FDR across them is not a family. Say so rather than print one number.
            notes.append(
                f"Pairs were tested against different universes ({universes}); their "
                "p-values are not comparable and the correction is not over one family."
            )

    if untested is None and "fdr" in overlaps.columns:
        missing = overlaps[~np.isfinite(overlaps["fdr"].astype(float))]
        untested = [f"{row['set_a']}/{row['set_b']}" for _, row in missing.iterrows()]
    if untested:
        notes.append(
            f"Marked ? and not tested: {', '.join(untested)} — a set with no members "
            "inside the universe has nothing to draw from; check the gene naming."
        )

    for side in ("dropped_a", "dropped_b"):
        if side in overlaps.columns and (overlaps[side].fillna(0) > 0).any():
            notes.append(
                "Some set members were outside the universe and were excluded before "
                "testing; per-pair counts are in the table's dropped_a/dropped_b columns."
            )
            break

    return notes


__all__ = ["set_overlap_matrix", "set_overlap_notes", "set_overlap_upset"]
