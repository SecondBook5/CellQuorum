"""Layout measured in inches, for figures whose text lives outside their axes.

``tight_layout`` and ``constrained_layout`` solve for the space an *axes* needs. They
cannot see figure-level text, text drawn outside an axes, or a square aspect that will
be imposed at draw time — and those three are exactly what a publication panel is made
of. Every figure in this package that carries row labels in a gutter, footnotes under
the axes, or an ``aspect="equal"`` matrix has therefore had the same three bugs:

* **A gutter estimated from a character count is wrong by a constant factor.** The
  factor is the ratio of the theme's font to whatever font the estimate assumed, so the
  same constant over-reserves 40% of the width on one panel and clips the labels on the
  next. :func:`widest_label_in` lays the text out and measures it instead.
* **``aspect="equal"`` is applied after the layout is solved.** ``tight_layout``
  reserves room for the rotated column labels, the aspect then shrinks the axes to a
  square and re-centres it, the labels lift away from the space reserved for them, and
  a band of dead figure is left behind. ``bbox_inches="tight"`` does not help: the space
  is genuinely occupied by the layout. :func:`square_matrix_canvas` sizes the axes to an
  exact square up front, so the aspect has nothing left to shrink.
* **Footnotes placed in figure fractions move when the figure resizes.** The same three
  notes crowd together on a short figure and drift apart on a tall one.
  :func:`write_notes` stacks them at a fixed inch pitch.
* **A footnote wider than the figure silently resizes the figure.** Every panel here is
  saved with ``bbox_inches="tight"``, so a one-line caveat 6 inches wide under a
  4-inch panel does not overflow — it stretches the saved figure to 6 inches and leaves
  a two-inch dead band beside the colourbar. The aspect ratio of the figure then
  depends on how many words the caption needed, which is not a layout. So the notes are
  wrapped to the figure's own width (:func:`wrap_notes`), and a matrix panel spends a
  little of the width its notes want on a larger data area before wrapping the rest.

The two panel shapes differ on that last point, on purpose. A matrix
(:func:`square_matrix_canvas`) is read on its own, so it may grow its data area to use
width the caption wanted. A row-per-item panel (:func:`row_panel_canvas`) is read against
the panel beside it, so its data area is a fixed width: a coefficient of 0.6 has to be the
same length in every figure that draws one, or the comparison those figures exist for is
not actually available to the reader.

Nothing here is study-specific or figure-specific; it is the arithmetic that any
inch-measured panel needs, kept in one place so the modules that need it do not each carry
their own copy.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Near-black rather than pure black for body text and marks: pure black against white
#: is harsher than print needs, and the difference survives greyscale reproduction.
INK = "#1a1a1a"
#: Secondary text — footnotes, axis sub-labels, anything that must be legible without
#: competing with the data.
MUTED = "#6b6b6b"
#: Absent/not-applicable fill, light enough to read as "no value" rather than "low value".
ABSENT = "#d9d9d9"

#: Vertical room one footnote line occupies, in inches.
NOTE_LINE_IN = 0.20

#: Fallback width per character, in inches at 9pt, for the rare backend that cannot hand
#: out a renderer. A deliberate over-estimate: too much label room costs whitespace, too
#: little costs the label.
CHAR_IN = 0.068

#: sin(40°) — rotated column labels are laid over at 40 degrees, so the vertical room
#: they need is their rendered width times this.
_ROTATED_LABEL_RISE = 0.643

#: Fraction of the figure width a footnote may occupy. Notes start one percent in from
#: the left edge; the rest is a matching right margin.
_NOTE_WIDTH_FRACTION = 0.98

#: Shortest line :func:`wrap_notes` will wrap to. A figure narrow enough to force fewer
#: characters than this would turn a caveat into a column of words, which is less
#: legible than letting it run a little wide.
_MIN_WRAP_CHARS = 28

#: Hanging indent on a wrapped note's continuation lines. Wide enough to see, narrow
#: enough that the note still starts where the others do.
_HANGING_INDENT = "   "


def measure_labels_in(labels: Sequence[str], fontsize: float = 9.0) -> list[float]:
    """Rendered width of each label in inches, measured rather than estimated.

    The gutter a label sits in has to be sized before the axes exist, so the text is
    laid out on one throwaway canvas and measured. Only a backend that cannot produce a
    renderer falls back to counting characters.

    Args:
        labels: The strings to measure.
        fontsize: Points, matching whatever the caller will draw the labels at.

    Returns:
        One width per label, in inches. Tick pads and margins are the caller's.
    """
    import matplotlib.pyplot as plt

    labels = list(labels)
    if not labels:
        return []
    probe = plt.figure(figsize=(1.0, 1.0))
    try:
        renderer = probe.canvas.get_renderer()
    except AttributeError:
        plt.close(probe)
        scale = fontsize / 9.0
        return [CHAR_IN * scale * len(label) for label in labels]
    try:
        return [
            probe.text(0, 0, label, fontsize=fontsize).get_window_extent(renderer).width / probe.dpi
            for label in labels
        ]
    finally:
        plt.close(probe)


def widest_label_in(labels: Sequence[str], fontsize: float = 9.0) -> float:
    """The widest label's rendered width in inches. See :func:`measure_labels_in`."""
    return max(measure_labels_in(labels, fontsize=fontsize), default=0.0)


def wrap_notes(notes: Sequence[str], *, width_in: float, fontsize: float = 7.5) -> list[str]:
    """Break each note onto as many lines as the figure is wide.

    The character budget is derived from each note's *own* measured width rather than
    from a global characters-per-inch constant: a note full of narrow glyphs fits more
    characters than one full of wide ones, and a constant is wrong for both.

    Args:
        notes: The notes, in reading order.
        width_in: Room available for one line.
        fontsize: Points the notes will be drawn at.

    Returns:
        The lines, in reading order. A note that already fits is returned untouched, so
        a caller can compare the two lists to see whether anything wrapped at all.
        Continuation lines carry a hanging indent: without it a caveat that wrapped onto
        two lines is indistinguishable from two separate caveats, and a block of four
        notes reads as seven.
    """
    notes = list(notes)
    widths = measure_labels_in(notes, fontsize=fontsize)
    lines: list[str] = []
    for note, measured in zip(notes, widths, strict=True):
        if measured <= width_in or not note.strip():
            lines.append(note)
            continue
        budget = max(_MIN_WRAP_CHARS, int(len(note) * width_in / measured))
        lines.extend(textwrap.wrap(note, width=budget, subsequent_indent=_HANGING_INDENT) or [note])
    return lines


def wrap_label(text: str, *, width_in: float, fontsize: float = 8.0) -> list[str]:
    """Break one label onto as many lines as the room it is drawn against is long.

    :func:`wrap_notes` is for a block of footnotes and hangs its continuation lines, which is
    right there and wrong here: an axis or colourbar label is one string drawn centred, and an
    indent on its second line reads as a misalignment. The bug this exists for is a colourbar
    label longer than its bar — on a landscape grid the bar is an inch tall and the label
    naming the quantity is three inches long, so it is drawn straight off the figure and the
    colour ends up unexplained.

    Args:
        text: The label.
        width_in: Room available for one line, along whatever axis it is drawn on.
        fontsize: Points it will be drawn at.

    Returns:
        The lines, in reading order; a label that already fits is returned as one line.
    """
    if not text.strip() or widest_label_in([text], fontsize=fontsize) <= width_in:
        return [text]
    # Measured greedily, word by word, rather than by :func:`wrap_notes`' character budget.
    # That budget carries a minimum line length, because a footnote broken to one word per
    # line is less legible than one that overhangs the figure — true of a footnote, false
    # here: a label that overhangs its bar is drawn off the page, so a short line is the
    # lesser evil and the width has to be honoured exactly rather than estimated.
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and widest_label_in([candidate], fontsize=fontsize) > width_in:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def grid_data_box_in(
    *,
    n_rows: int,
    n_cols: int,
    cell_in: float = 0.30,
    min_data_in: float = 1.8,
    max_data_in: float = 7.0,
    max_cell_in: float = 0.46,
) -> tuple[float, float]:
    """The width and height, in inches, of the data area :func:`grid_canvas` will place.

    Exposed because a caller sometimes has to size something *against* the panel before the
    panel exists — a colourbar label has to be wrapped to the bar's height, and the bar is as
    tall as the data area. Deriving that a second time in the caller is how the two drift.

    Args:
        n_rows: Rows in the grid.
        n_cols: Columns in the grid.
        cell_in: Requested side of one cell, in inches.
        min_data_in: Floor on the data width.
        max_data_in: Ceiling on the data width.
        max_cell_in: Ceiling on how far ``min_data_in`` may enlarge a cell.

    Returns:
        ``(width_in, height_in)``. Cells are square, so the height is the cell size times
        ``n_rows``.
    """
    # The width floor may enlarge the cells, but only to `max_cell_in`: height follows the
    # cell size, so on a narrow grid an unchecked floor buys width by spending height.
    width_floor_in = min(min_data_in, max_cell_in * max(n_cols, 1))
    data_w_in = min(max_data_in, max(width_floor_in, cell_in * n_cols))
    # No floor on the height: raising a few-row grid's height without raising its width
    # would leave the cells rectangular, in a helper whose one promise is that they are not.
    return data_w_in, (data_w_in / max(n_cols, 1)) * n_rows


def square_matrix_canvas(
    *,
    n: int,
    label_in: float,
    notes: Sequence[str] = (),
    has_title: bool = False,
    figsize: tuple[float, float] | None = None,
    cell_in: float = 0.46,
    min_data_in: float = 2.6,
    max_data_in: float = 4.4,
    note_fontsize: float = 7.5,
    colorbar: bool = True,
) -> tuple[Figure, Axes, Axes | None]:
    """Place a square n-by-n matrix, its label gutter, and optionally its colourbar.

    The notes are passed in rather than counted, because they determine the layout
    twice: a long note buys the data area some extra width (up to ``max_data_in``,
    so the figure grows its panel rather than growing a dead band), and whatever
    still does not fit is wrapped, which is what actually sets the height reserved
    below the axes. :func:`write_notes` wraps to the same width, so the room
    reserved here is the room the notes take.

    Args:
        n: Rows (and columns) in the matrix.
        label_in: Rendered width of the widest row label, from
            :func:`widest_label_in`. The same text is assumed to run along the
            columns, rotated, so the room below the axes is derived from it.
        notes: Footnotes the caller will write with :func:`write_notes`.
        has_title: Whether room is needed above the axes for a title.
        figsize: Overrides the computed size. The axes stay square in *data* terms
            but will no longer be square on the page, which is the caller's choice
            to make.
        cell_in: Side of one matrix cell, in inches.
        min_data_in: Floor on the data area, so a 3-by-3 matrix is not postage-stamp
            sized.
        max_data_in: Ceiling on how far a long note may grow the data area. Without
            it a wordy caveat would inflate a 3-by-3 panel to a full page.
        note_fontsize: Points the notes will be drawn at, for the wrap measurement.
        colorbar: When ``False`` the third return value is ``None`` and the width the
            colourbar would have taken is returned to the data area.

    Returns:
        ``(figure, axes, colourbar_axes)``.
    """
    import matplotlib.pyplot as plt

    column_in = label_in * _ROTATED_LABEL_RISE + 0.12
    # The tick pad the labels are offset by, on top of the text width itself.
    label_in += 0.14

    cbar_pad_in, cbar_in, cbar_label_in = (0.24, 0.17, 0.62) if colorbar else (0.0, 0.0, 0.10)
    side_in = label_in + cbar_pad_in + cbar_in + cbar_label_in

    data_in = max(min_data_in, cell_in * n)
    # Spend some of the width the notes want on the panel. The alternative is a figure
    # whose width is set by its caption and whose right third is empty.
    widest_note_in = widest_label_in(notes, fontsize=note_fontsize) / _NOTE_WIDTH_FRACTION
    data_in = min(max_data_in, max(data_in, widest_note_in - side_in))

    width = side_in + data_in
    n_lines = len(wrap_notes(notes, width_in=width * _NOTE_WIDTH_FRACTION, fontsize=note_fontsize))
    top_in = 0.42 if has_title else 0.12
    notes_in = (0.09 + NOTE_LINE_IN * n_lines) if n_lines else 0.10

    height = top_in + data_in + column_in + notes_in
    if figsize is not None:
        width, height = figsize

    fig = plt.figure(figsize=(width, height))
    bottom = (column_in + notes_in) / height
    ax = fig.add_axes((label_in / width, bottom, data_in / width, data_in / height))
    cax = (
        fig.add_axes(
            (
                (label_in + data_in + cbar_pad_in) / width,
                bottom,
                cbar_in / width,
                data_in / height,
            )
        )
        if colorbar
        else None
    )
    return fig, ax, cax


def grid_canvas(
    *,
    n_rows: int,
    n_cols: int,
    row_label_in: float,
    col_label_in: float,
    notes: Sequence[str] = (),
    has_title: bool = False,
    figsize: tuple[float, float] | None = None,
    cell_in: float = 0.30,
    min_data_in: float = 1.8,
    max_data_in: float = 7.0,
    max_cell_in: float = 0.46,
    note_fontsize: float = 7.5,
    colorbar: bool = True,
    cbar_label_in: float = 0.62,
    axis_label_in: float = 0.0,
) -> tuple[Figure, Axes, Axes | None]:
    """Place a rectangular ``n_rows`` x ``n_cols`` grid of equal cells and its two gutters.

    :func:`square_matrix_canvas` assumes the row and column labels are the same strings —
    true of a correlation or overlap matrix, false of everything else. A ligand-by-sender
    grid has long gene names down the left and cell-type names along the bottom, and forcing
    it through the square helper either clips one gutter or reserves the wrong one.

    Cells are kept square in *inches*, so a 20-by-4 grid is a tall narrow panel rather than
    a stretched one: a reader compares cells by area, and a grid whose cells are 3:1
    rectangles reads as though the columns matter more than the rows.

    Args:
        n_rows: Rows in the grid.
        n_cols: Columns in the grid.
        row_label_in: Rendered width of the widest row label, from :func:`widest_label_in`.
        col_label_in: Rendered width of the widest column label. Assumed to be laid over,
            so the room reserved below the axes is derived from it.
        notes: Footnotes the caller will write with :func:`write_notes`.
        has_title: Whether room is needed above the axes for a title.
        figsize: Overrides the computed size.
        cell_in: Side of one cell, in inches.
        min_data_in: Floor on the data *width*, so a two-column grid is not a strip.
        max_data_in: Ceiling on the data width, so a 40-column grid does not run off a page.
        max_cell_in: Ceiling on how far ``min_data_in`` may enlarge a cell. The floor is on
            the *width*, and the height is derived from the cell size, so on a narrow grid
            an unchecked floor pays for its width in height: a 3-column 15-row grid asked
            for 0.34 in cells gets 0.60 in ones and a 9 in tall panel, and a single-column
            one gets 1.8 in cells and a 27 in panel. Past this size the grid is left narrow
            instead, which is the lesser of the two illegibilities.
        note_fontsize: Points the notes will be drawn at, for the wrap measurement.
        colorbar: When ``False`` the third return value is ``None``.
        cbar_label_in: Room to the right of the colourbar for its tick labels and its own
            label. One line's worth by default; a caller that wraps a long label to a short
            bar (see :func:`wrap_label`) has to buy the extra lines here, or they are drawn
            over the notes.
        axis_label_in: Extra room, in inches, for a ``set_xlabel``/``set_ylabel`` pair beyond
            the tick labels. Zero by default because a grid whose rows and columns are named
            in the notes needs none. Nonzero is not optional when the caller does set them:
            the tick-label reservation is measured from the *tick* strings, so an axis label
            drawn beneath it lands in the footnote block and overprints the first line of
            text. That is how "Candidate sender" came to be drawn across a caveat about what
            the colour means.

    Returns:
        ``(figure, axes, colourbar_axes)``.
    """
    import matplotlib.pyplot as plt

    column_in = col_label_in * _ROTATED_LABEL_RISE + 0.12 + axis_label_in
    gutter_in = row_label_in + 0.14 + axis_label_in

    cbar_pad_in, cbar_in, label_in = (0.24, 0.17, cbar_label_in) if colorbar else (0.0, 0.0, 0.10)
    side_in = gutter_in + cbar_pad_in + cbar_in + label_in

    data_w_in, data_h_in = grid_data_box_in(
        n_rows=n_rows,
        n_cols=n_cols,
        cell_in=cell_in,
        min_data_in=min_data_in,
        max_data_in=max_data_in,
        max_cell_in=max_cell_in,
    )

    width = side_in + data_w_in
    n_lines = len(wrap_notes(notes, width_in=width * _NOTE_WIDTH_FRACTION, fontsize=note_fontsize))
    top_in = 0.42 if has_title else 0.12
    notes_in = (0.09 + NOTE_LINE_IN * n_lines) if n_lines else 0.10

    height = top_in + data_h_in + column_in + notes_in
    if figsize is not None:
        width, height = figsize

    fig = plt.figure(figsize=(width, height))
    bottom = (column_in + notes_in) / height
    ax = fig.add_axes((gutter_in / width, bottom, data_w_in / width, data_h_in / height))
    cax = (
        fig.add_axes(
            (
                (gutter_in + data_w_in + cbar_pad_in) / width,
                bottom,
                cbar_in / width,
                data_h_in / height,
            )
        )
        if colorbar
        else None
    )
    return fig, ax, cax


def row_panel_canvas(
    *,
    n_rows: int,
    label_in: float,
    data_in: float,
    notes: Sequence[str] = (),
    has_title: bool = False,
    label_pad_in: float = 0.18,
    right_in: float = 0.9,
    row_in: float = 0.24,
    min_rows_in: float = 1.6,
    xaxis_in: float = 0.58,
    top_in: float | None = None,
    figsize: tuple[float, float] | None = None,
    note_fontsize: float = 7.5,
) -> tuple[Figure, Axes]:
    """Place one row-per-item panel: a label gutter, a fixed-width data area, notes below.

    The counterpart to :func:`square_matrix_canvas` for the other shape a panel in this
    package takes — one row per pathway, program, or pair, with the names running down the
    left outside the axes and a shared x scale across them.

    The data area is a *fixed* width and is deliberately not grown to fit the notes, which
    is the opposite of what :func:`square_matrix_canvas` does. A matrix panel is read on
    its own; these are read against each other, and a coefficient of 0.6 has to be the
    same length in every figure that draws one or the comparison is not available. So a
    wordy caveat wraps instead of widening the panel.

    Args:
        n_rows: How many rows the panel draws.
        label_in: Rendered width of the widest row label, from :func:`widest_label_in`.
        data_in: Width of the data area, in inches. The cross-figure invariant.
        notes: Footnotes the caller will write with :func:`write_notes`.
        has_title: Whether room is needed above the axes for a title.
        label_pad_in: Gap between the labels and the axes, on top of the text width.
        right_in: Room right of the axes, for marks that would otherwise land on a point.
        row_in: Height of one row, in inches.
        min_rows_in: Floor on the data height, so a three-row panel is not a strip.
        xaxis_in: Room below the axes for the tick labels and the axis label. Folding it
            into the axes height instead leaves the notes stacked from the figure's bottom
            edge with the x axis printed straight through them.
        top_in: Room above the axes. Defaults to enough for a title, or for a legend
            placed just above the axes when there is none.
        figsize: Overrides the computed size.
        note_fontsize: Points the notes will be drawn at, for the wrap measurement.

    Returns:
        ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    gutter_in = label_in + label_pad_in
    rows_in = max(min_rows_in, row_in * n_rows)
    if top_in is None:
        top_in = 0.66 if has_title else 0.36

    width = gutter_in + data_in + right_in
    n_lines = len(wrap_notes(notes, width_in=width * _NOTE_WIDTH_FRACTION, fontsize=note_fontsize))
    notes_in = (0.09 + NOTE_LINE_IN * n_lines) if n_lines else 0.10

    height = top_in + rows_in + xaxis_in + notes_in
    if figsize is not None:
        width, height = figsize

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(
        (
            gutter_in / width,
            (notes_in + xaxis_in) / height,
            data_in / width,
            rows_in / height,
        )
    )
    return fig, ax


def xy_panel_canvas(
    *,
    data_in: float,
    height_in: float | None = None,
    yaxis_in: float = 0.62,
    xaxis_in: float = 0.58,
    notes: Sequence[str] = (),
    has_title: bool = False,
    right_in: float = 0.18,
    top_in: float | None = None,
    figsize: tuple[float, float] | None = None,
    note_fontsize: float = 7.5,
) -> tuple[Figure, Axes]:
    """Place a panel with two continuous axes: numeric ticks on both sides, notes below.

    The third shape a panel in this package takes, after :func:`square_matrix_canvas` and
    :func:`row_panel_canvas` — a scatter, a volcano, an observed-against-expected plot. Those
    two both size their left gutter from the *widest category name*, measured with
    :func:`widest_label_in`, because a category axis carries text outside the axes. An x/y
    panel has none: its gutter is set by a couple of numerals and one axis label, which is a
    constant and not a measurement. Passing a label width of ``0.4`` to
    :func:`row_panel_canvas` and treating ``n_rows`` as a size dial gets close, and then the
    room below the axes comes out of a row count that means nothing, so the axis label prints
    through the notes.

    The data area is a *fixed* width, for the same reason as :func:`row_panel_canvas`: a
    decade of p-value has to be the same length in every figure that draws one.

    Args:
        data_in: Width of the data area, in inches. The cross-figure invariant.
        height_in: Height of the data area. Defaults to ``data_in``, i.e. square on the page —
            which is what an axis-against-comparable-axis panel wants, so that departure from
            the diagonal is read as distance rather than as slope.
        yaxis_in: Room left of the axes for the tick labels and the y axis label.
        xaxis_in: Room below the axes for the tick labels and the x axis label. Kept out of
            the axes height so the notes are not stacked underneath the axis label.
        notes: Footnotes the caller will write with :func:`write_notes`.
        has_title: Whether room is needed above the axes for a title.
        right_in: Room right of the axes, for a point sitting on the last tick.
        top_in: Room above the axes. Defaults to enough for a title, or for a legend placed
            just above the axes when there is none.
        figsize: Overrides the computed size.
        note_fontsize: Points the notes will be drawn at, for the wrap measurement.

    Returns:
        ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    if data_in <= 0:
        raise ValueError(f"data_in must be positive, got {data_in}")
    rows_in = data_in if height_in is None else height_in
    if rows_in <= 0:
        raise ValueError(f"height_in must be positive, got {height_in}")
    if top_in is None:
        top_in = 0.66 if has_title else 0.36

    width = yaxis_in + data_in + right_in
    n_lines = len(wrap_notes(notes, width_in=width * _NOTE_WIDTH_FRACTION, fontsize=note_fontsize))
    notes_in = (0.09 + NOTE_LINE_IN * n_lines) if n_lines else 0.10

    height = top_in + rows_in + xaxis_in + notes_in
    if figsize is not None:
        width, height = figsize

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(
        (
            yaxis_in / width,
            (notes_in + xaxis_in) / height,
            data_in / width,
            rows_in / height,
        )
    )
    return fig, ax


def stacked_panel_canvas(
    *,
    heights_in: Sequence[float],
    label_in: float,
    data_in: float,
    notes: Sequence[str] = (),
    has_title: bool = False,
    label_pad_in: float = 0.18,
    right_in: float = 0.14,
    gap_in: float = 0.06,
    xaxis_in: float = 0.58,
    top_in: float | None = None,
    figsize: tuple[float, float] | None = None,
    note_fontsize: float = 7.5,
) -> tuple[Figure, list[Axes]]:
    """Place several tracks over one shared x axis, each track its own stated height.

    The shape a figure takes when two or three quantities are read *against the same
    horizontal position* — a curve, the positions that produced it, the values it was
    computed from. Splitting them across separate axes rather than twinning one keeps each
    track's y scale its own, and stating the heights in inches is what makes the tall track
    tall: a height ratio applied to a figure whose size was solved for the labels gives a
    different split on every figure.

    Only the bottom track gets room below it, so the tracks above must have their tick
    labels suppressed by the caller — which is the point of stacking them.

    Args:
        heights_in: Height of each track in inches, top to bottom.
        label_in: Rendered width of the widest y-axis label, from :func:`widest_label_in`.
        data_in: Width of the shared data area, in inches.
        notes: Footnotes the caller will write with :func:`write_notes`.
        has_title: Whether room is needed above the top track for a title.
        label_pad_in: Gap between the labels and the axes.
        right_in: Room right of the axes.
        gap_in: Vertical gap between adjacent tracks.
        xaxis_in: Room below the bottom track for its tick labels and axis label.
        top_in: Room above the top track. Defaults to enough for a title, or a small
            margin when there is none.
        figsize: Overrides the computed size.
        note_fontsize: Points the notes will be drawn at, for the wrap measurement.

    Returns:
        ``(figure, axes)`` with the axes in the order the heights were given, top first.

    Raises:
        ValueError: ``heights_in`` is empty or holds a non-positive height.
    """
    import matplotlib.pyplot as plt

    heights = [float(h) for h in heights_in]
    if not heights:
        raise ValueError("no tracks given; stacked_panel_canvas needs at least one height")
    if any(h <= 0 for h in heights):
        raise ValueError(f"every track height must be positive, got {heights}")

    gutter_in = label_in + label_pad_in
    if top_in is None:
        top_in = 0.42 if has_title else 0.12

    width = gutter_in + data_in + right_in
    n_lines = len(wrap_notes(notes, width_in=width * _NOTE_WIDTH_FRACTION, fontsize=note_fontsize))
    notes_in = (0.09 + NOTE_LINE_IN * n_lines) if n_lines else 0.10

    tracks_in = sum(heights) + gap_in * (len(heights) - 1)
    height = top_in + tracks_in + xaxis_in + notes_in
    if figsize is not None:
        width, height = figsize

    fig = plt.figure(figsize=(width, height))
    axes: list[Axes] = []
    top_edge_in = height - top_in
    for track_in in heights:
        top_edge_in -= track_in
        axes.append(
            fig.add_axes(
                (gutter_in / width, top_edge_in / height, data_in / width, track_in / height)
            )
        )
        top_edge_in -= gap_in
    return fig, axes


def write_notes(fig: Figure, notes: Sequence[str], *, fontsize: float = 7.5) -> None:
    """Stack notes under the axes, in reading order, at a fixed inch pitch.

    Wrapped to the figure's own width: these figures are saved with
    ``bbox_inches="tight"``, so a note wider than the figure does not overflow the page,
    it widens the page and leaves the panel sitting in a dead band.
    """
    height = fig.get_figheight()
    lines = wrap_notes(notes, width_in=fig.get_figwidth() * _NOTE_WIDTH_FRACTION, fontsize=fontsize)
    for position, note in enumerate(lines):
        offset = 0.09 + NOTE_LINE_IN * (len(lines) - 1 - position)
        fig.text(
            0.01,
            offset / height,
            note,
            fontsize=fontsize,
            color=MUTED,
            ha="left",
            va="bottom",
        )


__all__ = [
    "ABSENT",
    "CHAR_IN",
    "INK",
    "MUTED",
    "NOTE_LINE_IN",
    "grid_canvas",
    "grid_data_box_in",
    "measure_labels_in",
    "row_panel_canvas",
    "square_matrix_canvas",
    "stacked_panel_canvas",
    "widest_label_in",
    "wrap_label",
    "wrap_notes",
    "write_notes",
    "xy_panel_canvas",
]
