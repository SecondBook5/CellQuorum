"""One rectangular grid of values, drawn so that "not measured" never reads as zero.

Almost every table this engine produces wants to be seen as a grid — programs by clusters,
ligands by senders, panels by subtypes, a partition against another partition — and the
default heatmap is wrong for all of them in the same way. It has two visual states and the
data has three: a low value, a value that failed some stated threshold, and no value at
all. A colormap maps NaN to whatever the "bad" colour happens to be, which is usually an
end of the scale, so a cell that was never computed reads as the strongest or the weakest
cell in the figure.

So the absent cells are filled with a neutral grey that is in neither direction of the
scale, and a threshold failure gets a dashed outline rather than a different colour, which
leaves the colour axis meaning one thing only. That distinction was worth a module: it is
the difference between "this sender does not express the ligand" and "this sender was not
in the tested pool", and between "this subtype scores low for the panel" and "this subtype
had too few cells to score".

The grid is also where a reader looks for the numbers, so ``annotate`` writes a second
frame of text into the cells. A grid whose cells hold counts and whose colour holds a
fraction says both halves of a composition at once, and a reader who has to convert a
colour back into a count will do it wrong. That text takes its colour from the colour the
cell was actually painted, because a number written white on a near-white cell does not
look like a bug — it looks like an empty cell, which is the third visual state again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import apply_cellquorum_theme
from cellquorum.visualization.measured_layout import (
    ABSENT,
    INK,
    grid_canvas,
    grid_data_box_in,
    widest_label_in,
    wrap_label,
    write_notes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


#: Relative luminance below which a cell needs light text. WCAG puts the crossover for white
#: versus near-black text at about 0.18 for maximum contrast ratio; this sits a little higher
#: because the ink here is a dark grey rather than pure black.
_LIGHT_TEXT_BELOW = 0.22

#: Width one extra line of a wrapped colourbar label costs, in inches, at 8 pt rotated.
_CBAR_LINE_IN = 0.13

#: Room one extra line of a wrapped axis label costs, in inches, at 8.5 pt.
_AXIS_LINE_IN = 0.14

#: How far inside its cell a ``below`` outline is drawn, in cell widths. An outline on the cell
#: boundary is shared geometry: on a grid where most cells are marked, every dash sits between
#: two cells and a reader cannot tell which of the two it belongs to, so the mark stops carrying
#: information exactly when it is carrying the most. Inset by this much and the gap between two
#: neighbouring outlines is twice it, which is unambiguous at any cell size the canvas allows.
_BELOW_INSET = 0.09


def _text_on(rgba: tuple[float, float, float, float]) -> str:
    """Ink or white, whichever the reader can see against this cell.

    A cell's fill is whatever the colormap chose, and a caller cannot know that without
    evaluating the map, so the choice is made from the painted colour's relative luminance
    (BT.709 on linearized sRGB) rather than from the value's rank on the scale.
    """

    def _linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (_linear(float(c)) for c in rgba[:3])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < _LIGHT_TEXT_BELOW else INK


def value_grid(
    values: pd.DataFrame,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    notes: Sequence[str],
    row_label: str,
    col_label: str,
    below: pd.DataFrame | None = None,
    annotate: pd.DataFrame | None = None,
    annotate_fontsize: float = 6.5,
    cell_in: float = 0.30,
) -> tuple[Figure, Axes]:
    """Draw one rectangular grid of values, with absent cells and marked cells distinguished.

    Args:
        values: Rows by columns of numbers. The index and columns are the tick labels and
            their rendered widths set the gutters, so they arrive already in reading order
            and already displaying the names the caller wants shown.
        cmap: Colormap name for the value axis.
        vmin: Low end of the colour scale.
        vmax: High end of the colour scale.
        colorbar_label: What the colour means. Not optional in practice — a grid whose
            colour is unlabelled is a picture.
        notes: Footnotes, written under the axes with :func:`write_notes` and measured into
            the canvas height so they cannot overprint the figure.
        row_label: Axis label for the rows.
        col_label: Axis label for the columns.
        below: Optional boolean frame, aligned to ``values``, marking cells that failed a
            threshold the caller states in ``notes``. Marked with a dashed outline rather
            than a colour, so the colour axis keeps meaning one thing. The outline is inset
            inside the cell rather than drawn on its boundary, so that on a grid where most
            cells are marked each dash still belongs visibly to one cell.
        annotate: Optional frame of strings, aligned to ``values``, written into the cells.
            Use it when the number is the point and the colour is the pattern.
        annotate_fontsize: Points for the cell text.
        cell_in: Side of one cell, in inches. Cells are square in inches, so a tall narrow
            grid stays tall and narrow instead of being stretched to a page.

    Returns:
        ``(figure, axes)``. The colourbar axes is not returned: a caller that wants to
        restyle it has a grid whose colour means something this function cannot describe,
        and should draw its own.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    apply_cellquorum_theme()
    # The colourbar is as tall as the data area, so its label has to be wrapped to that
    # height before the canvas is sized — on a landscape grid the bar is an inch tall and a
    # label naming the quantity is three inches long, which matplotlib draws straight off the
    # figure and leaves the colour unexplained. Each extra line is bought back in width.
    data_w_in, data_h_in = grid_data_box_in(
        n_rows=values.shape[0], n_cols=values.shape[1], cell_in=cell_in
    )
    bar_label = "\n".join(wrap_label(colorbar_label, width_in=data_h_in * 0.94, fontsize=8))
    # The axis labels run *along* their axis, so on a landscape grid the row label is longer
    # than the panel is tall and matplotlib centres it off the top of the figure. Same wrap,
    # and the lines are bought back in gutter width the same way.
    row_lines = wrap_label(row_label, width_in=data_h_in * 0.94, fontsize=8.5)
    col_lines = wrap_label(col_label, width_in=data_w_in * 0.94, fontsize=8.5)
    fig, ax, cax = grid_canvas(
        n_rows=values.shape[0],
        n_cols=values.shape[1],
        row_label_in=widest_label_in([str(i) for i in values.index], fontsize=8),
        col_label_in=widest_label_in([str(c) for c in values.columns], fontsize=8),
        notes=notes,
        cell_in=cell_in,
        cbar_label_in=0.62 + _CBAR_LINE_IN * bar_label.count("\n"),
        # This function sets both axis labels below, so the room has to be asked for.
        axis_label_in=0.24 + _AXIS_LINE_IN * (max(len(row_lines), len(col_lines)) - 1),
    )

    data = values.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad(ABSENT)
    mesh = ax.imshow(
        masked,
        cmap=palette,
        norm=Normalize(vmin=vmin, vmax=vmax),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    if below is not None:
        flags = below.reindex(index=values.index, columns=values.columns).to_numpy(dtype=bool)
        side = 1.0 - 2 * _BELOW_INSET
        for r, c in zip(*np.where(flags), strict=True):
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5 + _BELOW_INSET, r - 0.5 + _BELOW_INSET),
                    side,
                    side,
                    fill=False,
                    edgecolor=INK,
                    linewidth=0.55,
                    linestyle=(0, (1.6, 1.2)),
                )
            )
    if annotate is not None:
        text = annotate.reindex(index=values.index, columns=values.columns)
        # Text colour comes from the colour the cell was actually painted, not from where the
        # value sits on the scale. Position works only for a diverging map, where both ends are
        # dark; on a sequential map like Blues the low end is near-white, and "the extremes take
        # light text" writes white on white and deletes the number.
        norm = Normalize(vmin=vmin, vmax=vmax)
        for r, row_name in enumerate(values.index):
            for c, col_name in enumerate(values.columns):
                label = text.loc[row_name, col_name]
                if label is None or (isinstance(label, float) and np.isnan(label)):
                    continue
                label = str(label)
                if not label:
                    continue
                ax.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annotate_fontsize,
                    color=_text_on(palette(norm(data[r, c]))),
                )

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(c) for c in values.columns], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([str(i) for i in values.index], fontsize=8)
    ax.set_xlabel("\n".join(col_lines), fontsize=8.5)
    ax.set_ylabel("\n".join(row_lines), fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if cax is not None:
        bar = fig.colorbar(mesh, cax=cax)
        bar.set_label(bar_label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
        bar.outline.set_visible(False)
    write_notes(fig, notes)
    return fig, ax


__all__ = ["value_grid"]
