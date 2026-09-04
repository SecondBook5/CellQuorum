"""Program correlations drawn so the reader can see what the coefficient is made of.

A program-by-program correlation heatmap is the second table of almost every
multi-program analysis, and it is drawn from ``DataFrame.corr()`` almost every time.
That gets the coefficient right and hides the three things a reader needs in order to
believe it, so the two figures here put all three on the page.

**The unit.** A coefficient over 2,000 cells from nine donors is a coefficient with an
n of nine, and the difference is not cosmetic — it is the difference between p = 0.02
and p = 0.9. Both figures state the unit and its count on the figure itself, so a panel
cannot be read without them.

**The condition.** Two programs that are both raised in disease correlate across samples
without co-varying within either arm, because the condition is a common cause of both.
:func:`program_correlation_heatmap` therefore draws the raw coefficient below the
diagonal and the condition-adjusted one above it, and
:func:`program_correlation_slopes` draws the two as one segment per pair — which is the
figure that makes the correction visible, because a pair whose correlation *is* the
condition collapses toward zero along a visibly long segment.

**The shared genes.** Two programs sharing seven of eight members must track each other;
the resulting r is arithmetic, not biology. Pairs that share genes are marked and the
counts are footnoted, so a definitional correlation cannot be read as an empirical one.

Both figures read the long-form frame
:func:`cellquorum.stats.program_correlation.program_correlation_tests` writes rather
than a matrix, because the matrix is precisely the object that cannot carry any of the
above. A figure and the CSV beside it therefore cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cellquorum.visualization.figstyle import (
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    significance_stars,
)
from cellquorum.visualization.measured_layout import (
    ABSENT,
    INK,
    row_panel_canvas,
    square_matrix_canvas,
    widest_label_in,
    write_notes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.colors import Colormap

#: The pooled coefficient is drawn in grey and the adjusted one in the house blue, so
#: the eye lands on the value that survived the adjustment rather than on the one that
#: did not. Two tones differing in both hue and lightness, which is all this contrast
#: needs — a categorical palette here would imply the two are peer categories.
_RAW = "#9a9a9a"
_ADJUSTED = "#24608F"

#: Diverging, ColorBrewer, symmetric about zero: a correlation is signed, and a
#: sequential ramp would make -0.9 and +0.1 look equally unremarkable.
_CORRELATION_CMAP = "RdBu_r"

#: Above this absolute coefficient the cell is dark enough that ink-on-cell is
#: unreadable and the label flips to white.
_DARK_CELL = 0.55

_COEFFICIENT_LABELS = {"spearman": "Spearman ρ", "pearson": "Pearson r"}

#: Marks, reused from the mediation figures' vocabulary so one reader learns them once.
_SHARED_MARK = "‡"
_LOST_MARK = "×"
_UNTESTED_MARK = "?"

#: How many pair names a footnote enumerates before summarising the remainder. Long
#: enough to name the finding, short enough that the note stays one line at 7.5pt.
_MAX_NAMED = 4

#: Width of the slope figure's data area, in inches. Fixed rather than derived: the
#: x-axis is always a correlation on [-1, 1], so a panel that widened with the number of
#: pairs would make the same coefficient look different from one figure to the next.
_SLOPE_DATA_IN = 3.6


def _pretty(name: str, labels: Mapping[str, str] | None) -> str:
    if labels and name in labels:
        return labels[name]
    return str(name).replace("_", " ")


def _pair(row: pd.Series, labels: Mapping[str, str] | None) -> str:
    return f"{_pretty(row['program_a'], labels)}/{_pretty(row['program_b'], labels)}"


def _enumerate_pairs(names: Sequence[str]) -> str:
    """Name the first few and count the rest, so a note stays one line."""
    if len(names) <= _MAX_NAMED:
        return ", ".join(names)
    return f"{', '.join(names[:_MAX_NAMED])}, and {len(names) - _MAX_NAMED} more"


def _coefficient_label(tests: pd.DataFrame) -> str:
    methods = {str(m).lower() for m in tests.get("method", pd.Series(dtype=str)).dropna().unique()}
    if len(methods) == 1:
        return _COEFFICIENT_LABELS.get(next(iter(methods)), "correlation")
    return "correlation"


def _unit_phrase(tests: pd.DataFrame) -> str:
    """``"9 donors"``, or a description of the disagreement if the rows disagree.

    The unit arrives as the obs column name the frame was aggregated over, and
    ``"12 sample_ids"`` is an identifier where the sentence wants a noun. An ``_id``
    suffix is a naming convention rather than part of the thing counted, so it is
    dropped; nothing else about the name is guessed at.
    """
    if "unit" not in tests.columns or "n_units" not in tests.columns:
        return "units"
    pairs = {(str(u), int(n)) for u, n in zip(tests["unit"], tests["n_units"], strict=True)}
    if len(pairs) != 1:
        return "differing units"
    unit, count = next(iter(pairs))
    noun = unit.replace("_", " ")
    for suffix in (" id", " ids"):
        if noun.endswith(suffix) and noun != suffix.strip():
            noun = noun[: -len(suffix)]
            break
    return f"{count:,} {noun}{'s' if count != 1 and not noun.endswith('s') else ''}"


def _sentence_case(phrase: str) -> str:
    """Raise the first letter and leave the rest alone — ``str.capitalize`` lowercases
    the remainder, which turns a column named ``TotalCounts`` into ``Totalcounts``."""
    return phrase[:1].upper() + phrase[1:]


def _adjustment_phrase(tests: pd.DataFrame) -> str:
    """What the adjusted coefficient removed, as a noun phrase without its article.

    The condition is the adjustment this figure was built for, but it is not the only one
    a study needs — sequencing depth raises every score at once and is the other common
    cause worth removing. So the panel reads what was removed out of the frame instead of
    asserting "condition" and being wrong the first time someone passes a covariate.
    """
    if "adjusted_for" not in tests.columns:
        return "condition"
    values = {str(value).strip() for value in tests["adjusted_for"].dropna()}
    values.discard("")
    if len(values) != 1:
        return "condition"
    parts = [part.strip().replace("_", " ") for part in next(iter(values)).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return "condition"
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _program_order(tests: pd.DataFrame, order: Sequence[str] | None) -> list[str]:
    """The axis order: the caller's, else first appearance in the frame.

    First appearance rather than sorted, because the frame arrives from the stage in the
    canonical program order the other tables and panels already use, and re-sorting here
    would give this one figure a different axis from the dot-grid beside it.
    """
    seen: list[str] = []
    for column in ("program_a", "program_b"):
        for name in tests[column].astype(str):
            if name not in seen:
                seen.append(name)
    if order is None:
        return seen
    chosen = [str(name) for name in order if str(name) in set(seen)]
    return chosen + [name for name in seen if name not in set(chosen)]


def _lost_its_call(tests: pd.DataFrame, alpha: float) -> pd.Series:
    """Significant pooled across conditions, not significant within them.

    The adjustment has to have been *attempted* for this to mean anything: a NaN
    ``fdr_adjusted`` because no condition was supplied is not a pair that lost its call,
    and marking it as one would invent a finding.
    """
    if "fdr_adjusted" not in tests.columns or "fdr" not in tests.columns:
        return pd.Series(False, index=tests.index)
    raw = pd.to_numeric(tests["fdr"], errors="coerce")
    adjusted = pd.to_numeric(tests["fdr_adjusted"], errors="coerce")
    return (raw < alpha) & adjusted.notna() & ~(adjusted < alpha)


def program_correlation_notes(
    tests: pd.DataFrame,
    *,
    alpha: float = 0.05,
    program_labels: Mapping[str, str] | None = None,
) -> list[str]:
    """
    The notes a program-correlation panel has to carry, derived from the table.

    Public and pure so a composed figure can place them once, and so what a figure
    claims can be tested against the frame it came from rather than against a rendering.

    Args:
        tests: The frame from
            :func:`~cellquorum.stats.program_correlation.program_correlation_tests`.
        alpha: The FDR threshold the marks used.
        program_labels: Display names, keyed by the program names in the frame. Pass the
            same mapping the panel's axes use: a note naming ``endomt_lec`` under an axis
            reading ``EndoMT (LEC)`` makes the reader translate between two vocabularies
            to find the pair the mark refers to.

    Returns:
        Sentences, in reading order. Never empty: the unit and the test family always
        have to be stated, because a correlation panel without them is not
        interpretable.
    """
    notes: list[str] = []
    labels = program_labels
    n_pairs = len(tests)
    tested = pd.to_numeric(tests.get("p_value", pd.Series(dtype=float)), errors="coerce").notna()

    notes.append(
        f"{_coefficient_label(tests)} over {_unit_phrase(tests)}; two-sided, "
        f"Benjamini–Hochberg across the {int(tested.sum())} of {n_pairs} testable "
        f"pair(s); * FDR < {alpha:g}, ** < 0.01, *** < 0.001."
    )

    if "unit" in tests.columns and (tests["unit"].astype(str) == "row").all():
        # The one failure this whole module exists to prevent, stated on the figure
        # whenever the frame admits to it.
        notes.append(
            "The unit is a row of the score matrix. If those rows are cells rather than "
            "samples, the p-values treat cells from one donor as independent "
            "observations and are anticonservative."
        )

    if "shared_genes" in tests.columns:
        shared = pd.to_numeric(tests["shared_genes"], errors="coerce").fillna(-1)
        if (shared < 0).all():
            notes.append(
                "Gene lists were not supplied, so shared membership is unknown. Two "
                "programs sharing members correlate partly by construction."
            )
        elif (shared > 0).any():
            named = [
                f"{_pair(row, labels)} ({int(row['shared_genes'])})"
                for _, row in tests[shared > 0].iterrows()
            ]
            notes.append(
                f"{_SHARED_MARK} share genes and so correlate partly by construction: "
                f"{_enumerate_pairs(named)}."
            )

    lost = _lost_its_call(tests, alpha)
    if lost.any():
        named = [_pair(row, labels) for _, row in tests[lost].iterrows()]
        notes.append(
            f"{_LOST_MARK} significant as measured and not after removing "
            f"the {_adjustment_phrase(tests)}: {_enumerate_pairs(named)} — a common cause "
            "of both programs rather than evidence that they co-vary."
        )

    if {"n_units_adjusted", "n_units"} <= set(tests.columns):
        adjusted_n = pd.to_numeric(tests["n_units_adjusted"], errors="coerce")
        total_n = pd.to_numeric(tests["n_units"], errors="coerce")
        short = adjusted_n.gt(0) & adjusted_n.lt(total_n)
        if short.any():
            # Two coefficients computed over different units is exactly the thing a
            # reader would never guess, so it is stated rather than left in the CSV.
            notes.append(
                f"The adjusted coefficient uses {int(adjusted_n[short].min()):,}–"
                f"{int(adjusted_n[short].max()):,} of the {int(total_n.max()):,} units: the "
                "rest are missing a value for the variables being removed."
            )

    if "reason" in tests.columns:
        refused = tests[~tested & tests["reason"].astype(str).str.len().gt(0)]
        if len(refused):
            named = [f"{_pair(row, labels)} ({row['reason']})" for _, row in refused.iterrows()]
            notes.append(f"{_UNTESTED_MARK} not tested: {_enumerate_pairs(named)}.")

    return notes


def program_correlation_heatmap(
    tests: pd.DataFrame,
    *,
    program_labels: Mapping[str, str] | None = None,
    program_order: Sequence[str] | None = None,
    alpha: float = 0.05,
    title: str | None = None,
    footnotes: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Draw every program pair once, with its significance and its caveats in the cell.

    Three layout choices worth knowing about:

    * **The two triangles carry different quantities.** Below the diagonal is the
      coefficient at the stated unit; above it is the same coefficient after the
      condition has been removed from both programs. The conventional mirrored matrix
      spends half its area on redundant ink; here that half answers the question a
      reader of the lower triangle immediately has. When no condition was supplied the
      upper triangle is left empty and a note says why, rather than being mirrored.
    * **Significance is drawn into the cell**, not into a second panel. A colour and a
      mark that disagree — a large coefficient with no stars, at n = 9 — is the honest
      picture and is what a reader needs to see.
    * **The diagonal is empty.** A program correlates with itself at 1 and would own the
      end of the colour scale.

    Args:
        tests: The frame from
            :func:`~cellquorum.stats.program_correlation.program_correlation_tests`.
        program_labels: Display names, keyed by the program names in the frame.
        program_order: Axis order. Defaults to first appearance in ``tests``, which is
            the canonical order the stage's other tables use.
        alpha: FDR threshold for the stars and for the lost-call mark.
        title: Figure title.
        footnotes: Place :func:`program_correlation_notes` under the axes.
        figsize: Overrides the size derived from the number of programs and the width of
            their names.

    Returns:
        The figure.

    Raises:
        ValueError: ``tests`` is empty or is missing the columns the frame is defined by.
    """
    apply_cellquorum_theme()

    required = {"program_a", "program_b", "r"}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(
            f"tests is missing {sorted(missing)}; pass the frame from "
            "cellquorum.stats.program_correlation_tests"
        )
    if tests.empty:
        raise ValueError("tests is empty; there is no pair to draw")

    order = _program_order(tests, program_order)
    n = len(order)
    position = {name: index for index, name in enumerate(order)}
    labels = [_pretty(name, program_labels) for name in order]

    raw = pd.to_numeric(tests["r"], errors="coerce")
    adjusted = pd.to_numeric(tests.get("r_adjusted", pd.Series(index=tests.index)), errors="coerce")
    has_adjusted = bool(adjusted.notna().any())
    lost = _lost_its_call(tests, alpha)

    values = np.full((n, n), np.nan)
    marks: dict[tuple[int, int], str] = {}
    for key, row in tests.iterrows():
        i, j = position[str(row["program_a"])], position[str(row["program_b"])]
        low, high = (max(i, j), min(i, j)), (min(i, j), max(i, j))
        values[low] = raw.loc[key]
        stars = _cell_mark(row.get("fdr"), row.get("p_value"), alpha)
        if bool(row.get("shares_genes", False)):
            stars += _SHARED_MARK
        marks[low] = stars
        if has_adjusted:
            values[high] = adjusted.loc[key]
            adjusted_stars = _cell_mark(row.get("fdr_adjusted"), row.get("p_adjusted"), alpha)
            marks[high] = adjusted_stars + (_LOST_MARK if lost.loc[key] else "")

    notes = (
        program_correlation_notes(tests, alpha=alpha, program_labels=program_labels)
        if footnotes
        else []
    )
    if not has_adjusted:
        notes.append(
            "The upper triangle is empty because no condition was supplied: there is "
            "nothing to adjust for, and mirroring the lower triangle would only repeat it."
        )
    else:
        notes.insert(
            0,
            "Below the diagonal: the coefficient as measured. Above: the same "
            f"coefficient with the {_adjustment_phrase(tests)} removed from both programs.",
        )

    fig, ax, cax = square_matrix_canvas(
        n=n,
        label_in=widest_label_in(labels),
        notes=notes,
        has_title=title is not None,
        figsize=figsize,
    )

    masked = np.ma.masked_invalid(values)
    cmap = _masked_cmap()
    image = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    for (i, j), text in marks.items():
        if not text:
            continue
        value = values[i, j]
        dark = np.isfinite(value) and abs(value) > _DARK_CELL
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color="white" if dark else INK,
        )

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    apply_cellquorum_axis_style(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # The minor grid is the cell boundary, so it has to be drawn after the house style
    # (which turns grids off) and it has to be white rather than grey: a grey line over
    # a dark cell reads as data.
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    bar = fig.colorbar(image, cax=cax)
    bar.set_label(f"{_coefficient_label(tests)} over {_unit_phrase(tests)}", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)

    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    if notes:
        write_notes(fig, notes)
    return fig


def _cell_mark(fdr: object, p_value: object, alpha: float) -> str:
    """Stars for a tested pair, ``?`` for an untested one, nothing for a null result.

    ``ns`` is deliberately not printed. In a matrix of fifty-five cells it is fifty-five
    pieces of ink saying nothing, and it makes the handful of cells that *do* carry a
    result harder to find rather than easier.
    """
    if not np.isfinite(pd.to_numeric(p_value, errors="coerce")):
        return _UNTESTED_MARK
    value = pd.to_numeric(fdr, errors="coerce")
    if not np.isfinite(value):
        return _UNTESTED_MARK
    # The house thresholds for the two inner tiers, and the caller's alpha for the
    # outer one — so a figure drawn at alpha=0.1 marks what its own footnote claims.
    stars = significance_stars(float(value))
    if stars != "ns":
        return stars
    return "*" if float(value) < alpha else ""


def _masked_cmap() -> Colormap:
    """The diverging map with an explicit "no value" colour for the empty cells.

    Matplotlib's default for masked data is transparency, which shows the figure's white
    background — indistinguishable from a genuine zero at the centre of a diverging map.
    """
    import matplotlib as mpl

    return mpl.colormaps[_CORRELATION_CMAP].with_extremes(bad=ABSENT)


def program_correlation_slopes(
    tests: pd.DataFrame,
    *,
    program_labels: Mapping[str, str] | None = None,
    alpha: float = 0.05,
    max_pairs: int | None = None,
    title: str | None = None,
    footnotes: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Draw each pair's coefficient before and after the condition is removed.

    This is the figure that carries the correction. A pair whose correlation is really
    the disease contrast read twice has a long segment collapsing toward zero; a pair
    that genuinely co-varies within each arm has a short one. Neither is visible in a
    heatmap of the pooled coefficient, which is what such analyses normally publish.

    Pairs that share genes are marked, because a short segment on a definitional pair is
    not evidence of co-regulation — the two scores would track each other in a single
    arm too.

    Args:
        tests: The frame from
            :func:`~cellquorum.stats.program_correlation.program_correlation_tests`.
        program_labels: Display names, keyed by the program names in the frame.
        alpha: FDR threshold for the significance marks.
        max_pairs: Keep at most this many pairs, the largest ``|r|`` first. ``None``
            draws every pair. Truncation is footnoted, never silent.
        title: Figure title.
        footnotes: Place :func:`program_correlation_notes` under the axes.
        figsize: Overrides the size derived from the number of pairs.

    Returns:
        The figure.

    Raises:
        ValueError: ``tests`` has no pair with a finite coefficient to draw.
    """
    apply_cellquorum_theme()

    frame = tests.copy()
    frame["_raw"] = pd.to_numeric(frame["r"], errors="coerce")
    frame["_adjusted"] = pd.to_numeric(
        frame.get("r_adjusted", pd.Series(index=frame.index)), errors="coerce"
    )
    frame = frame[frame["_raw"].notna()]
    if frame.empty:
        raise ValueError("no pair has a finite coefficient; there is nothing to draw")

    frame = frame.reindex(frame["_raw"].abs().sort_values(ascending=False).index)
    dropped = 0
    if max_pairs is not None and len(frame) > max_pairs:
        dropped = len(frame) - max_pairs
        frame = frame.iloc[:max_pairs]

    lost = _lost_its_call(tests, alpha).reindex(frame.index).fillna(False)
    rows = list(frame.index)
    pair_labels = [_pair(frame.loc[key], program_labels) for key in rows]

    notes = (
        program_correlation_notes(tests, alpha=alpha, program_labels=program_labels)
        if footnotes
        else []
    )
    if dropped:
        notes.append(
            f"{dropped} further pair(s) with smaller coefficients are not drawn; every "
            "pair is in the table."
        )

    # One axes, sized in inches for the same reason the matrix is: the pair names are
    # the widest thing on the figure and they live outside the axes. The legend sits
    # above the axes, so the room over it is never zero even without a title.
    fig, ax = row_panel_canvas(
        n_rows=len(rows),
        label_in=widest_label_in(pair_labels),
        data_in=_SLOPE_DATA_IN,
        notes=notes,
        has_title=bool(title),
        figsize=figsize,
    )

    y = np.arange(len(rows))
    ax.axvline(0.0, color=_RAW, linewidth=0.8, zorder=1)
    for index, key in enumerate(rows):
        raw = float(frame.loc[key, "_raw"])
        adjusted = frame.loc[key, "_adjusted"]
        if np.isfinite(adjusted):
            ax.plot(
                [raw, float(adjusted)],
                [index, index],
                color=_RAW,
                linewidth=1.1,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter([float(adjusted)], [index], s=26, color=_ADJUSTED, zorder=4)
        ax.scatter(
            [raw],
            [index],
            s=26,
            facecolor="white",
            edgecolor=_RAW,
            linewidth=1.1,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            label + (f" {_SHARED_MARK}" if bool(frame.loc[key].get("shares_genes", False)) else "")
            for label, key in zip(pair_labels, rows, strict=True)
        ],
        fontsize=8.5,
    )
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(f"{_coefficient_label(tests)} over {_unit_phrase(tests)}", fontsize=9)
    apply_cellquorum_axis_style(ax)
    ax.tick_params(axis="y", length=0)

    # The marks go outside the axes on the right, where they cannot land on a dot.
    for index, key in enumerate(rows):
        mark = _cell_mark(frame.loc[key].get("fdr"), frame.loc[key].get("p_value"), alpha)
        if lost.loc[key]:
            mark += _LOST_MARK
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

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=_RAW,
            markersize=6,
            label="As measured",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=_ADJUSTED,
            markersize=6,
            label=f"{_sentence_case(_adjustment_phrase(tests))} removed",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22)
    if notes:
        write_notes(fig, notes)
    return fig


__all__ = [
    "program_correlation_heatmap",
    "program_correlation_notes",
    "program_correlation_slopes",
]
