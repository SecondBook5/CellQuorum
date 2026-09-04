"""The mediation figure: a decomposition drawn as a decomposition.

Mediation results are almost always published as a table of six numbers per
mediator, which hides the one thing a reader needs to judge — whether the indirect
path accounts for the total effect or is a sliver of it. A forest plot of the three
*effects* (indirect, direct, total) on one shared axis shows that at a glance: when
the indirect interval sits on top of the total and the direct straddles zero, the
effect is mediated, and no proportion needs to be quoted to see it.

Three deliberate choices:

* **Paths a and b are not plotted.** They are in different units from the effects
  (a is treatment-on-mediator, b is mediator-on-outcome), so putting them on the
  same axis would invite a comparison of magnitudes that means nothing. They stay
  in the table.
* **The proportion mediated is not plotted either**, for the reason
  :mod:`cellquorum.stats.causal_mediation` withholds it: it is a ratio whose
  denominator is one of the quantities already on the axis, so when the total is
  near zero the point flies off the plot and drags the scale with it.
* **Every guard the table records is drawn**, not dropped in the transfer from
  table to figure. A mediator whose score shares genes with the outcome is marked,
  a fit that was refused says so where its rows would have been, and a term whose
  significance depended on ignoring the donor pairing is flagged. A figure that
  silently omitted these would be more misleading than the table it came from.

There is a second figure here, :func:`mediation_sensitivity`, because the question a
reader actually asks about a mediation is not "is it significant" but "does it survive
the obvious objections". A panel that puts the primary estimate next to the same
estimate with the shared genes removed, with library depth adjusted for, or in a
lineage where it should be absent, answers that in one look — and it makes a mediator
that only holds in the primary fit impossible to present as though it held generally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cellquorum.visualization.figstyle import apply_cellquorum_axis_style, apply_cellquorum_theme
from cellquorum.visualization.measured_layout import INK, MUTED

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from matplotlib.axes import Axes

_ZERO_LINE = "#9a9a9a"

# The three terms drawn, in the order they read as an argument: this much went
# through the mediator, this much did not, and this is the sum.
_PLOTTED_TERMS: tuple[str, ...] = ("acme", "direct", "total")

_TERM_LABELS = {
    "acme": "Indirect (mediated)",
    "direct": "Direct",
    "total": "Total",
}

# Filled for the mediated path, hollow for the other two: the indirect effect is
# the claim, and the eye should land on it first.
_TERM_STYLE = {
    "acme": {"marker": "o", "filled": True, "size": 7.0},
    "direct": {"marker": "s", "filled": False, "size": 6.0},
    "total": {"marker": "D", "filled": False, "size": 5.5},
}

# Vertical offsets within a mediator's band, so the three intervals do not overlap.
_TERM_OFFSET = {"acme": 0.26, "direct": 0.0, "total": -0.26}

_CIRCULARITY_MARK = {
    "overlapping": "†",  # dagger
    "nested": "‡",  # double dagger
}


def _term_row(block: pd.DataFrame, term: str) -> pd.Series | None:
    match = block[block["term"] == term]
    return None if match.empty else match.iloc[0]


def _reason_text(value: object) -> str:
    """A row's ``reason``, or "" when it has none.

    Written out rather than done inline with ``astype(str)`` because that turns a
    missing reason into the string ``"nan"``, which is three characters long and so
    reads as "there was a reason" to any truthiness or length test. That mistake put
    a withheld-proportion footnote on a figure where nothing had been withheld.
    """
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:  # pragma: no cover - non-float, handled below
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _is_true(value: object) -> bool:
    """A recorded boolean flag, tolerant of how it round-trips through CSV.

    ``bool(float("nan"))`` is ``True``, so a flag that was never recorded reads as
    "yes" to the obvious test — which would put a "this depended on ignoring the
    pairing" mark on an estimate where nothing of the kind was found. NaN is checked
    before truthiness for exactly that reason.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(value)


def _mediator_caption(block: pd.DataFrame, labels: dict[str, str] | None) -> str:
    """Row label: the mediator's display name plus any circularity mark."""
    name = str(block["mediator"].iloc[0])
    label = (labels or {}).get(name, name.replace("_", " "))
    grade = block["circularity"].iloc[0]
    mark = _CIRCULARITY_MARK.get(str(grade)) if grade is not None else None
    return f"{label} {mark}" if mark else label


def mediation_forest(
    table: pd.DataFrame,
    *,
    group: str | None = None,
    program_labels: dict[str, str] | None = None,
    title: str | None = None,
    outcome_label: str | None = None,
    effect_label: str = "Effect on outcome score (per-sample units)",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    footnotes: bool = True,
    legend: bool = True,
) -> Figure:
    """
    Draw indirect / direct / total effects per candidate mediator as a forest plot.

    Args:
        table: A :func:`cellquorum.stats.causal_mediation.mediation_effects` or
            ``mediation_grid`` result. One panel row per mediator.
        group: Which ``group`` value to draw. None draws the only group present, and
            raises if the table holds several — silently picking one would publish a
            subtype's result under the cohort's name.
        program_labels: Optional display names for mediator columns.
        outcome_label: Display name for the outcome, used in the default title.
        title: Overrides the default title entirely.
        effect_label: X-axis label. The default names the units rather than the
            formula, since "effect" alone does not say per what.
        figsize: Overrides the height-per-mediator default. Ignored when ``ax`` is given.
        ax: Draw into an existing axes instead of a new figure, for multi-panel figures
            such as a lineage beside its specificity control. The caller then owns the
            layout, so pass ``footnotes=False`` on every panel and place one combined
            block from :func:`mediation_footnotes` — two panels each writing their own
            footnotes at the figure's bottom left would overprint each other.
        footnotes: Draw the recorded caveats under the figure. Only turn this off when
            the caller is placing them itself.
        legend: Draw the three-effect key above the axes. Off for the second and later
            panels of a shared figure, where one key serves all of them.

    Returns:
        The figure the panel was drawn into. Not saved — the caller chooses the path
        and formats (see :func:`cellquorum.visualization.figstyle.save_figure`).

    Raises:
        ValueError: The table is empty, or holds several groups and ``group`` is None.
    """

    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("nothing to draw: the mediation table is empty")

    groups = list(dict.fromkeys(table["group"]))
    if group is None:
        if len(groups) > 1:
            raise ValueError(
                f"the table holds {len(groups)} groups ({groups[:4]}); pass group= to say "
                "which one to draw rather than having one chosen for you"
            )
        group = groups[0]
    panel = table[table["group"] == group]
    if panel.empty:
        raise ValueError(f"no rows for group {group!r}; present groups are {groups}")

    apply_cellquorum_theme()

    mediators = list(dict.fromkeys(panel["mediator"]))
    owns_figure = ax is None
    if ax is None:
        height = max(2.4, 0.85 * len(mediators) + 1.5)
        _, ax = plt.subplots(figsize=figsize or (7.2, height))
    fig = ax.figure

    ax.axvline(0.0, color=_ZERO_LINE, lw=1.0, ls="--", zorder=1)

    y_ticks: list[float] = []
    y_labels: list[str] = []
    seen_terms: list[str] = []
    colors = _term_colors()

    for index, mediator in enumerate(mediators):
        # Top-to-bottom reading order: the first mediator in the table sits highest.
        base = float(len(mediators) - 1 - index)
        block = panel[panel["mediator"] == mediator]
        y_ticks.append(base)
        y_labels.append(_mediator_caption(block, program_labels))

        if not np.isfinite(block["estimate"].astype(float)).any():
            ax.text(
                0.0,
                base,
                "  not estimable",
                va="center",
                ha="left",
                fontsize=7.5,
                color=MUTED,
                style="italic",
            )
            continue

        for term in _PLOTTED_TERMS:
            row = _term_row(block, term)
            if row is None or not np.isfinite(float(row["estimate"])):
                continue
            y = base + _TERM_OFFSET[term]
            style = _TERM_STYLE[term]
            color = colors[term]
            ax.plot(
                [float(row["ci_low"]), float(row["ci_high"])],
                [y, y],
                color=color,
                lw=1.6,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                [float(row["estimate"])],
                [y],
                marker=style["marker"],
                markersize=style["size"],
                color=color,
                markerfacecolor=color if style["filled"] else "white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                linestyle="none",
                zorder=3,
                label=_TERM_LABELS[term] if term not in seen_terms else None,
            )
            seen_terms.append(term)
            if _is_true(row["clustering_changes_the_call"]):
                # A term whose significance turns on whether the donor pairing was
                # respected. Marked on the point, because it is a property of that
                # estimate and not of the mediator.
                ax.plot(
                    [float(row["estimate"])],
                    [y],
                    marker="x",
                    markersize=5.0,
                    color=INK,
                    markeredgewidth=1.2,
                    linestyle="none",
                    zorder=4,
                )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.7, len(mediators) - 0.3)
    ax.set_xlabel(effect_label)

    outcome = outcome_label or str(panel["outcome"].iloc[0]).replace("_", " ")
    if title is None:
        scope = "" if group == "all" else f" — {group}"
        title = f"Mediation of the condition effect on {outcome}{scope}"

    # Header is three stacked bands above the axes — title, the n line, then the key —
    # so nothing lands on the data. The legend in particular cannot go inside: a
    # forest plot's whole right side is where the significant intervals are.
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=34)

    n_donors = int(panel["n_donors"].iloc[0])
    n_samples = int(panel["n_samples"].iloc[0])
    ax.annotate(
        f"{n_samples} samples from {n_donors} donors; 95% CI from donor-clustered bootstrap",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 18),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
        annotation_clip=False,
    )

    if seen_terms and legend:
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(_PLOTTED_TERMS),
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            columnspacing=1.4,
            borderaxespad=0.2,
        )

    if footnotes:
        notes = mediation_footnotes(panel, program_labels=program_labels)
        if notes:
            fig.text(0.01, -0.02, "\n".join(notes), fontsize=7, color=MUTED, va="top")

    apply_cellquorum_axis_style(ax)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if owns_figure:
        # A caller composing panels owns the layout; calling tight_layout from inside
        # each panel would fight whatever the caller did.
        fig.tight_layout()
    return fig


def mediation_sensitivity(
    tables: Mapping[str, pd.DataFrame],
    *,
    term: str = "acme",
    group: str | None = None,
    program_labels: dict[str, str] | None = None,
    title: str | None = None,
    effect_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    footnotes: bool = True,
) -> Figure:
    """
    Draw one effect per mediator once per sensitivity analysis, as stacked intervals.

    The reference analysis goes first and is drawn filled; every later one is hollow,
    so the eye reads "here is the claim, and here is what happened to it". A mediator
    whose interval clears zero in the first row and straddles it in the rest is
    visibly not robust, which is the judgement a table of p-values makes the reader
    do in their head.

    Args:
        tables: Analysis label to mediation table, in the order to draw them. The first
            entry is the reference, and its mediators set the row order — a sensitivity
            analysis run on a subset of mediators simply leaves gaps.
        term: Which decomposed effect to compare. ``"acme"`` (the indirect effect) is
            the one a sensitivity analysis is about; ``"total"`` checks that the effect
            being decomposed survived at all.
        group: Which ``group`` value to read from each table.
        program_labels: Optional display names for mediator columns.
        title: Overrides the default title.
        effect_label: X-axis label; defaults to naming the term.
        figsize: Overrides the default, which grows with mediators × analyses.
        ax: Draw into an existing axes rather than a new figure.
        footnotes: Draw the caveats recorded across all the tables.

    Returns:
        The figure the panel was drawn into.

    Raises:
        ValueError: ``tables`` is empty, or no table holds a finite estimate for ``term``.
    """

    import matplotlib.pyplot as plt

    if not tables:
        raise ValueError("nothing to draw: no sensitivity analyses were given")

    labels = list(tables)
    panels: dict[str, pd.DataFrame] = {}
    for label in labels:
        frame = tables[label]
        selected = frame if group is None else frame[frame["group"] == group]
        panels[label] = selected[selected["term"] == term]

    mediators = list(dict.fromkeys(panels[labels[0]]["mediator"]))
    if not mediators:
        raise ValueError(
            f"the reference analysis {labels[0]!r} has no {term!r} rows"
            + (f" for group {group!r}" if group is not None else "")
        )

    apply_cellquorum_theme()
    owns_figure = ax is None
    if ax is None:
        height = max(2.6, 0.34 * len(mediators) * max(len(labels), 2) + 1.6)
        _, ax = plt.subplots(figsize=figsize or (7.4, height))
    fig = ax.figure

    ax.axvline(0.0, color=_ZERO_LINE, lw=1.0, ls="--", zorder=1)

    colors = _analysis_colors(labels)
    # Intervals are stacked *within* a mediator's band, tightest spacing that still
    # leaves a visible gutter between mediators.
    span = 0.62
    offsets = [0.0] if len(labels) == 1 else list(np.linspace(span / 2, -span / 2, len(labels)))

    y_ticks: list[float] = []
    y_labels: list[str] = []
    seen: set[str] = set()

    for index, mediator in enumerate(mediators):
        base = float(len(mediators) - 1 - index)
        y_ticks.append(base)
        reference_block = panels[labels[0]]
        block = reference_block[reference_block["mediator"] == mediator]
        y_labels.append(_mediator_caption(block, program_labels))

        for label, offset in zip(labels, offsets, strict=True):
            rows = panels[label]
            row_block = rows[rows["mediator"] == mediator]
            if row_block.empty:
                continue
            row = row_block.iloc[0]
            if not np.isfinite(float(row["estimate"])):
                # Refused or withheld here but present elsewhere: say so in place, so a
                # gap is never read as an estimate of zero.
                ax.text(
                    0.0,
                    base + offset,
                    "  n/a",
                    va="center",
                    ha="left",
                    fontsize=6.5,
                    color=MUTED,
                    style="italic",
                )
                continue
            first = label == labels[0]
            color = colors[label]
            ax.plot(
                [float(row["ci_low"]), float(row["ci_high"])],
                [base + offset, base + offset],
                color=color,
                lw=1.5,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                [float(row["estimate"])],
                [base + offset],
                marker="o",
                markersize=6.5 if first else 5.5,
                color=color,
                markerfacecolor=color if first else "white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                linestyle="none",
                zorder=3,
                label=label if label not in seen else None,
            )
            seen.add(label)
            if _is_true(row["clustering_changes_the_call"]):
                ax.plot(
                    [float(row["estimate"])],
                    [base + offset],
                    marker="x",
                    markersize=4.5,
                    color=INK,
                    markeredgewidth=1.1,
                    linestyle="none",
                    zorder=4,
                )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.75, len(mediators) - 0.25)
    ax.set_xlabel(effect_label or f"{_TERM_LABELS.get(term, term)} effect (per-sample units)")

    if title is None:
        title = f"{_TERM_LABELS.get(term, term)} effect under each sensitivity analysis"
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=30)

    if seen:
        # Built in the caller's order rather than left to matplotlib's first-drawn
        # order: an analysis that skipped the topmost mediator would otherwise appear
        # in the key after one drawn below it, and the key is what tells the reader
        # which interval in a band is which.
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                color=colors[label],
                markerfacecolor=colors[label] if label == labels[0] else "white",
                markeredgecolor=colors[label],
                markeredgewidth=1.3,
                markersize=6.5 if label == labels[0] else 5.5,
                linestyle="none",
            )
            for label in labels
            if label in seen
        ]
        ax.legend(
            handles,
            [label for label in labels if label in seen],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=min(len(labels), 3),
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            columnspacing=1.2,
            borderaxespad=0.2,
        )

    if footnotes:
        notes: list[str] = []
        for label in labels:
            for note in mediation_footnotes(panels[label], program_labels=program_labels):
                prefixed = note if note.startswith(("†", "‡", "×")) else f"{label}: {note}"
                if prefixed not in notes:
                    notes.append(prefixed)
        if notes:
            fig.text(0.01, -0.02, "\n".join(notes), fontsize=7, color=MUTED, va="top")

    apply_cellquorum_axis_style(ax)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if owns_figure:
        fig.tight_layout()
    return fig


def _analysis_colors(labels: Sequence[str]) -> dict[str, str]:
    """One colour per sensitivity analysis, from the audited categorical palette."""
    from cellquorum.visualization.figstyle import palette_colors

    picked = palette_colors(max(len(labels), 2))
    return {label: picked[index] for index, label in enumerate(labels)}


def _term_colors() -> dict[str, str]:
    """Colours for the three effects, taken from the audited categorical palette."""
    from cellquorum.visualization.figstyle import palette_colors

    picked = palette_colors(3)
    return {"acme": picked[0], "direct": picked[1], "total": picked[2]}


def mediation_footnotes(
    table: pd.DataFrame,
    *,
    group: str | None = None,
    program_labels: dict[str, str] | None = None,
) -> list[str]:
    """
    The caveats a mediation table records, as the lines that must appear under a figure.

    Derived from the table rather than accumulated while drawing, so a caller composing
    several panels can place one combined block, and so the same caveats appear whether
    the numbers are drawn or written into a legend.

    Args:
        table: A mediation result table.
        group: Which ``group`` to read. None reads every row given.
        program_labels: Optional display names, used in the not-estimable lines.

    Returns:
        Footnote lines, in reading order; empty when the table earned no caveats.
    """
    panel = table if group is None else table[table["group"] == group]
    notes: list[str] = []

    plotted = panel[panel["term"].isin(_PLOTTED_TERMS)]["clustering_changes_the_call"]
    flagged_clustering = any(_is_true(value) for value in plotted)

    refused: list[str] = []
    for mediator in dict.fromkeys(panel["mediator"]):
        block = panel[panel["mediator"] == mediator]
        if not np.isfinite(block["estimate"].astype(float)).any():
            reason = _reason_text(block["reason"].iloc[0]) or "not estimable"
            refused.append(f"{_mediator_caption(block, program_labels)}: {reason}")

    grades = {str(g) for g in panel["circularity"].dropna().unique()}
    if "overlapping" in grades:
        notes.append(
            f"{_CIRCULARITY_MARK['overlapping']} mediator and outcome scores share some "
            "genes, so part of the mediator-outcome path is definitional."
        )
    if "nested" in grades:
        notes.append(
            f"{_CIRCULARITY_MARK['nested']} most of the mediator's genes are also in the "
            "outcome's; the mediated path here is largely definitional."
        )
    if flagged_clustering:
        notes.append(
            "× this term's significance changes if the paired donors are treated as "
            "independent samples; the plotted interval respects the pairing."
        )
    # A withheld proportion is a *specific* pair of facts: the estimate is absent and a
    # reason was recorded for its absence. Testing the reason alone would footnote every
    # figure, including the ones where the proportion is reported.
    proportions = panel[panel["term"] == "proportion_mediated"]
    withheld = [
        row
        for _, row in proportions.iterrows()
        if not np.isfinite(float(row["estimate"])) and _reason_text(row["reason"])
    ]
    if withheld and any(np.isfinite(panel["estimate"].astype(float))):
        notes.append(
            "Proportion mediated is withheld where the total effect's interval crosses "
            "zero; read the indirect effect in its own units instead."
        )
    notes.extend(f"Not estimable — {reason}" for reason in refused)
    return notes


__all__ = ["mediation_footnotes", "mediation_forest", "mediation_sensitivity"]
