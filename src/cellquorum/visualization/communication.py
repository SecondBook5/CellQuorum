"""Ligand-activity figures: what was ranked, how exceptional it is, and who could have sent it.

A NicheNet result is normally published as a bar chart of the top ten ligands' AUPR, which
hides three things, and the third is the one the analysis was usually run to answer.

* **Whether the top ligand is exceptional or merely first.** Every ranking has a top row.
  ``aupr_corrected`` carries no p-value — nichenetr does not compute one — so a bar chart of
  ten bars gives a reader no way to tell a ligand that stands out from the pool from one that
  is a hair above the median of a thousand. :func:`ligand_activity_ranking` draws the top rows
  against the distribution of *every* tested ligand, with the pool's percentiles marked.
* **That activity is a property of the receiver, not of the sender.** The AUPR says how well a
  ligand's predicted targets explain the receiver's response. It says nothing about which cell
  type produced the ligand, and a figure captioned "fibroblast ligands" over a ranking made
  from a pooled ligand pool has asserted something that was never computed.
* **Which cell types actually express it.** That is an expression question with its own answer,
  and :func:`sender_attribution_grid` is that answer: one row per ligand, one column per
  candidate sender, the fraction of that sender's cells expressing it. Cells below the
  detection threshold used to build the ligand pool are drawn as such, so the reader can see
  the threshold rather than take the pool on faith.

:func:`ligand_activity_arm_comparison` asks the specificity question — is a ligand's activity
a property of *this* receiver or of the tissue — and it has a failure mode a pathway comparison
does not. A ligand can be missing from one receiver's ranking because that receiver expresses no
cognate receptor, so it was never a candidate. That is a stronger dissociation than a low score
and in a table it is indistinguishable from one, because the row is simply absent.

:func:`ligand_target_grid` closes the loop on the other side: which of the receiver's response
genes each top ligand is predicted to regulate, which is what makes the ranking checkable at
the gene level rather than on faith in a score.

All three read the frames the NicheNet method writes — ``nichenet_activities.csv``,
``nichenet_sender_expression.csv``, ``nichenet_ligand_target_weights.csv`` — and nothing else,
so a figure can be redrawn from a finished run directory with no recompute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import (
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
)
from cellquorum.visualization.grids import value_grid
from cellquorum.visualization.measured_layout import (
    ABSENT,
    INK,
    MUTED,
    row_panel_canvas,
    widest_label_in,
    write_notes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

#: Width of the ranking panel's data area, in inches. Fixed rather than derived, so an AUPR
#: of 0.15 is the same distance from zero in every panel that draws one — two receivers'
#: rankings side by side are only comparable if that holds.
_RANK_DATA_IN = 2.9

#: Room per ligand row in the ranking panel.
_RANK_ROW_IN = 0.24

#: Percentiles of the whole tested pool marked behind the top rows. The median says where the
#: middle of the pool is; the 95th says where "unusual for this run" starts. Two lines, because
#: one is a threshold a reader will treat as a test and two are visibly a distribution.
_POOL_PERCENTILES = (50.0, 95.0)

#: Colour of the highlighted rows — a ligand the caller named, drawn whether or not it made
#: the cut, so the figure cannot answer "where is TGFB1" by omission.
_HIGHLIGHT = "#B8860B"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """One column as floats, or all-NaN when the frame does not carry it."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _flags(frame: pd.DataFrame, column: str) -> pd.Series:
    """One boolean column, read tolerantly of a CSV round-trip.

    ``bool("FALSE")`` is ``True`` and R writes booleans as text, so the strings are matched
    rather than cast.
    """
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == object:
        return values.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes"})
    return values.fillna(False).astype(bool)


def ligand_activity_ranking(
    activities: pd.DataFrame,
    *,
    top_n: int = 20,
    highlight: Sequence[str] = (),
    ligand_col: str = "test_ligand",
    score_col: str = "aupr_corrected",
    receiver_label: str | None = None,
    score_label: str = "AUPR, corrected for the background rate",
) -> Figure:
    """Top ligands by activity, drawn against the distribution of every ligand tested.

    Args:
        activities: One row per tested ligand, with ``ligand_col`` and ``score_col``.
        top_n: How many rows to draw, best first.
        highlight: Ligands to draw and mark whichever rank they reached. A ranking figure
            that silently omits the ligand a manuscript is about answers the question by
            leaving it out; naming it here forces the figure to show where it landed.
        ligand_col: Column holding the ligand name.
        score_col: Column holding the activity score.
        receiver_label: The cell type whose response was scored, named on the axis. The
            score is a property of that response, and an unnamed axis invites reading it as
            a property of the ligand.
        score_label: What the axis measures. Renaming the column without renaming the axis
            is how a figure comes to be mislabelled.

    Returns:
        The figure.

    Raises:
        ValueError: If the frame lacks the named columns, holds no finite scores, or
            ``top_n`` is not positive.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    for column in (ligand_col, score_col):
        if column not in activities.columns:
            raise ValueError(f"activities has no {column!r} column")

    table = activities[[ligand_col, score_col]].copy()
    table.columns = ["ligand", "score"]
    table["ligand"] = table["ligand"].astype(str)
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    table = table.dropna(subset=["score"])
    if table.empty:
        raise ValueError(f"activities has no finite {score_col!r} values")
    table = table.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    pool = table["score"].to_numpy(dtype=float)
    drawn = table.head(int(top_n)).copy()
    wanted = [str(name) for name in highlight]
    extra = table[table["ligand"].isin(wanted) & ~table["ligand"].isin(drawn["ligand"])]
    if not extra.empty:
        drawn = pd.concat([drawn, extra], ignore_index=True)
    drawn = drawn.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    missing = sorted(set(wanted) - set(table["ligand"]))

    notes = _ranking_notes(
        n_pool=len(table),
        n_drawn=len(drawn),
        below_cut=extra["ligand"].tolist(),
        missing=missing,
        receiver_label=receiver_label,
    )

    apply_cellquorum_theme()
    labels = [f"{row.ligand}  ({row.rank})" for row in drawn.itertuples()]
    fig, ax = row_panel_canvas(
        n_rows=len(drawn),
        label_in=widest_label_in(labels),
        data_in=_RANK_DATA_IN,
        notes=notes,
        row_in=_RANK_ROW_IN,
    )

    highlighted = set(wanted)
    positions = np.arange(len(drawn))[::-1]
    for y, row in zip(positions, drawn.itertuples(), strict=True):
        color = _HIGHLIGHT if row.ligand in highlighted else INK
        ax.plot([0.0, row.score], [y, y], color=color, linewidth=1.1, solid_capstyle="butt")
        ax.plot([row.score], [y], marker="o", markersize=4.4, color=color, zorder=3)

    for percentile in _POOL_PERCENTILES:
        value = float(np.percentile(pool, percentile))
        ax.axvline(value, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        ax.text(
            value,
            len(drawn) - 0.35,
            f"p{percentile:g}",
            fontsize=6.5,
            color=MUTED,
            ha="center",
            va="bottom",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.8, len(drawn) - 0.2)
    ax.set_xlim(min(0.0, float(pool.min())), float(drawn["score"].max()) * 1.12)
    ax.set_xlabel(_activity_axis_label(score_label, receiver_label), fontsize=8.5)
    apply_cellquorum_axis_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    write_notes(fig, notes)
    return fig


def _activity_axis_label(score_label: str, receiver_label: str | None) -> str:
    """Name the quantity and whose response it measures.

    An axis reading "AUPR" alone lets a reader take the score for a property of the ligand.
    It is a property of how well the ligand's predicted targets explain one cell type's
    response, and that cell type belongs on the axis.
    """
    if receiver_label:
        return f"{score_label}\n(response of {receiver_label})"
    return score_label


def _ranking_notes(
    *,
    n_pool: int,
    n_drawn: int,
    below_cut: Sequence[str],
    missing: Sequence[str],
    receiver_label: str | None,
) -> list[str]:
    """What a reader has to be told to read an activity ranking correctly."""
    notes = [
        f"Rows are the top {n_drawn} of {n_pool} ligands tested, best first; the number in "
        "brackets is the rank in the full ranking.",
        "Dashed lines are percentiles of all tested ligands. NicheNet reports no p-value for "
        "ligand activity, so these mark the pool, not a significance threshold.",
    ]
    if receiver_label:
        notes.append(
            f"Activity scores how well a ligand's predicted targets explain {receiver_label}'s "
            "response. It is not evidence that any particular cell type sent the ligand."
        )
    if below_cut:
        notes.append(
            f"{', '.join(sorted(below_cut))} {'is' if len(below_cut) == 1 else 'are'} drawn "
            "despite falling outside the top rows, because the figure was asked for "
            f"{'it' if len(below_cut) == 1 else 'them'} by name."
        )
    if missing:
        notes.append(
            f"{', '.join(missing)} {'was' if len(missing) == 1 else 'were'} not in the tested "
            "ligand pool at all, so no activity was estimated — which is not a low score."
        )
    return notes


def sender_attribution_grid(
    expression: pd.DataFrame,
    *,
    ligands: Sequence[str] | None = None,
    senders: Sequence[str] | None = None,
    expr_prop: float | None = None,
    sender_col: str = "sender",
    ligand_col: str = "ligand",
    value_col: str = "fraction_expressing",
    order_label: str | None = None,
    receiver_label: str | None = None,
) -> Figure:
    """Which candidate senders express each ranked ligand, and in what fraction of their cells.

    Args:
        expression: One row per (sender, ligand), with ``value_col`` and optionally
            ``n_cells`` and ``expressed``.
        ligands: Row order. Defaults to descending total fraction across senders — which is
            *not* the activity ranking, so the default is stated in the notes. Pass the
            activity order when the figure is read against the ranking panel.
        senders: Column order. Defaults to descending total fraction.
        expr_prop: The detection threshold used to build the ligand pool. Cells below it are
            outlined, so the threshold is visible rather than implied.
        sender_col: Column holding the sender name.
        ligand_col: Column holding the ligand name.
        value_col: Column holding the fraction.
        order_label: What put the rows in this order, when the caller supplied them. The
            footnote has to be true of the figure, and "in the order given" is all the
            function can honestly say without it.
        receiver_label: The receiver whose response was scored, when it is *not* among the
            columns. A ligand-activity run excludes the receiver from its own sender pool by
            design, because a cell signalling to itself is a different mechanism — but on the
            page that shows up as one cell type simply missing, indistinguishable from a cell
            type that was tested and expressed nothing. Naming it adds the footnote. Leave it
            ``None`` when the receiver is a column, and the note is not written.

    Returns:
        The figure.

    Raises:
        ValueError: If a required column is missing, the frame is empty, or it holds two rows
            for one (sender, ligand) pair — silently keeping the last would make the figure a
            function of row order.
    """
    for column in (sender_col, ligand_col, value_col):
        if column not in expression.columns:
            raise ValueError(f"expression has no {column!r} column")
    table = expression.copy()
    table[sender_col] = table[sender_col].astype(str)
    table[ligand_col] = table[ligand_col].astype(str)
    table[value_col] = _numeric(table, value_col)
    if table.empty:
        raise ValueError("expression is empty")

    duplicated = table.duplicated(subset=[sender_col, ligand_col], keep=False)
    if duplicated.any():
        pairs = table.loc[duplicated, [sender_col, ligand_col]].drop_duplicates()
        raise ValueError(
            "expression holds more than one row for "
            f"{len(pairs)} (sender, ligand) pair(s), e.g. {tuple(pairs.iloc[0])}"
        )

    wide = table.pivot(index=ligand_col, columns=sender_col, values=value_col)
    supplied_rows = ligands is not None
    supplied_cols = senders is not None
    row_order = _resolve_order(wide.index, ligands, wide.sum(axis=1))
    col_order = _resolve_order(wide.columns, senders, wide.sum(axis=0))
    missing_rows = [str(name) for name in (ligands or []) if str(name) not in set(wide.index)]
    missing_cols = [str(name) for name in (senders or []) if str(name) not in set(wide.columns)]
    wide = wide.reindex(index=row_order, columns=col_order)

    below = None
    if expr_prop is not None:
        below = wide.notna() & (wide < float(expr_prop))
    elif "expressed" in table.columns:
        flags = table.assign(_e=_flags(table, "expressed")).pivot(
            index=ligand_col, columns=sender_col, values="_e"
        )
        aligned = flags.reindex(index=row_order, columns=col_order).astype("object")
        below = wide.notna() & ~aligned.where(aligned.notna(), False).astype(bool)

    counts = None
    if "n_cells" in table.columns:
        counts = (
            table.drop_duplicates(subset=[sender_col])
            .set_index(sender_col)["n_cells"]
            .reindex(col_order)
        )

    notes = _attribution_notes(
        expr_prop=expr_prop,
        marked=int(below.to_numpy().sum()) if below is not None else 0,
        absent=int(wide.isna().to_numpy().sum()),
        counts=counts,
        supplied_rows=supplied_rows,
        supplied_cols=supplied_cols,
        order_label=order_label,
        missing_rows=missing_rows,
        missing_cols=missing_cols,
        excluded_receiver=(
            receiver_label
            if receiver_label is not None and str(receiver_label) not in set(wide.columns)
            else None
        ),
    )
    fig, _ = value_grid(
        wide,
        cmap="Purples",
        vmin=0.0,
        vmax=float(np.nanmax(wide.to_numpy(dtype=float))) if wide.notna().any().any() else 1.0,
        colorbar_label="Fraction of cells expressing",
        notes=notes,
        row_label="Ligand",
        col_label="Candidate sender",
        below=below,
    )
    return fig


def _resolve_order(
    available: pd.Index, requested: Sequence[str] | None, totals: pd.Series
) -> list[str]:
    """The caller's order restricted to what exists, or descending totals."""
    if requested is not None:
        present = set(str(name) for name in available)
        return [str(name) for name in requested if str(name) in present]
    return list(totals.sort_values(ascending=False, kind="mergesort").index)


def _attribution_notes(
    *,
    expr_prop: float | None,
    marked: int,
    absent: int,
    counts: pd.Series | None,
    supplied_rows: bool,
    supplied_cols: bool,
    order_label: str | None,
    missing_rows: Sequence[str],
    missing_cols: Sequence[str],
    excluded_receiver: str | None = None,
) -> list[str]:
    """What a reader has to be told to read a sender-attribution grid correctly."""
    notes = [
        "Colour is expression, not activity: it says which cell types could have sent a "
        "ligand, not that any of them drove the receiver's response."
    ]
    if excluded_receiver is not None:
        notes.append(
            f"{excluded_receiver} is the receiver and is not a column: a cell type signalling "
            "to itself is autocrine, which is a different mechanism, so it is held out of its "
            "own candidate sender pool. Its absence here is that design decision, not a "
            "fraction of zero, and an autocrine reading needs its fraction measured separately."
        )
    if supplied_rows:
        notes.append(
            f"Rows are in the order given, by {order_label}."
            if order_label
            else "Rows are in the order given."
        )
    else:
        notes.append(
            "Rows are ordered by total fraction across senders, largest first — not by "
            "ligand activity."
        )
    if not supplied_cols:
        notes.append("Columns are ordered by total fraction across ligands, largest first.")
    if expr_prop is not None:
        notes.append(
            f"{marked} cell(s) outlined fall below the {expr_prop:g} detection threshold used "
            "to build the candidate ligand pool."
        )
    elif marked:
        notes.append(f"{marked} outlined cell(s) were not counted as expressed.")
    if absent:
        notes.append(
            f"{absent} grey cell(s) have no measurement for that pair, which is not a "
            "fraction of zero."
        )
    if counts is not None and counts.notna().any():
        pairs = ", ".join(f"{name} {int(value):,}" for name, value in counts.dropna().items())
        notes.append(f"Cells per sender: {pairs}.")
    if missing_rows:
        notes.append(
            f"{', '.join(missing_rows)} were asked for but absent from the table, so no row "
            "is drawn for them."
        )
    if missing_cols:
        notes.append(
            f"{', '.join(missing_cols)} were asked for but absent from the table, so no "
            "column is drawn for them."
        )
    return notes


def ligand_activity_arm_comparison(
    activities: Mapping[str, pd.DataFrame],
    *,
    ligands: Sequence[str] | None = None,
    top_n: int | None = 20,
    highlight: Sequence[str] = (),
    ligand_col: str = "test_ligand",
    score_col: str = "aupr_corrected",
    score_label: str = "AUPR, corrected for the background rate",
    order_label: str | None = None,
) -> Figure:
    """One row per ligand, one dot per receiver, ordered by how far the receivers disagree.

    "Signals to lymphatic and not to blood endothelium" is a claim about a difference between
    two rankings, and a reader handed two tables has to hold forty numbers in their head to
    check it. Here the two dots sit on one axis with the gap drawn.

    Two ways a receiver can fail to support a ligand are drawn differently, because the table
    conflates them:

    * **tested and scored low** — a dot near zero, so the reader sees the score was estimated;
    * **never a candidate** — no dot, and a dagger on the row label. NicheNet's candidate pool
      is the ligands with an expressed cognate receptor in that receiver, so an absent ligand
      means the receptor was not detected. That is the strongest form of receptor-level
      dissociation, and it is *not* a low activity.

    Rows are ordered by how far apart the arms are, largest first. Not by either arm's score:
    the top-scoring ligands are usually the ones both arms agree on, and those are the
    tissue-level response, which is what a specificity figure is not about. For a ligand missing
    from an arm the distance is taken as the score it *did* reach, because a receptor-level
    absence is a dissociation at least that large -- so a strong single-arm ligand outranks a
    weak one, and a two-arm gap is not automatically buried beneath every single-arm row. That
    last part is not hypothetical: on a real pool of 450 ligands with 106 single-arm, ordering
    every absence first filled twenty of twenty-three rows with an alphabetical slice of the
    absent tier and left the actual comparison to the three rows ``highlight`` rescued.

    Args:
        activities: One activity table per receiver, keyed by its display name. Drawn in the
            order the mapping gives.
        ligands: Draw exactly these ligands, in this order, overriding the automatic
            selection and ``top_n``.
        top_n: Cap on the automatic selection. ``None`` draws every ligand in any arm.
        highlight: Ligands drawn whatever their spread, and marked. As in
            :func:`ligand_activity_ranking`, so a panel cannot answer "and TGFB1?" by omission.
        ligand_col: Column holding the ligand name.
        score_col: Column holding the activity score.
        score_label: What the axis measures.
        order_label: What put the rows in this order, when ``ligands`` supplies one.

    Returns:
        The figure.

    Raises:
        ValueError: If fewer than two arms are given, an arm lacks the named columns, or no
            arm holds a finite score.
    """
    import matplotlib.pyplot as plt

    arms = [str(name) for name in activities]
    if len(arms) < 2:
        raise ValueError(f"a comparison needs at least two arms, got {arms}")

    scores: dict[str, pd.Series] = {}
    for arm in arms:
        frame = activities[arm]
        for column in (ligand_col, score_col):
            if column not in frame.columns:
                raise ValueError(f"arm {arm!r} has no {column!r} column")
        series = pd.Series(
            pd.to_numeric(frame[score_col], errors="coerce").to_numpy(),
            index=frame[ligand_col].astype(str).to_numpy(),
        )
        scores[arm] = series[~series.index.duplicated(keep="first")].dropna()
    wide = pd.DataFrame(scores)
    if wide.empty:
        raise ValueError(f"no arm holds a finite {score_col!r} value")

    untested = wide.isna().any(axis=1)
    # An arm that never tested a ligand contributes 0 to the distance, not its own minimum over
    # the arms that did: 'absent' is further from a score than any measured value below it.
    floor = wide.min(axis=1).where(~untested, 0.0)
    spread = wide.max(axis=1) - floor
    wanted = [str(name) for name in highlight]
    if ligands is not None:
        present = set(wide.index)
        order = [str(name) for name in ligands if str(name) in present]
        missing = [str(name) for name in ligands if str(name) not in present]
    else:
        order = list(spread.sort_values(ascending=False, kind="mergesort").index)
        if top_n is not None:
            kept = order[: int(top_n)]
            available = set(wide.index)
            order = kept + [n for n in wanted if n in available and n not in kept]
        missing = [name for name in wanted if name not in set(wide.index)]
    drawn = wide.reindex(order)

    notes = _comparison_notes(
        arms=arms,
        n_pool=len(wide),
        n_drawn=len(drawn),
        untested=drawn.isna(),
        supplied=ligands is not None,
        order_label=order_label,
        missing=missing,
    )

    apply_cellquorum_theme()
    labels = [f"{name} †" if drawn.loc[name].isna().any() else str(name) for name in drawn.index]
    fig, ax = row_panel_canvas(
        n_rows=len(drawn),
        label_in=widest_label_in(labels),
        data_in=_RANK_DATA_IN,
        notes=notes,
        row_in=_RANK_ROW_IN,
        top_in=0.52,
    )

    colors = _receiver_colors(arms)
    highlighted = set(wanted)
    positions = np.arange(len(drawn))[::-1]
    for y, name in zip(positions, drawn.index, strict=True):
        values = drawn.loc[name].dropna()
        if len(values) > 1:
            ax.plot(
                [float(values.min()), float(values.max())],
                [y, y],
                color=ABSENT if name not in highlighted else _HIGHLIGHT,
                linewidth=1.4 if name in highlighted else 1.0,
                solid_capstyle="round",
                zorder=1,
            )
        for arm in arms:
            value = drawn.loc[name, arm]
            if pd.isna(value):
                continue
            ax.plot(
                [float(value)],
                [y],
                marker="o",
                markersize=4.6,
                color=colors[arm],
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=colors[arm], label=arm)
        for arm in arms
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=len(arms),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.1,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.8, len(drawn) - 0.2)
    finite = drawn.to_numpy(dtype=float)
    ax.set_xlim(min(0.0, float(np.nanmin(finite))), float(np.nanmax(finite)) * 1.12)
    ax.set_xlabel(score_label, fontsize=8.5)
    apply_cellquorum_axis_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    write_notes(fig, notes)
    return fig


def _receiver_colors(arms: Sequence[str]) -> dict[str, str]:
    """One colour per receiver, matching the two-arm pair the enrichment figures use."""
    from cellquorum.visualization.enrichment import _arm_colors

    return _arm_colors(list(arms))


def _comparison_notes(
    *,
    arms: Sequence[str],
    n_pool: int,
    n_drawn: int,
    untested: pd.DataFrame,
    supplied: bool,
    order_label: str | None,
    missing: Sequence[str],
) -> list[str]:
    """What a reader has to be told to read a two-receiver activity comparison correctly."""
    notes = [
        f"One row per ligand, one dot per receiver ({', '.join(arms)}); the bar is the gap "
        f"between them. {n_drawn} rows drawn of {n_pool} ligands ranked in at least one "
        f"receiver.",
        "NicheNet reports no p-value for ligand activity, so no dot here is filled or open by "
        "significance and no gap between two dots is a tested difference.",
    ]
    if supplied:
        notes.append(
            f"Rows are in the order given, by {order_label}."
            if order_label
            else "Rows are in the order given."
        )
    else:
        notes.append(
            "Rows are ordered by the distance between receivers, largest first — not by either "
            "receiver's score, since the highest scores are usually the ligands both receivers "
            "agree on. For a daggered row the distance is the score it did reach, because an "
            "absent candidate is further away than any measured value."
        )
    for arm in arms:
        names = [str(name) for name in untested.index[untested[arm].to_numpy(dtype=bool)]]
        if names:
            notes.append(
                f"† {', '.join(names)} {'was' if len(names) == 1 else 'were'} never a candidate "
                f"in {arm}: no cognate receptor passed the detection threshold there, so no "
                "activity was estimated. That is a receptor-level absence, not a low score."
            )
    if missing:
        notes.append(
            f"{', '.join(sorted(missing))} {'was' if len(missing) == 1 else 'were'} asked for "
            "but ranked in no receiver, so no row is drawn."
        )
    return notes


def ligand_target_grid(
    weights: pd.DataFrame,
    *,
    ligands: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    max_targets: int | None = 40,
    ligand_col: str = "ligand",
    target_col: str = "target",
    value_col: str = "weight",
    target_groups: Mapping[str, str] | None = None,
    order_label: str | None = None,
) -> Figure:
    """Which of the receiver's response genes each ligand is predicted to regulate.

    This is the panel that makes an activity score checkable: the AUPR is a summary of these
    weights against the gene set, and a reader who can see the genes can judge whether the
    ligand is plausible for the biology or is scoring on a handful of housekeeping targets.

    Args:
        weights: One row per (ligand, target), with ``value_col``.
        ligands: Row order. Defaults to descending total weight.
        targets: Column order. Defaults to descending total weight, capped by ``max_targets``.
        max_targets: Cap on the columns drawn, applied only when ``targets`` is not given. A
            truncated panel is footnoted; an unfootnoted one implies the rest scored zero.
        ligand_col: Column holding the ligand name.
        target_col: Column holding the target gene name.
        value_col: Column holding the regulatory potential.
        target_groups: Optional map from target gene to a group name, appended to the column
            label. Curated-module membership is the usual use, and it is what turns a wall of
            gene symbols into an argument.
        order_label: What put the rows in this order, when the caller supplied them.

    Returns:
        The figure.

    Raises:
        ValueError: If a required column is missing or the frame is empty.
    """
    for column in (ligand_col, target_col, value_col):
        if column not in weights.columns:
            raise ValueError(f"weights has no {column!r} column")
    table = weights.copy()
    table[ligand_col] = table[ligand_col].astype(str)
    table[target_col] = table[target_col].astype(str)
    table[value_col] = _numeric(table, value_col)
    table = table.dropna(subset=[value_col])
    if table.empty:
        raise ValueError("weights has no finite values")

    wide = table.pivot_table(index=ligand_col, columns=target_col, values=value_col, aggfunc="max")
    row_order = _resolve_order(wide.index, ligands, wide.sum(axis=1))
    col_order = _resolve_order(wide.columns, targets, wide.sum(axis=0))
    dropped = 0
    if targets is None and max_targets is not None and len(col_order) > max_targets:
        dropped = len(col_order) - int(max_targets)
        col_order = col_order[: int(max_targets)]
    wide = wide.reindex(index=row_order, columns=col_order)

    notes = [
        "Colour is NicheNet's predicted regulatory potential from the prior model, not a "
        "measured effect in these cells.",
        (
            f"Rows are in the order given, by {order_label}."
            if order_label and ligands is not None
            else "Rows are in the order given."
            if ligands is not None
            else "Rows are ordered by total regulatory potential over the drawn targets."
        ),
    ]
    if dropped:
        notes.append(
            f"{dropped} further target gene(s) are omitted; the table carries all of them."
        )
    absent = int(wide.isna().to_numpy().sum())
    if absent:
        notes.append(
            f"{absent} grey cell(s) mean the prior model links no potential for that pair, "
            "which is not a weight of zero from these data."
        )
    if target_groups:
        labelled = {
            name: f"{name} · {target_groups[name]}" for name in col_order if name in target_groups
        }
        wide = wide.rename(columns=labelled)
        notes.append(
            f"{len(labelled)} of {len(col_order)} target genes carry their curated-module "
            "membership after the gene symbol."
        )

    fig, _ = value_grid(
        wide,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=float(np.nanmax(wide.to_numpy(dtype=float))),
        colorbar_label="Regulatory potential (prior)",
        notes=notes,
        row_label="Ligand",
        col_label="Predicted target in the receiver's response",
        cell_in=0.22,
    )
    return fig


__all__ = [
    "ligand_activity_arm_comparison",
    "ligand_activity_ranking",
    "ligand_target_grid",
    "sender_attribution_grid",
]
