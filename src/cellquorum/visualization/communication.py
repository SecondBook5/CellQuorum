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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import apply_cellquorum_axis_style, apply_theme
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

    apply_theme("notebook")
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
    for y, ligand, score in zip(
        positions, drawn["ligand"], drawn["score"].astype(float), strict=True
    ):
        color = _HIGHLIGHT if ligand in highlighted else INK
        ax.plot(
            [0.0, float(score)],
            [float(y)] * 2,
            color=color,
            linewidth=1.1,
            solid_capstyle="butt",
        )
        ax.plot([float(score)], [float(y)], marker="o", markersize=4.4, color=color, zorder=3)

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

    apply_theme("notebook")
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


#: Panel geometry of :func:`ligand_receiver_composite`, in inches. Fixed, so two composites
#: (one per receiver) put the same quantity the same distance from the same edge.
_COMP_ROW_IN = 0.30
_COMP_TARGET_IN = 0.26
_COMP_SENDER_IN = 0.31
_COMP_RANK_IN = 0.36
_COMP_GAP_IN = 0.22
_COMP_LABEL_IN = 1.35


@dataclass(frozen=True)
class _CompositeData:
    """Everything the composite draws, resolved from the input frames."""

    ligands: list[str]
    senders: list[str]
    a_score: dict[str, float]
    a_rank: dict[str, int]
    c_score: dict[str, float]
    c_rank: dict[str, int]
    top_score: float
    expr_scaled: np.ndarray
    target_weights: np.ndarray
    target_names: list[str]
    effect: dict[tuple[str, str], tuple[float, float, int]]


@dataclass(frozen=True)
class _CompositeLayout:
    """Panel positions, in inches, and the resulting figure size."""

    x: dict[str, float]
    panels: list[tuple[str, float]]
    right_edge: float
    fig_w: float
    fig_h: float
    base: float
    top: float


@dataclass(frozen=True)
class _CompositeOptions:
    """How the composite labels and thresholds what it draws."""

    receiver_label: str
    comparison_label: str
    highlight: Sequence[str]
    sender_labels: dict[str, str]
    change_cap: float
    alpha: float


def _composite_data(
    *,
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    targets: pd.DataFrame,
    change: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    highlight: Sequence[str],
    top_n: int,
    max_targets: int,
    ligand_col: str,
    score_col: str,
) -> _CompositeData:
    """Choose the rows and read the five panels' values from the frames."""
    act = activity.copy()
    if "rank" not in act.columns:
        act["rank"] = act[score_col].rank(ascending=False, method="first")
    act = act.sort_values("rank")
    names_ = [str(g) for g in act[ligand_col]]
    a_score = dict(zip(names_, (float(v) for v in act[score_col]), strict=True))
    a_rank = dict(zip(names_, (int(v) for v in act["rank"]), strict=True))
    present = set(expression["ligand"].astype(str))
    ranked = [g for g in names_ if g in present]
    if not ranked:
        raise ValueError("none of the ranked ligands appear in the sender-expression frame")
    ligands = ranked[:top_n]
    ligands += [g for g in highlight if g in a_rank and g in present and g not in ligands]
    ligands = sorted(ligands, key=lambda g: a_rank[g])

    if expression["sender"].isna().any():
        raise ValueError("the sender-expression frame has rows with no sender name")
    senders = list(dict.fromkeys(expression["sender"].astype(str)))
    expr_frame = expression.pivot(index="ligand", columns="sender", values="mean_expression")
    expr = expr_frame.reindex(index=ligands, columns=senders).fillna(0.0).to_numpy(dtype=float)
    peak = expr.max(axis=1, keepdims=True)
    expr_scaled = np.divide(expr, peak, out=np.zeros_like(expr), where=peak > 0)

    weight = targets[targets["ligand"].isin(ligands)].pivot_table(
        index="ligand", columns="target", values="weight", aggfunc="max"
    )
    grid = weight.reindex(ligands).fillna(0.0)
    grid = grid[grid.max(axis=0).sort_values(ascending=False).head(max_targets).index]
    tw = grid.to_numpy(dtype=float)
    best = tw.argmax(axis=0)
    order = sorted(range(tw.shape[1]), key=lambda j: (best[j], -tw[best[j], j]))
    tw = tw[:, order]
    target_names = [str(grid.columns[j]) for j in order]

    c_score: dict[str, float] = {}
    c_rank: dict[str, int] = {}
    if comparison is not None:
        c_names = [str(g) for g in comparison[ligand_col]]
        c_score = dict(zip(c_names, (float(v) for v in comparison[score_col]), strict=True))
        if "rank" in comparison.columns:
            c_rank = dict(zip(c_names, (int(v) for v in comparison["rank"]), strict=True))
    top_score = max([a_score[g] for g in ligands] + list(c_score.values()))

    effect: dict[tuple[str, str], tuple[float, float, int]] = {}
    if change is not None:
        for sender, lig, eff, fdr, donors in zip(
            change["sender"].astype(str),
            change["ligand"].astype(str),
            change["effect"].astype(float),
            change["fdr"].astype(float),
            change["n_donors_concordant"].astype(float),
            strict=True,
        ):
            if np.isfinite(eff):
                effect[(sender, lig)] = (float(eff), float(fdr), int(donors))

    return _CompositeData(
        ligands=ligands,
        senders=senders,
        a_score=a_score,
        a_rank=a_rank,
        c_score=c_score,
        c_rank=c_rank,
        top_score=top_score,
        expr_scaled=expr_scaled,
        target_weights=tw,
        target_names=target_names,
        effect=effect,
    )


def _composite_layout(
    data: _CompositeData, *, has_comparison: bool, has_change: bool
) -> _CompositeLayout:
    n, n_s, n_t = len(data.ligands), len(data.senders), data.target_weights.shape[1]
    panels: list[tuple[str, float]] = [("activity", _COMP_RANK_IN)]
    if has_comparison:
        panels.append(("comparison", _COMP_RANK_IN))
    panels.append(("expression", n_s * _COMP_SENDER_IN))
    if has_change:
        panels.append(("change", n_s * _COMP_SENDER_IN))
    panels.append(("targets", n_t * _COMP_TARGET_IN))
    x: dict[str, float] = {}
    cur = _COMP_LABEL_IN
    for key, width in panels:
        x[key] = cur
        cur += width + _COMP_GAP_IN
    base = 1.25
    return _CompositeLayout(
        x=x,
        panels=panels,
        right_edge=cur,
        fig_w=cur + 1.7,
        fig_h=n * _COMP_ROW_IN + 2.4,
        base=base,
        top=base + n * _COMP_ROW_IN,
    )


class _CompositeDrawer:
    """Draws the composite's five panels, headers and legends on one set of rows."""

    def __init__(
        self,
        data: _CompositeData,
        layout: _CompositeLayout,
        options: _CompositeOptions,
        *,
        has_comparison: bool,
        has_change: bool,
    ) -> None:
        import matplotlib.pyplot as plt

        self.data, self.layout, self.options = data, layout, options
        self.has_comparison, self.has_change = has_comparison, has_change
        self.plt = plt
        self.fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
        self.orange, self.blues = plt.get_cmap("Oranges"), plt.get_cmap("Blues")
        self.purples, self.diverging = plt.get_cmap("Purples"), plt.get_cmap("RdBu_r")
        self.axes = self._add_axes()

    def _add_axes(self) -> dict[str, Any]:
        layout, data = self.layout, self.data
        n, n_s, n_t = len(data.ligands), len(data.senders), data.target_weights.shape[1]
        n_cols = {"activity": 1, "comparison": 1, "expression": n_s, "change": n_s, "targets": n_t}
        axes = {}
        for key, width in layout.panels:
            rect = (
                layout.x[key] / layout.fig_w,
                layout.base / layout.fig_h,
                width / layout.fig_w,
                n * _COMP_ROW_IN / layout.fig_h,
            )
            ax = self.fig.add_axes(rect, label=key)
            ax.set_xlim(0, n_cols[key])
            ax.set_ylim(n, 0)
            ax.axis("off")
            axes[key] = ax
        return axes

    def draw(self) -> Figure:
        for i, g in enumerate(self.data.ligands):
            self._activity_cell(i, g)
            if self.has_comparison:
                self._comparison_cell(i, g)
            for j, s in enumerate(self.data.senders):
                self._expression_cell(i, j, g, s)
                if self.has_change:
                    self._change_cell(i, j, g, s)
            self._target_cells(i, g)
            self._row_label(i, g)
        self._headers_and_titles()
        self._legends()
        return self.fig

    def _activity_cell(self, i: int, g: str) -> None:
        from matplotlib.patches import Rectangle

        frac = self.data.a_score[g] / self.data.top_score
        activity_fill = self.orange(0.06 + 0.94 * frac)
        self.axes["activity"].add_patch(
            Rectangle((0, i), 1, 1, fc=activity_fill, ec="white", lw=1, gid=f"{g}|activity")
        )
        self.axes["activity"].text(
            0.5,
            i + 0.5,
            str(self.data.a_rank[g]),
            ha="center",
            va="center",
            fontsize=6.8,
            color="white" if frac > 0.6 else INK,
        )

    def _comparison_cell(self, i: int, g: str) -> None:
        from matplotlib.patches import Rectangle

        data = self.data
        has = g in data.c_score
        cfrac = data.c_score[g] / data.top_score if has else 0.0
        fill = self.orange(0.06 + 0.94 * cfrac) if has else "#f1f1f1"
        self.axes["comparison"].add_patch(
            Rectangle((0, i), 1, 1, fc=fill, ec="white", lw=1, gid=f"{g}|comparison")
        )
        label = str(data.c_rank[g]) if has and g in data.c_rank else ("" if has else "–")
        txt = self.axes["comparison"].text(
            0.5,
            i + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color=("white" if cfrac > 0.6 else INK) if has else MUTED,
        )
        txt.set_gid(f"comparison_rank|{g}")

    def _expression_cell(self, i: int, j: int, g: str, s: str) -> None:
        from matplotlib.patches import Rectangle

        self.axes["expression"].add_patch(
            Rectangle(
                (j, i),
                1,
                1,
                fc=self.blues(0.04 + 0.96 * self.data.expr_scaled[i, j]),
                ec="white",
                lw=1,
                gid=f"{g}|{s}",
            )
        )

    def _change_cell(self, i: int, j: int, g: str, s: str) -> None:
        from matplotlib.patches import Rectangle

        change_cap = self.options.change_cap
        row = self.data.effect.get((s, g))
        if row is None:
            self.axes["change"].add_patch(
                Rectangle((j, i), 1, 1, fc="#f4f4f4", ec="white", lw=1, gid=f"{g}|{s}")
            )
            return
        eff, fdr, donors = row
        sig = fdr < self.options.alpha
        self.axes["change"].add_patch(
            Rectangle(
                (j, i),
                1,
                1,
                fc=self.diverging(0.5 + 0.5 * float(np.clip(eff / change_cap, -1, 1))),
                ec="#111111" if sig else "white",
                lw=1.6 if sig else 1.0,
                alpha=0.95 if sig else 0.45,
                zorder=3 if sig else 1,
                gid=f"{g}|{s}",
            )
        )
        if sig:
            self.axes["change"].text(
                j + 0.5,
                i + 0.5,
                str(donors),
                ha="center",
                va="center",
                fontsize=6.5,
                zorder=4,
                color="white" if abs(eff) / change_cap > 0.6 else INK,
            )

    def _target_cells(self, i: int, g: str) -> None:
        from matplotlib.patches import Rectangle

        for j, name in enumerate(self.data.target_names):
            v = float(self.data.target_weights[i, j])
            fill = self.purples(0.05 + 0.95 * min(v, 0.3) / 0.3) if v > 0.004 else "#F4F4F4"
            self.axes["targets"].add_patch(
                Rectangle((j, i), 1, 1, fc=fill, ec="white", lw=0.7, gid=f"{g}|{name}")
            )

    def _row_label(self, i: int, g: str) -> None:
        layout = self.layout
        n = len(self.data.ligands)
        label_text = self.fig.text(
            (_COMP_LABEL_IN - 0.1) / layout.fig_w,
            (layout.base + (n - i - 0.5) * _COMP_ROW_IN) / layout.fig_h,
            g,
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold" if g in self.options.highlight else "normal",
            color=INK,
        )
        label_text.set_gid("row_label")

    def _head(self, x0: float, text: str) -> None:
        self.fig.text(
            x0 / self.layout.fig_w,
            (self.layout.top + 0.06) / self.layout.fig_h,
            text,
            rotation=55,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )

    def _title(self, key: str, text: str) -> None:
        self.fig.text(
            self.layout.x[key] / self.layout.fig_w,
            (self.layout.top + 0.95) / self.layout.fig_h,
            text,
            fontsize=9,
            fontweight="bold",
            ha="left",
            color=INK,
        )

    def _headers_and_titles(self) -> None:
        x, layout = self.layout.x, self.layout
        short = self.options.sender_labels
        self._head(x["activity"] + _COMP_RANK_IN / 2, self.options.receiver_label)
        self._title("activity", "Activity (rank)")
        if self.has_comparison:
            self._head(x["comparison"] + _COMP_RANK_IN / 2, self.options.comparison_label)
        for key in [k for k in ("expression", "change") if k in x]:
            for j, s in enumerate(self.data.senders):
                self._head(x[key] + (j + 0.5) * _COMP_SENDER_IN, short.get(s, s))
        self._title("expression", "Expressed by sender")
        if "change" in x:
            self._title("change", "Change in sender, LE − Normal")
        for j, name in enumerate(self.data.target_names):
            self.fig.text(
                (x["targets"] + (j + 0.5) * _COMP_TARGET_IN) / layout.fig_w,
                (layout.base - 0.05) / layout.fig_h,
                name,
                rotation=90,
                ha="center",
                va="top",
                fontsize=7.4,
                color=INK,
            )
        self._title("targets", f"Predicted targets in {self.options.receiver_label} (LE-up genes)")

    def _legends(self) -> None:
        from matplotlib.colors import Normalize

        plt, fig, layout = self.plt, self.fig, self.layout
        change_cap, alpha = self.options.change_cap, self.options.alpha
        lx, ly = layout.right_edge + 0.05, layout.top - 0.35
        legends = [
            ("Oranges", Normalize(0, 1), "Activity\n(relative)"),
            ("Blues", Normalize(0, 1), "Expression\n(scaled per ligand)"),
        ]
        if self.has_change:
            legends.append(
                (
                    "RdBu_r",
                    Normalize(-change_cap, change_cap),
                    f"Δ expression (LE − Normal)\noutline: FDR < {alpha}; number = donors",
                )
            )
        legends.append(("Purples", Normalize(0, 0.3), "Regulatory\npotential"))
        for cmap_name, norm, label in legends:
            cax = fig.add_axes(
                (lx / layout.fig_w, ly / layout.fig_h, 1.2 / layout.fig_w, 0.1 / layout.fig_h)
            )
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap_name), cax=cax, orientation="horizontal"
            )
            cbar.ax.tick_params(labelsize=6.5, length=2)
            for spine in cbar.ax.spines.values():
                spine.set_visible(False)
            fig.text(
                lx / layout.fig_w,
                (ly + 0.17) / layout.fig_h,
                label,
                fontsize=7.2,
                fontweight="bold",
                va="bottom",
            )
            ly -= 0.85


def ligand_receiver_composite(
    *,
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    targets: pd.DataFrame,
    receiver_label: str,
    change: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
    comparison_label: str = "comparison",
    highlight: Sequence[str] = (),
    top_n: int = 15,
    max_targets: int = 32,
    sender_labels: Mapping[str, str] | None = None,
    change_cap: float = 0.4,
    alpha: float = 0.05,
    ligand_col: str = "test_ligand",
    score_col: str = "aupr_corrected",
) -> Figure:
    """One row per ligand: how active, how specific, who sends it, who changed it, its targets.

    The five panels are the chain of reasoning a NicheNet result is read through, and drawing them
    on one set of rows is what keeps a reader from stopping at the first.

    * **Activity** (orange, rank printed in the cell) -- how well the ligand's predicted
    targets explain
      the receiver's response. A property of the receiver.
    * **Activity in the comparison receiver** (same scale, same orange, its own rank) -- is
    the ligand
      a property of this receiver or of the tissue. A ligand the comparison never ranked
      prints ``–``
      and is drawn empty: it was never a candidate there (no cognate receptor expressed), which is a
      stronger dissociation than a low rank and must not be read as rank zero.
    * **Expressed by sender** (blue, scaled per ligand) -- who could have supplied it.
    * **Change in sender, LE - Normal** (diverging) -- who supplies *more* of it in disease.
    Cells whose
      FDR clears ``alpha`` are outlined and carry the number of donors moving the same way;
      the rest are
      faded, because an effect that did not clear its test must not be as saturated as one that did.
      A measurement that does not exist is blank grey, never "no change".
    * **Predicted targets** (purple) -- the receiver's response genes each ligand is predicted to
      regulate, strongest first, grouped under the ligand that regulates each most.

    Args:
        activity: ``nichenet_activities.csv`` of the receiver (ligand, score, ``rank``).
        expression: ``nichenet_sender_expression.csv`` (sender, ligand, ``mean_expression``).
        targets: ligand-target weights restricted to the receiver's response genes
            (ligand, target, weight).
        receiver_label: Name printed on the activity column and the targets title.
        change: Optional per-sender condition effects (sender, ligand, effect, fdr,
            n_donors_concordant).
        comparison: Optional activities of a second receiver, same columns as ``activity``.
        comparison_label: Name of that receiver.
        highlight: Ligands drawn whatever their rank, in bold.
        top_n: Rows taken from the top of the ranking.
        max_targets: Target columns drawn.
        sender_labels: Short names for the sender columns.
        change_cap: Effect at which the diverging colour saturates.
        alpha: FDR below which a change is outlined.

    Raises:
        ValueError: If none of the ranked ligands appear in ``expression``.
    """
    apply_theme("compact")
    data = _composite_data(
        activity=activity,
        expression=expression,
        targets=targets,
        change=change,
        comparison=comparison,
        highlight=highlight,
        top_n=top_n,
        max_targets=max_targets,
        ligand_col=ligand_col,
        score_col=score_col,
    )
    has_comparison, has_change = comparison is not None, change is not None
    layout = _composite_layout(data, has_comparison=has_comparison, has_change=has_change)
    options = _CompositeOptions(
        receiver_label=receiver_label,
        comparison_label=comparison_label,
        highlight=highlight,
        sender_labels=dict(sender_labels or {}),
        change_cap=change_cap,
        alpha=alpha,
    )
    drawer = _CompositeDrawer(
        data, layout, options, has_comparison=has_comparison, has_change=has_change
    )
    return drawer.draw()


#: Signed-chord ribbon colours: red where the weight is higher in the case arm, blue lower.
_CHORD_UP, _CHORD_DOWN, _CHORD_GREY = "#C1121F", "#2A78D2", "#B9BEC4"


@dataclass(frozen=True)
class _ChordColumns:
    sender: str
    ligand: str
    receiver: str
    receptor: str
    weight: str


def _chord_pt(r: float, t: float) -> tuple[float, float]:
    return (r * np.cos(t), r * np.sin(t))


def _chord_spread(mid: list[float], min_gap: float = 0.075) -> list[float]:
    """Push neighbouring angles apart until none is closer than ``min_gap``."""
    idx = np.argsort(mid)
    out = np.array(mid, dtype=float)
    for _ in range(60):
        ordered_angles = out[idx]
        for i in range(1, len(ordered_angles)):
            short = min_gap - (ordered_angles[i] - ordered_angles[i - 1])
            if short > 0:
                ordered_angles[i] += short / 2
                ordered_angles[i - 1] -= short / 2
        out[idx] = ordered_angles
    return list(out)


def _chord_frame(edges: pd.DataFrame, cols: _ChordColumns, top_n: int | None) -> pd.DataFrame:
    """The drawable edges: finite non-zero weights, optionally the strongest ``top_n``."""
    d = edges.copy()
    d[cols.weight] = d[cols.weight].astype(float)
    d = d[np.isfinite(d[cols.weight]) & (d[cols.weight].abs() > 0)]
    if top_n is not None:
        d = d.reindex(d[cols.weight].abs().sort_values(ascending=False).index).head(top_n)
    return d


class _ChordDrawer:
    """Draws one chord: sectors by group, one ribbon per pair, names kept clear of each other."""

    def __init__(
        self,
        d: pd.DataFrame,
        cols: _ChordColumns,
        *,
        sender_palette: Mapping[str, str],
        receiver_palette: Mapping[str, str],
        flags: pd.Series,
        marks: dict[str, str],
        signed: bool,
        ribbon_color: str,
        scale: float | None,
    ) -> None:
        import matplotlib.pyplot as plt

        self.d, self.cols, self.flags, self.marks = d, cols, flags, marks
        self.sender_palette, self.receiver_palette = sender_palette, receiver_palette
        self.signed, self.ribbon_color = signed, ribbon_color

        self.lig_keys = self._order(cols.sender, cols.ligand, sender_palette)
        self.rec_keys = self._order(cols.receiver, cols.receptor, receiver_palette)
        total = float(scale) if scale is not None else float(d["w_abs"].sum())
        self.gap, margin = 0.045, 0.18
        self.unit = (
            np.pi - 2 * margin - self.gap * max(len(self.lig_keys), len(self.rec_keys))
        ) / total
        self.lig_ang = self._angles(self.lig_keys, cols.sender, cols.ligand, np.pi / 2 + margin, 1)
        self.rec_ang = self._angles(
            self.rec_keys, cols.receiver, cols.receptor, np.pi / 2 - margin, -1
        )
        self.fig, self.ax = plt.subplots(figsize=(7.2, 5.6))
        self.ax.set_xlim(-2.1, 2.3)
        self.ax.set_ylim(-1.42, 1.3)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.label_texts: dict[tuple[str, str], list[tuple[float, object]]] = {}
        self.group_items: list[dict] = []
        self.renderer: Any = None

    def _order(
        self, group_col: str, mol_col: str, palette: Mapping[str, str]
    ) -> list[tuple[str, str]]:
        sums = self.d.groupby([group_col, mol_col])["w_abs"].sum()
        groups = sorted(
            {g for g, _ in sums.index},
            key=lambda g: (list(palette).index(g) if g in palette else len(palette), g),
        )
        return [(g, m) for g in groups for m in sums[g].sort_values(ascending=False).index]

    def _angles(
        self, keys: list[tuple[str, str]], col_g: str, col_m: str, start: float, direction: int
    ) -> dict[tuple[str, str], tuple[float, float]]:
        d = self.d
        out, cursor = {}, start
        for g, m in keys:
            w = float(d.loc[(d[col_g] == g) & (d[col_m] == m), "w_abs"].sum()) * self.unit
            out[(g, m)] = (cursor, cursor + direction * w)
            cursor += direction * (w + self.gap)
        return out

    def draw(self) -> Figure:
        self._ribbons()
        self._molecule_labels()
        self._start_renderer()
        self._group_arc(self.lig_keys, self.lig_ang, self.sender_palette, "L")
        self._group_arc(self.rec_keys, self.rec_ang, self.receiver_palette, "R")
        return self.fig

    def finish(self, title: str | None, note: str | None) -> Figure:
        """Title and note, then keep the names off each other and inside the frame."""
        if title:
            self.ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        if note:
            import textwrap

            key = self.ax.text(
                0.0,
                -0.02,
                textwrap.fill(note, width=90),
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=INK,
            )
            key.set_gid("note")
        self._clear_overlaps()
        self._fit_frame()
        self._clear_overlaps()
        self._fit_frame()
        return self.fig

    def _ribbon_colour(self, row: dict) -> str:
        cols, marks = self.cols, self.marks
        if self.signed:
            return _CHORD_UP if row[cols.weight] > 0 else _CHORD_DOWN
        if row[cols.ligand] in marks:
            return marks[row[cols.ligand]]
        if self.ribbon_color == "sender":
            return self.sender_palette.get(row[cols.sender], _CHORD_GREY)
        return _CHORD_GREY

    def _ribbons(self) -> None:
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath

        cols, unit = self.cols, self.unit
        cursor_l = {k: min(v) for k, v in self.lig_ang.items()}
        cursor_r = {k: max(v) for k, v in self.rec_ang.items()}
        ordered = self.d.assign(strong=self.flags).sort_values("strong")
        for row in ordered.to_dict("records"):
            w = row["w_abs"] * unit
            a0 = cursor_l[(row[cols.sender], row[cols.ligand])]
            cursor_l[(row[cols.sender], row[cols.ligand])] = a0 + w
            b1 = cursor_r[(row[cols.receiver], row[cols.receptor])]
            cursor_r[(row[cols.receiver], row[cols.receptor])] = b1 - w
            b0 = b1 - w
            verts = [_chord_pt(0.98, a0)]
            codes = [MplPath.MOVETO]
            for t in np.linspace(a0, a0 + w, 8)[1:]:
                verts.append(_chord_pt(0.98, t))
                codes.append(MplPath.LINETO)
            verts += [(0.0, 0.0), _chord_pt(0.98, b0)]
            codes += [MplPath.CURVE3, MplPath.CURVE3]
            for t in np.linspace(b0, b1, 8)[1:]:
                verts.append(_chord_pt(0.98, t))
                codes.append(MplPath.LINETO)
            verts += [(0.0, 0.0), _chord_pt(0.98, a0)]
            codes += [MplPath.CURVE3, MplPath.CURVE3]
            colour = self._ribbon_colour(row)
            strong = bool(row["strong"])
            self.ax.add_patch(
                PathPatch(
                    MplPath(verts, codes),
                    fc=colour,
                    ec="none",
                    alpha=0.85 if strong else 0.3,
                    zorder=3 if strong else 2,
                    gid=f"ribbon|{row[cols.ligand]}>{row[cols.receptor]}|{row[cols.receiver]}",
                )
            )

    def _molecule_labels(self) -> None:
        marks, ax = self.marks, self.ax
        lig_mid = _chord_spread([sum(self.lig_ang[k]) / 2 for k in self.lig_keys])
        rec_mid = _chord_spread([sum(self.rec_ang[k]) / 2 for k in self.rec_keys])
        for (group, name), t in zip(self.lig_keys, lig_mid, strict=True):
            colour = marks.get(name, INK)
            lab = ax.text(
                *_chord_pt(1.06, t),
                name,
                ha="right",
                va="center",
                fontsize=7.6,
                color=colour,
                fontweight="bold" if name in marks else "normal",
                rotation=np.rad2deg(t) - 180,
                rotation_mode="anchor",
            )
            lab.set_gid("ligand_label")
            self.label_texts.setdefault(("L", group), []).append((t, lab))
        for (group, name), t in zip(self.rec_keys, rec_mid, strict=True):
            lab = ax.text(
                *_chord_pt(1.06, t),
                name,
                ha="left",
                va="center",
                fontsize=7.6,
                color=INK,
                rotation=np.rad2deg(t),
                rotation_mode="anchor",
            )
            lab.set_gid("receptor_label")
            self.label_texts.setdefault(("R", group), []).append((t, lab))

    def _start_renderer(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = self.fig
        self.renderer = (
            fig.canvas.get_renderer()
            if hasattr(fig.canvas, "get_renderer")
            else FigureCanvasAgg(fig).get_renderer()
        )

    def _outer_radius(self, texts: list) -> float:
        """How far from the centre a group's own molecule names reach, measured on the page."""
        ax = self.ax
        ax.apply_aspect()
        radius = 0.0
        for lab in texts:
            box = lab.get_window_extent(self.renderer).transformed(ax.transData.inverted())
            for x in (box.x0, box.x1):
                for y in (box.y0, box.y1):
                    radius = max(radius, float(np.hypot(x, y)))
        return radius

    def _group_arc(
        self,
        keys: list[tuple[str, str]],
        ang: dict[tuple[str, str], tuple[float, float]],
        palette: Mapping[str, str],
        side: str,
    ) -> None:
        from matplotlib.patches import Wedge

        ax = self.ax
        groups = list(dict.fromkeys(k[0] for k in keys))
        spans = {g: [ang[k] for k in keys if k[0] == g] for g in groups}
        lows = {g: min(min(s) for s in spans[g]) for g in groups}
        highs = {g: max(max(s) for s in spans[g]) for g in groups}
        for g in groups:
            ax.add_patch(
                Wedge(
                    (0, 0),
                    1.0,
                    np.rad2deg(lows[g]),
                    np.rad2deg(highs[g]),
                    width=0.03,
                    fc=palette.get(g, "#6B7280"),
                    ec="none",
                )
            )
        # Small neighbouring sectors would print their names on top of each other, so the names
        # are spread along the arc (a leader line would be the next step if it ever matters).
        mids = _chord_spread([(lows[g] + highs[g]) / 2 for g in groups], min_gap=0.24)
        for g, t in zip(groups, mids, strict=True):
            # Beyond the molecule names of its own sector that lie near it on the circle, so the
            # two never overlap; a long name far away along the arc does not push it out.
            near = [lab for a, lab in self.label_texts.get((side, g), []) if abs(a - t) < 0.3]
            reach = max(1.36, self._outer_radius(near) + 0.07)
            lab = ax.text(
                *_chord_pt(reach, t),
                g,
                ha="right" if side == "L" else "left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=palette.get(g, INK),
            )
            lab.set_gid("group_label")
            self.group_items.append({"lab": lab, "angle": t, "reach": reach})

    _NAMES = ("ligand_label", "receptor_label", "group_label")

    def _clear_overlaps(self) -> None:
        """Push each group name outward until it sits on no other name."""
        ax, renderer = self.ax, self.renderer
        ax.apply_aspect()
        for item in self.group_items:
            lab, angle = item["lab"], item["angle"]
            others = [t for t in ax.texts if t.get_gid() in self._NAMES and t is not lab]
            for _ in range(80):
                box = lab.get_window_extent(renderer)
                if not any(box.overlaps(o.get_window_extent(renderer)) for o in others):
                    break
                item["reach"] += 0.05
                lab.set_position(_chord_pt(item["reach"], angle))

    def _fit_frame(self) -> None:
        """Widen the frame to whatever the names reach. Widening zooms the drawing out, which
        makes the names larger in data units, so it is repeated until it stops changing."""
        ax, renderer = self.ax, self.renderer
        for _ in range(5):
            ax.apply_aspect()
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            new = [x_lo, x_hi, y_lo, y_hi]
            for lab in ax.texts:
                if lab.get_gid() not in self._NAMES:
                    continue
                box = lab.get_window_extent(renderer).transformed(ax.transData.inverted())
                new = [
                    min(new[0], box.x0 - 0.05),
                    max(new[1], box.x1 + 0.05),
                    min(new[2], box.y0 - 0.05),
                    max(new[3], box.y1 + 0.05),
                ]
            if np.allclose(new, [x_lo, x_hi, y_lo, y_hi], atol=1e-6):
                return
            ax.set_xlim(new[0], new[1])
            ax.set_ylim(new[2], new[3])


def ligand_receptor_chord(
    edges: pd.DataFrame,
    *,
    sender_palette: Mapping[str, str],
    receiver_palette: Mapping[str, str],
    weight_col: str = "weight",
    significant_col: str | None = None,
    signed: bool = False,
    scale: float | None = None,
    top_n: int | None = None,
    highlight: Mapping[str, str] | None = None,
    ribbon_color: str = "neutral",
    title: str | None = None,
    note: str | None = None,
    sender_col: str = "sender",
    ligand_col: str = "ligand",
    receiver_col: str = "receiver",
    receptor_col: str = "receptor",
) -> Figure:
    """Ligands on the left, grouped by sender; receptors on the right, grouped by receiver.

    One ribbon per ligand-receptor pair, its width on both ends proportional to ``|weight|``.
    Three things keep it honest:

    * ``scale`` is the total weight the half-circle represents. Two chords meant to be compared
      (Normal and lymphedema) must be drawn with the same ``scale``, or the same ribbon is a
      different width in each and the comparison is made by the eye on the wrong quantity.
    * With ``signed=True`` the ribbon is red where the weight is positive and blue where negative;
      its width is the magnitude of the change.
    * With ``significant_col`` given, ribbons whose flag is false are faded, so a change that did
      not clear its test is not drawn as strongly as one that did.

    ``highlight`` maps ligand names to a colour: those ligands' labels (and, when unsigned, their
    ribbons) take it, so a ligand the figure is about can be found without reading every label.
    ``ribbon_color`` is ``"neutral"`` (grey, the default) or ``"sender"``: with ``"sender"`` the
    unsigned ribbons that are not highlighted take their sender's colour from ``sender_palette``,
    so the figure says who is signalling and not only that something is. Signed ribbons are always
    red and blue. ``title`` and ``note`` are drawn as given (the function cannot know which
    condition or contrast the weights describe, nor what the colours stand for in the caller's
    terms): the title above the circle, the note below it.
    """
    if ribbon_color not in ("neutral", "sender"):
        raise ValueError(f"ribbon_color must be 'neutral' or 'sender', got {ribbon_color!r}")
    import matplotlib.pyplot as plt

    apply_theme("compact")
    cols = _ChordColumns(sender_col, ligand_col, receiver_col, receptor_col, weight_col)
    d = _chord_frame(edges, cols, top_n)
    if d.empty:
        fig = plt.figure(figsize=(6, 5.5))
        fig.add_subplot(111).axis("off")
        return fig
    for col in (sender_col, ligand_col, receiver_col, receptor_col):
        d[col] = d[col].astype(str)
    d["w_abs"] = d[weight_col].abs()
    flags = d[significant_col].astype(bool) if significant_col else pd.Series(True, index=d.index)

    drawer = _ChordDrawer(
        d,
        cols,
        sender_palette=sender_palette,
        receiver_palette=receiver_palette,
        flags=flags,
        marks=dict(highlight or {}),
        signed=signed,
        ribbon_color=ribbon_color,
        scale=scale,
    )
    drawer.draw()
    return drawer.finish(title, note)


__all__ = [
    "ligand_activity_arm_comparison",
    "ligand_activity_ranking",
    "ligand_receiver_composite",
    "ligand_receptor_chord",
    "ligand_target_grid",
    "sender_attribution_grid",
]
