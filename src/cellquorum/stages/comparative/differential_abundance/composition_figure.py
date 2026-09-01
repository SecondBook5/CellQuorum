"""Cell-type composition stacked-bar figures for differential abundance.

Generalizes the project owner's validated composition plots into study-agnostic
engine figures driven by the tidy proportions table from
:func:`build_composition_proportions`. Two figures are produced:

* a condition-level stacked bar (pooled percentages, control vs case), and
* a per-patient stacked bar laid out as control block | divider | case block.

Everything is biology-free: condition labels and colors are arguments, and
cell-type colors come from the CVD-validated categorical palette in figstyle.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.visualization.figstyle import (
    LE_RED,
    NORMAL_BLUE,
    cell_type_palette,
    set_style,
)


def composition_cell_type_order(proportions: pd.DataFrame) -> list[str]:
    """Return cell types ordered by pooled abundance (descending, alpha tie-break).

    The most abundant cell type is drawn at the base of every stacked bar, which
    keeps the dominant populations anchored to the axis across all bars.

    Args:
        proportions: Tidy composition table with ``cell_type`` and ``count``
            columns, as produced by :func:`build_composition_proportions`.

    Returns:
        Cell-type labels, most abundant first; ties broken alphabetically.
    """

    totals = proportions.groupby("cell_type")["count"].sum()
    return sorted((str(ct) for ct in totals.index), key=lambda ct: (-totals[ct], ct))


def pooled_condition_proportions(
    proportions: pd.DataFrame,
    *,
    case: str,
    control: str,
    cell_type_order: list[str] | None = None,
) -> pd.DataFrame:
    """Pool per-sample counts into per-condition composition fractions.

    Sums raw counts within each condition and normalizes so each condition's
    fractions sum to 1 (the pooled "every cell contributes" view, matching the
    Cell Distribution Summary's relative percentages).

    Args:
        proportions: Tidy composition table from
            :func:`build_composition_proportions`.
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control/normal arm.
        cell_type_order: Column order for the returned frame; defaults to
            :func:`composition_cell_type_order`.

    Returns:
        A wide DataFrame indexed by ``[control, case]`` with one column per cell
        type (in ``cell_type_order``) holding pooled fractions in ``[0, 1]``.
    """

    order = cell_type_order or composition_cell_type_order(proportions)
    pooled = proportions.groupby(["condition", "cell_type"])["count"].sum().unstack(fill_value=0)
    totals = pooled.sum(axis=1)
    fractions = pooled.div(totals.where(totals > 0, 1.0), axis=0)
    return fractions.reindex(index=[control, case], columns=order, fill_value=0.0)


def plot_condition_composition(
    proportions: pd.DataFrame,
    *,
    case: str,
    control: str,
    case_color: str = LE_RED,
    control_color: str = NORMAL_BLUE,
    cell_type_order: list[str] | None = None,
    min_label_pct: float = 4.0,
) -> Figure:
    """Stacked composition bar comparing control vs case at the condition level.

    Args:
        proportions: Tidy composition table from
            :func:`build_composition_proportions`.
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control/normal arm.
        case_color: Tick-label color for the case arm.
        control_color: Tick-label color for the control arm.
        cell_type_order: Bottom-to-top segment order; defaults to abundance order.
        min_label_pct: Segments at least this many percent get an in-bar label.

    Returns:
        The composed matplotlib :class:`~matplotlib.figure.Figure`.
    """

    set_style()
    order = cell_type_order or composition_cell_type_order(proportions)
    colors = cell_type_palette(order)
    frac = (
        pooled_condition_proportions(proportions, case=case, control=control, cell_type_order=order)
        * 100.0
    )

    conditions = [control, case]
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    bottoms = np.zeros(len(conditions))
    for ct in order:
        vals = frac[ct].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.62,
            color=colors[ct],
            edgecolor="white",
            linewidth=0.4,
            label=ct,
        )
        for xi, (v, b) in enumerate(zip(vals, bottoms, strict=False)):
            if v >= min_label_pct:
                ax.text(
                    x[xi],
                    b + v / 2.0,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    for tick_label, color in zip(ax.get_xticklabels(), [control_color, case_color], strict=False):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Composition (%)")
    ax.set_title("Cell-type composition by condition")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        title="Cell type",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def plot_per_patient_composition(
    proportions: pd.DataFrame,
    *,
    case: str,
    control: str,
    case_color: str = LE_RED,
    control_color: str = NORMAL_BLUE,
    cell_type_order: list[str] | None = None,
) -> Figure:
    """Per-sample stacked composition, laid out control block | divider | case block.

    The block layout (rather than donor-interleaved) works for paired and
    unpaired cohorts alike, since it never assumes a donor appears in both arms.

    Args:
        proportions: Tidy composition table from
            :func:`build_composition_proportions` (already control-first ordered).
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control/normal arm.
        case_color: Header/label color for the case block.
        control_color: Header/label color for the control block.
        cell_type_order: Bottom-to-top segment order; defaults to abundance order.

    Returns:
        The composed matplotlib :class:`~matplotlib.figure.Figure`.
    """

    set_style()
    order = cell_type_order or composition_cell_type_order(proportions)
    colors = cell_type_palette(order)

    # Preserve the table's control-first sample order.
    samples = list(dict.fromkeys(proportions["sample"].astype(str)))
    per_sample = proportions.drop_duplicates("sample").set_index(proportions["sample"].name)
    donor_of = per_sample["donor"].astype(str)
    cond_of = per_sample["condition"].astype(str)

    wide = (
        proportions.pivot_table(
            index="sample", columns="cell_type", values="proportion", fill_value=0.0
        ).reindex(index=samples, columns=order, fill_value=0.0)
        * 100.0
    )

    x = np.arange(len(samples))
    width = max(6.0, 0.5 * len(samples) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 5.0))
    bottoms = np.zeros(len(samples))
    for ct in order:
        vals = wide[ct].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.82,
            color=colors[ct],
            edgecolor="white",
            linewidth=0.3,
            label=ct,
        )
        bottoms += vals

    # Divider + colored headers between the control and case blocks.
    n_control = int((cond_of.reindex(samples) == control).sum())
    if 0 < n_control < len(samples):
        ax.axvline(n_control - 0.5, ls="--", color="0.4", lw=0.8)
        ax.text(
            (n_control - 1) / 2.0,
            103.0,
            str(control),
            ha="center",
            va="bottom",
            color=control_color,
            fontweight="bold",
        )
        ax.text(
            n_control + (len(samples) - n_control - 1) / 2.0,
            103.0,
            str(case),
            ha="center",
            va="bottom",
            color=case_color,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([donor_of.get(s, s) for s in samples], rotation=90, fontsize=7)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Composition (%)")
    ax.set_title("Cell-type composition per patient")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        title="Cell type",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


__all__ = [
    "composition_cell_type_order",
    "pooled_condition_proportions",
    "plot_condition_composition",
    "plot_per_patient_composition",
]
