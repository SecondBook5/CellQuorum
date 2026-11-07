"""How far apart a donor's two conditions sit, against what shuffled condition labels give.

An optimal-transport (earth mover's) distance between a donor's source-condition cells (the
control group) and the same donor's target-condition cells (the case group) is always positive:
two samples of the same population are never identical, and the distance grows as cell numbers
fall. A bare distance therefore says nothing. Each donor is drawn with the distance the same cells
give after their condition labels are shuffled within the donor (grey: mean to 95th
percentile), so a marker is only read against its own null. A donor whose distance does not
clear that null is an open marker. The donor is the unit; cells are never pooled across donors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellquorum.visualization.figstyle import apply_theme

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

_INK, _MUTED, _ACCENT, _NULL = "#202428", "#7A7A7A", "#C1121F", "#B9BEC4"
_SOURCE = "#3B6FA0"
_REQUIRED = ("state", "donor", "w2", "null_mean", "null_p95", "p_perm")


def transport_distance(
    table: pd.DataFrame,
    *,
    state_order: Sequence[str] | None = None,
    alpha: float = 0.05,
    measure_label: str | None = None,
    source_label: str = "Control",
    target_label: str = "Case",
) -> Figure:
    """One row per donor, grouped by state.

    Args:
        table: One row per donor and state; columns ``state``, ``donor``, ``w2`` (observed
            distance), ``null_mean`` and ``null_p95`` (the same distance after within-donor label
            shuffles), ``p_perm`` (permutation p-value). ``n_source`` and ``n_target`` (cells
            per condition) are written beside each donor when present.
        state_order: Order of the state blocks; defaults to first appearance in ``table``.
        alpha: p-value below which a donor's marker is filled; above it the marker is open.
        measure_label: What ``w2`` measures; the default is the transport distance between the
            two conditions. Any per-donor statistic with a within-donor shuffled-label null can
            be drawn the same way.
        source_label, target_label: Names of the two conditions (control first).
    """
    import matplotlib.pyplot as plt

    missing = [c for c in _REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"transport_distance needs columns {missing}")
    apply_theme("compact")
    states = list(state_order) if state_order is not None else list(dict.fromkeys(table["state"]))
    has_counts = {"n_source", "n_target"} <= set(table.columns)
    if measure_label is None:
        measure_label = f"Wasserstein-2 distance, {source_label} vs {target_label}"

    layout: list[tuple[str, str | None]] = []
    for state in states:
        layout.append((state, None))
        layout.extend((state, str(d)) for d in table.loc[table["state"] == state, "donor"])
    n = len(layout)
    fig, ax = plt.subplots(figsize=(5.4, 0.28 * n + 1.3))
    ax.set_label("transport")
    lo = float(min(table["null_mean"].min(), table["w2"].min()))
    hi = float(max(table["null_p95"].max(), table["w2"].max()))
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad + (0.22 * (hi - lo) if has_counts else 0))
    right = ax.get_xlim()[1]

    labels = []
    for i, (state, donor) in enumerate(layout):
        if donor is None:
            block = table[table["state"] == state]
            above = int((block["p_perm"] < alpha).sum())
            head = ax.text(
                ax.get_xlim()[0],
                i,
                f"{state}  {above} of {len(block)} donors above the shuffled-label null",
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=_INK,
            )
            head.set_gid("state")
            labels.append("")
            continue
        row = table[(table["state"] == state) & (table["donor"].astype(str) == donor)].iloc[0]
        called = float(row["p_perm"]) < alpha
        ax.plot(
            [row["null_mean"], row["null_p95"]],
            [i, i],
            color=_NULL,
            lw=3.2,
            zorder=1,
            solid_capstyle="butt",
        )
        null = ax.scatter([row["null_mean"]], [i], s=16, color=_NULL, zorder=2)
        null.set_gid(f"null|{state}|{donor}")
        pt = ax.scatter(
            [row["w2"]],
            [i],
            s=46,
            facecolors=_ACCENT if called else "none",
            edgecolors=_ACCENT,
            linewidths=1.4,
            zorder=3,
        )
        pt.set_gid(f"obs|{state}|{donor}")
        labels.append(donor)
        if has_counts:
            cells = ax.text(
                right,
                i,
                f"{int(row['n_source'])} | {int(row['n_target'])}",
                ha="right",
                va="center",
                fontsize=7.5,
                color=_MUTED,
            )
            cells.set_gid(f"cells|{state}|{donor}")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    suffix = f"\nright: cells {source_label} | {target_label}" if has_counts else ""
    ax.set_xlabel(
        f"{measure_label}\n" f"(filled: p < {alpha:g}; grey: shuffled-label null){suffix}",
        fontsize=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def transport_displacement(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    plan: np.ndarray,
    *,
    source_value: np.ndarray | None = None,
    target_value: np.ndarray | None = None,
    value_label: str = "Change along the arrow",
    background_xy: np.ndarray | None = None,
    source_label: str = "Control",
    target_label: str = "Case",
) -> Figure:
    """One donor's transport plan drawn as where each source cell is carried to.

    Each source cell is joined to the plan-weighted mean position of the target cells it sends
    its mass to (the barycentric projection of the plan). The arrow is a summary of a soft
    assignment, not a claim that the cell becomes one particular target cell. When values are given
    (a module score, say) each arrow is coloured by the plan-weighted target value minus the
    source value, so the colour reads as the change the transport implies.

    Args:
        source_xy: (n_source, 2) embedding of the source (control) cells.
        target_xy: (n_target, 2) embedding of the target (case) cells.
        plan: (n_source, n_target) transport plan; rows are the mass each source cell sends.
        source_value: Optional per-source-cell value for colouring.
        target_value: Optional per-target-cell value; both or neither value must be given.
        value_label: Colourbar label.
        background_xy: Optional (n, 2) embedding of every cell, drawn faint beneath for
            orientation; it plays no part in the plan.
        source_label, target_label: Names of the two conditions, used in the key.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    apply_theme("compact")
    if plan.shape != (len(source_xy), len(target_xy)):
        raise ValueError(
            f"plan is {plan.shape} but there are {len(source_xy)} {source_label} and "
            f"{len(target_xy)} {target_label} cells"
        )
    if (source_value is None) != (target_value is None):
        raise ValueError("give both source_value and target_value, or neither")
    mass = plan.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError(f"{int((mass <= 0).sum())} {source_label} cells send no mass in the plan")
    weights = plan / mass[:, None]
    ends = weights @ target_xy
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.set_label("displacement")
    if background_xy is not None:
        back = ax.scatter(background_xy[:, 0], background_xy[:, 1], s=4, color="#D9DCE0", zorder=1)
        back.set_gid("background")
    source = ax.scatter(source_xy[:, 0], source_xy[:, 1], s=9, color=_SOURCE, zorder=2)
    source.set_gid("source")
    target = ax.scatter(target_xy[:, 0], target_xy[:, 1], s=9, color=_ACCENT, alpha=0.55, zorder=2)
    target.set_gid("target")
    segments = list(np.stack([source_xy, ends], axis=1))
    arrows = LineCollection(segments, linewidths=0.8, zorder=3)
    arrows.set_gid("arrows")
    if source_value is not None and target_value is not None:
        change = weights @ target_value - source_value
        lim = float(np.abs(change).max()) or 1.0
        arrows.set_array(change)
        arrows.set_cmap("RdBu_r")
        arrows.set_clim(-lim, lim)
        fig.colorbar(arrows, ax=ax, shrink=0.6, pad=0.02).set_label(value_label, fontsize=8.5)
    else:
        arrows.set_color(_INK)
        arrows.set_alpha(0.5)
    ax.add_collection(arrows)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="", color=_SOURCE, label=f"{source_label} cells"
            ),
            Line2D(
                [0], [0], marker="o", linestyle="", color=_ACCENT, label=f"{target_label} cells"
            ),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


__all__ = ["transport_displacement", "transport_distance"]
